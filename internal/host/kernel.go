package host

import (
	"encoding/binary"
	"math"
	"sync"
)

// ggml's type ids, as gw_mul_mat_tile passes them.
const (
	kernelTypeF32  = 0
	kernelTypeF16  = 1
	kernelTypeQ4_0 = 2
	kernelTypeQ4_1 = 3
	kernelTypeQ5_0 = 6
	kernelTypeQ5_1 = 7
	kernelTypeQ8_0 = 8
	kernelTypeQ8_1 = 9
)

// The sizes of ggml's quantized blocks, of 32 values each.
const (
	kernelBlockQ4_0 = 18 // d float16; qs [16]uint8
	kernelBlockQ4_1 = 20 // d, m float16; qs [16]uint8
	kernelBlockQ5_0 = 22 // d float16; qh uint32; qs [16]uint8
	kernelBlockQ5_1 = 24 // d, m float16; qh uint32; qs [16]uint8
	kernelBlockQ8_0 = 34 // d float16; qs [32]int8
	kernelBlockQ8_1 = 36 // d, s float16; qs [32]int8
)

// Kernels runs ggml's matrix products for the module.
// They are most of whisper.cpp's time, and the Go compiler makes faster
// code of them than of wasm2go's translation, which checks the bounds of
// every access.
//
// The kernel functions compute them. With GOEXPERIMENT=simd,
// kernel_arm64.go does it with NEON, and kernel_amd64.go with AVX2 and FMA
// where the CPU has them. Otherwise kernel_generic.go does it in plain Go.
// The sums may differ from ggml's C in their last bits.
type Kernels struct {
	mem *Memory
}

// newKernels returns the kernels over mem.
//
//declscope:shared // env.go builds the env module, whose kernels these are
func newKernels(mem *Memory) *Kernels { return &Kernels{mem: mem} }

// kernelFormat is how a tile unpacks a row of x of one quantized type, and
// the y that type meets.
type kernelFormat struct {
	xBlock int
	decode func(q []byte, dm []float32, src []byte)
	// y is q8_1, with a sum s to scale x's min by, or else q8_0.
	q8_1 bool
}

// kernelFormats are the quantized types of gw_mul_mat_tile.
var kernelFormats = map[int32]kernelFormat{
	kernelTypeQ4_0: {kernelBlockQ4_0, kernelDecodeQ4_0, false},
	kernelTypeQ4_1: {kernelBlockQ4_1, kernelDecodeQ4_1, true},
	kernelTypeQ5_0: {kernelBlockQ5_0, kernelDecodeQ5_0, false},
	kernelTypeQ5_1: {kernelBlockQ5_1, kernelDecodeQ5_1, true},
	kernelTypeQ8_0: {kernelBlockQ8_0, kernelDecodeQ8_0, false},
}

// Xgw_mul_mat_tile computes a block of a matrix product, as ABI.md says.
// Each column of y and each row of x is converted, or unpacked, once for
// the block. The columns of a block share a row of x unless the block
// crosses a matrix of a batch, so each row is most often unpacked once.
// Then each row meets 4 columns at a time, so it is read a quarter as
// often. float32 columns, of float16 rows, are copied as they are.
func (k *Kernels) Xgw_mul_mat_tile(typ, ytyp, n, nr0, nr1, xsp, bx, ysp, dsp int32) {
	b := k.mem.Bytes()
	ptr := func(base int32, i int) int { return int(binary.LittleEndian.Uint32(b[MemoryAddr(base)+4*i:])) }
	cols, nb := int(nr1), int(n)/32
	sc := kernelScratchFor(int(n), cols)
	defer kernelScratches.Put(sc)

	f16 := typ == kernelTypeF16
	f, ok := kernelFormats[typ]
	switch {
	case f16 && (ytyp == kernelTypeF16 || ytyp == kernelTypeF32):
	case ok && ytyp == kernelYType(f.q8_1):
	default:
		panic("host: gw_mul_mat_tile of unknown types")
	}
	yBlock, _ := kernelYValues(f.q8_1)
	var ys [kernelMaxColumns][]byte
	for c := range cols {
		y := ptr(ysp, c)
		switch {
		case f16 && ytyp == kernelTypeF32:
			kernelFloatRow(sc.cols[c], b[y:y+4*int(n)])
			continue
		case f16:
			kernelHalfRow(sc.cols[c], b[y:y+2*int(n)])
			continue
		}
		ys[c] = b[y : y+nb*yBlock]
		kernelScalesQ8(sc.yd[c:], sc.ys[c:], ys[c], f.q8_1)
	}

	for c0 := 0; c0 < cols; {
		x := ptr(xsp, c0)
		c1 := c0 + 1
		for c1 < cols && ptr(xsp, c1) == x {
			c1++
		}
		// What depends on the column only, made once for every row: the
		// groups of 4 columns, and where their sums go.
		var f4 [kernelMaxColumns / 4][4][]float32
		var q4 [kernelMaxColumns / 4][4][]byte
		var out [kernelMaxColumns]int
		for c := c0; c < c1; c++ {
			out[c] = ptr(dsp, c)
		}
		for c := c0; c < c1; c += 4 {
			m := min(4, c1-c)
			if m > 1 {
				if f16 {
					f4[(c-c0)/4] = kernelFour(sc.cols[c : c+m])
				} else {
					q4[(c-c0)/4] = kernelFour(ys[c : c+m])
				}
			}
		}
		r0 := 0
		if f16 {
			// Two rows at a time: each vector of a column is loaded once
			// for both.
			for ; r0+2 <= int(nr0); r0 += 2 {
				row := x + r0*int(bx)
				kernelHalfRow(sc.row, b[row:row+2*int(n)])
				kernelHalfRow(sc.row2, b[row+int(bx):row+int(bx)+2*int(n)])
				for c := c0; c < c1; c += 4 {
					m := min(4, c1-c)
					var s0, s1 [4]float32
					if m == 1 {
						s0[0], s1[0] = kernelDotF32(sc.row, sc.cols[c]), kernelDotF32(sc.row2, sc.cols[c])
					} else {
						s0, s1 = kernelDotF32x2x4(sc.row, sc.row2, f4[(c-c0)/4])
					}
					for j := range m {
						binary.LittleEndian.PutUint64(b[out[c+j]+4*r0:], uint64(math.Float32bits(s0[j]))|uint64(math.Float32bits(s1[j]))<<32)
					}
				}
			}
		}
		if !f16 && kernelQuantPairs {
			for ; r0+2 <= int(nr0); r0 += 2 {
				row := x + r0*int(bx)
				f.decode(sc.q, sc.dm, b[row:row+nb*f.xBlock])
				f.decode(sc.q2, sc.dm2, b[row+int(bx):row+int(bx)+nb*f.xBlock])
				for c := c0; c < c1; c += 4 {
					m := min(4, c1-c)
					var s0, s1 [4]float32
					if m == 1 {
						s0[0] = kernelDotQ(sc.q, sc.dm, ys[c], sc.yd[c:], sc.ys[c:], f.q8_1)
						s1[0] = kernelDotQ(sc.q2, sc.dm2, ys[c], sc.yd[c:], sc.ys[c:], f.q8_1)
					} else {
						s0, s1 = kernelDotQ2x4(sc.q, sc.q2, sc.dm, sc.dm2, q4[(c-c0)/4], sc.yd[c:], sc.ys[c:], f.q8_1, typ != kernelTypeQ8_0)
					}
					for j := range m {
						binary.LittleEndian.PutUint64(b[out[c+j]+4*r0:], uint64(math.Float32bits(s0[j]))|uint64(math.Float32bits(s1[j]))<<32)
					}
				}
			}
		}
		for r := r0; r < int(nr0); r++ {
			row := x + r*int(bx)
			if f16 {
				kernelHalfRow(sc.row, b[row:row+2*int(n)])
			} else {
				f.decode(sc.q, sc.dm, b[row:row+nb*f.xBlock])
			}
			for c := c0; c < c1; c += 4 {
				m := min(4, c1-c)
				var sums [4]float32
				switch {
				case f16 && m == 1:
					sums[0] = kernelDotF32(sc.row, sc.cols[c])
				case f16:
					sums = kernelDotF32x4(sc.row, f4[(c-c0)/4])
				case m == 1:
					sums[0] = kernelDotQ(sc.q, sc.dm, ys[c], sc.yd[c:], sc.ys[c:], f.q8_1)
				default:
					sums = kernelDotQx4(sc.q, sc.dm, q4[(c-c0)/4], sc.yd[c:], sc.ys[c:], f.q8_1, typ != kernelTypeQ8_0)
				}
				for j := range m {
					binary.LittleEndian.PutUint32(b[out[c+j]+4*r:], math.Float32bits(sums[j]))
				}
			}
		}
		c0 = c1
	}
}

// kernelFour fills 4 columns from 2 to 4. A missing column repeats the
// first, and its sum is not written.
func kernelFour[T any](cols []T) [4]T {
	four := [4]T{cols[0], cols[0], cols[0], cols[0]}
	copy(four[:], cols)
	return four
}

// kernelScratch holds a block's columns of y and a row of x, converted or
// unpacked.
type kernelScratch struct {
	cols [kernelMaxColumns][]float32 // the columns of y, as float32
	row  []float32                   // a float16 row of x, as float32
	row2 []float32                   // the next row, for two at a time
	q    []byte                      // a quantized row of x, as one int8 value per byte
	dm   []float32                   // that row's scales: d, and m or 0, of each block
	q2   []byte                      // the next row, for two at a time
	dm2  []float32
	// The quantized columns' d and s. Block ib of column c is at
	// ib*kernelScaleStride + c, so that 4 columns next to each other are
	// one vector.
	yd, ys []float32
}

// kernelScaleStride is the stride of kernelScratch.yd and ys: 64 columns,
// and room for a vector of 4 that starts at the last.
const kernelScaleStride = kernelMaxColumns + 3

// kernelMaxColumns is the most columns of a block, as ABI.md says.
const kernelMaxColumns = 64

// kernelScratches keeps the buffers, since the module asks for hundreds of
// thousands of tiles a second.
var kernelScratches = sync.Pool{New: func() any { return new(kernelScratch) }}

// kernelScratchFor returns a scratch for rows of n values and nr1 columns.
// Put it back in kernelScratches after use.
func kernelScratchFor(n, nr1 int) *kernelScratch {
	sc, ok := kernelScratches.Get().(*kernelScratch)
	if !ok {
		sc = new(kernelScratch)
	}
	nb := n / 32
	if cap(sc.row) < n {
		sc.row = make([]float32, n)
		sc.row2 = make([]float32, n)
		sc.q = make([]byte, n)
		sc.dm = make([]float32, 2*nb)
		sc.q2 = make([]byte, n)
		sc.dm2 = make([]float32, 2*nb)
		sc.yd = make([]float32, nb*kernelScaleStride)
		sc.ys = make([]float32, nb*kernelScaleStride)
	}
	sc.row, sc.row2, sc.q, sc.dm = sc.row[:n], sc.row2[:n], sc.q[:n], sc.dm[:2*nb]
	sc.q2, sc.dm2 = sc.q2[:n], sc.dm2[:2*nb]
	sc.yd, sc.ys = sc.yd[:nb*kernelScaleStride], sc.ys[:nb*kernelScaleStride]
	for c := range nr1 {
		if cap(sc.cols[c]) < n {
			sc.cols[c] = make([]float32, n)
		}
		sc.cols[c] = sc.cols[c][:n]
	}
	return sc
}

// kernelScalesQ8 unpacks the d, and for q8_1 the s, of a column of q8_0 or
// q8_1 blocks, at a stride of kernelScaleStride.
func kernelScalesQ8(yd, ys []float32, y []byte, q8_1 bool) {
	block, _ := kernelYValues(q8_1)
	for ib := range len(y) / block {
		yd[ib*kernelScaleStride] = kernelHalfAt(y[ib*block:])
		if q8_1 {
			ys[ib*kernelScaleStride] = kernelHalfAt(y[ib*block+2:])
		}
	}
}

// kernelYType is the type of y that quantized rows meet: q8_1 or q8_0.
func kernelYType(q8_1 bool) int32 {
	if q8_1 {
		return kernelTypeQ8_1
	}
	return kernelTypeQ8_0
}

// kernelYValues is the size of a q8_0 or q8_1 block, and where it keeps
// its 32 values.
func kernelYValues(q8_1 bool) (block, off int) {
	if q8_1 {
		return kernelBlockQ8_1, 4
	}
	return kernelBlockQ8_0, 2
}

// kernelScaleQ is a block's share of a dot product, from the sum of its
// integer products: d and m are x's, yd and ys y's. The conversions keep
// ggml's rounding: its C fused no multiply with an add.
func kernelScaleQ(d, m, yd, ys float32, sumi int32, q8_1 bool) float32 {
	if !q8_1 {
		return float32(float32(sumi) * float32(d*yd))
	}
	return float32(float32(d*yd)*float32(sumi)) + float32(m*ys)
}

// kernelDecodeQ8_0 copies q8_0 blocks' values, and their d. A copy needs
// no vectors.
func kernelDecodeQ8_0(q []byte, dm []float32, src []byte) {
	for ib := range len(src) / kernelBlockQ8_0 {
		xb := src[ib*kernelBlockQ8_0 : (ib+1)*kernelBlockQ8_0]
		copy(q[32*ib:32*ib+32], xb[2:34])
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), 0
	}
}

// kernelHalf is every float16 as a float32, as ggml's table is.
var kernelHalf = func() *[1 << 16]float32 {
	var t [1 << 16]float32
	for h := range t {
		t[h] = kernelHalfToFloat32(uint16(h))
	}
	return &t
}()

// kernelHalfAt reads the float16 at the start of b.
func kernelHalfAt(b []byte) float32 { return kernelHalf[binary.LittleEndian.Uint16(b)] }

// kernelHalfToFloat32 converts an IEEE 754 binary16 exactly.
func kernelHalfToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	switch {
	case exp == 0x1f: // infinity or NaN
		return math.Float32frombits(sign | 0xff<<23 | frac<<13)
	case exp != 0: // normal
		return math.Float32frombits(sign | (exp+112)<<23 | frac<<13)
	case frac == 0: // zero
		return math.Float32frombits(sign)
	}
	// Subnormal: frac * 2^-24, which float32 holds exactly.
	f := float32(frac) * (1.0 / (1 << 24))
	if sign != 0 {
		f = -f
	}
	return f
}

// Xgw_soft_max is ggml's softmax of a row of n values, as ABI.md says:
// dst = softmax(src*scale + slope*mask), with mask float16 if maskF16 and
// none if mask is 0. src and dst may be the same.
func (k *Kernels) Xgw_soft_max(n, dst, src, mask, maskF16 int32, scale, slope float32) {
	b := k.mem.Bytes()
	sc := kernelScratchFor(int(n), 1)
	defer kernelScratches.Put(sc)
	w, m := sc.row, sc.row2
	kernelFloatRow(w, b[MemoryAddr(src):])
	if mask != 0 {
		if maskF16 != 0 {
			kernelHalfRow(m, b[MemoryAddr(mask):])
		} else {
			kernelFloatRow(m, b[MemoryAddr(mask):])
		}
	}
	var mm []float32
	if mask != 0 {
		mm = m
	}
	sum := kernelExpSum(w, kernelScaleMax(w, mm, scale, slope))
	kernelScaleStore(b[MemoryAddr(dst):], w, float32(1/sum))
}

// Xgw_quantize_row quantizes n values at x to blocks of q8_0 or q8_1 at y,
// as ggml's quantize_row_q8_0_ref and quantize_row_q8_1_ref do, to the bit.
func (k *Kernels) Xgw_quantize_row(typ, x, y, n int32) {
	q8_1 := typ == kernelTypeQ8_1
	if !q8_1 && typ != kernelTypeQ8_0 {
		panic("host: gw_quantize_row of an unknown type")
	}
	b := k.mem.Bytes()
	sc := kernelScratchFor(int(n), 1)
	defer kernelScratches.Put(sc)
	v := sc.row
	kernelFloatRow(v, b[MemoryAddr(x):])
	block, off := kernelYValues(q8_1)
	dst := b[MemoryAddr(y):][:int(n)/32*block]
	for ib := range int(n) / 32 {
		xb, yb := v[32*ib:32*ib+32], dst[ib*block:(ib+1)*block]
		var amax float32
		for _, f := range xb {
			amax = max(amax, float32(math.Abs(float64(f))))
		}
		d := amax / 127
		var id float32
		if d != 0 {
			id = 1 / d
		}
		binary.LittleEndian.PutUint16(yb, kernelHalfOf(d))
		sum := 0
		for j, f := range xb {
			// C's roundf: halves away from zero, as math.Round.
			q := int8(math.Round(float64(f * id)))
			yb[off+j] = byte(q)
			sum += int(q)
		}
		if q8_1 {
			binary.LittleEndian.PutUint16(yb[2:], kernelHalfOf(float32(sum)*d))
		}
	}
}

// kernelHalfOf rounds f to the nearest float16, ties to even, as ggml's
// ggml_compute_fp32_to_fp16 does.
func kernelHalfOf(f float32) uint16 {
	// Separate statements: the Go compiler may fuse x*y + z into one
	// rounding, and C's does not here.
	base := float32(math.Abs(float64(f))) * 0x1p+112
	base *= 0x1p-110
	w := math.Float32bits(f)
	shl1 := w + w
	sign := w & 0x80000000
	bias := max(shl1&0xFF000000, 0x71000000)
	base = math.Float32frombits(bias>>1+0x07800000) + base
	bits := math.Float32bits(base)
	nonsign := bits>>13&0x7C00 + bits&0x0FFF
	if shl1 > 0xFF000000 {
		nonsign = 0x7E00
	}
	return uint16(sign>>16 | nonsign)
}
