package host

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

func TestKernelHalfToFloat32(t *testing.T) {
	for _, tt := range []struct {
		h    uint16
		want float32
	}{
		{0x0000, 0},
		{0x3c00, 1},
		{0xc000, -2},
		{0x3555, 0.333251953125},
		{0x7bff, 65504},
		{0x0400, 1.0 / (1 << 14)},
		{0x0001, 1.0 / (1 << 24)},
		{0x83ff, -1023.0 / (1 << 24)},
		{0x7c00, float32(math.Inf(1))},
		{0xfc00, float32(math.Inf(-1))},
	} {
		if got := kernelHalfToFloat32(tt.h); got != tt.want {
			t.Errorf("%#04x: got %v, want %v", tt.h, got, tt.want)
		}
	}
	if got := kernelHalfToFloat32(0x7e00); !math.IsNaN(float64(got)) {
		t.Errorf("0x7e00: got %v, want NaN", got)
	}
	if got := math.Float32bits(kernelHalfToFloat32(0x8000)); got != 0x80000000 {
		t.Errorf("0x8000: got %#x, want negative zero", got)
	}
}

// TestKernelHalf compares the float16 kernels in use with the generic
// ones. The lengths are not all multiples of a vector, so the tails run.
func TestKernelHalf(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{1, 7, 32, 33, 39, 45, 49, 63, 100, 384, 1500, 1501, 1503, 1505} {
		xs, ys := kernelRandomHalves(rng, n), kernelRandomHalves(rng, n)
		xf, xg, yf := make([]float32, n), make([]float32, n), make([]float32, n)
		kernelHalfRow(xf, xs)
		kernelHalfRowGeneric(xg, xs)
		for i := range xf {
			if math.Float32bits(xf[i]) != math.Float32bits(xg[i]) {
				t.Fatalf("half row, n=%d, [%d]: got %v, want %v", n, i, xf[i], xg[i])
			}
		}
		kernelHalfRow(yf, ys)
		kernelCompare(t, fmt.Sprintf("f32 dot, n=%d", n), kernelDotF32(xf, yf), kernelDotF32Generic(xf, yf))
		cols := [4][]float32{yf, xf, yf, xg}
		got, want := kernelDotF32x4(xf, cols), kernelDotF32x4Generic(xf, cols)
		for c := range got {
			kernelCompare(t, fmt.Sprintf("f32 dot by 4, n=%d, column %d", n, c), got[c], want[c])
		}
		g0, g1 := kernelDotF32x2x4(xf, yf, cols)
		w0, w1 := kernelDotF32x2x4Generic(xf, yf, cols)
		for c := range g0 {
			kernelCompare(t, fmt.Sprintf("2x4 row 0, n=%d, col %d", n, c), g0[c], w0[c])
			kernelCompare(t, fmt.Sprintf("2x4 row 1, n=%d, col %d", n, c), g1[c], w1[c])
		}
	}
}

// TestKernelQuantized compares each quantized type's kernels in use with
// the generic ones, and both with ggml's C formula.
func TestKernelQuantized(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, typ := range []int32{kernelTypeQ4_0, kernelTypeQ4_1, kernelTypeQ5_0, kernelTypeQ5_1, kernelTypeQ8_0} {
		f := kernelFormats[typ]
		decodeGeneric := kernelGenericDecoders[typ]
		yBlock, _ := kernelYValues(f.q8_1)
		for _, n := range []int{32, 64, 384, 1504, 1536} {
			name := fmt.Sprintf("type %d, n=%d", typ, n)
			nb := n / 32
			x := kernelRandomBlocks(rng, nb, f.xBlock, kernelHalfFields(typ))
			var cols [4][]byte
			for c := range cols {
				cols[c] = kernelColumnValues(kernelRandomBlocks(rng, nb, yBlock, kernelHalfFields(map[bool]int32{false: kernelTypeQ8_0, true: -1}[f.q8_1])))
			}

			q, dm := make([]byte, n), make([]float32, 2*nb)
			qg, dmg := make([]byte, n), make([]float32, 2*nb)
			f.decode(q, dm, x)
			decodeGeneric(qg, dmg, x)
			if string(q) != string(qg) {
				t.Fatalf("%s: the unpacked values differ", name)
			}
			for i := range dm {
				if dm[i] != dmg[i] {
					t.Fatalf("%s: scale %d: got %v, want %v", name, i, dm[i], dmg[i])
				}
			}

			yd, ys := make([]float32, nb*kernelScaleStride), make([]float32, nb*kernelScaleStride)
			for c, y := range cols {
				kernelScalesQ8(yd[c:], ys[c:], y, f.q8_1)
			}
			got, want := kernelDotQx4(q, dm, cols, yd, ys, f.q8_1, typ != kernelTypeQ8_0), kernelDotQx4Generic(q, dm, cols, yd, ys, f.q8_1, false)
			x2 := kernelRandomBlocks(rng, nb, f.xBlock, kernelHalfFields(typ))
			q2, dm2 := make([]byte, n), make([]float32, 2*nb)
			f.decode(q2, dm2, x2)
			g0, g1 := kernelDotQ2x4(q, q2, dm, dm2, cols, yd, ys, f.q8_1, typ != kernelTypeQ8_0)
			w1 := kernelDotQx4Generic(q2, dm2, cols, yd, ys, f.q8_1, false)
			for c := range g0 {
				kernelCompare(t, fmt.Sprintf("%s, 2 rows, row 0, column %d", name, c), g0[c], want[c])
				kernelCompare(t, fmt.Sprintf("%s, 2 rows, row 1, column %d", name, c), g1[c], w1[c])
			}
			for c, y := range cols {
				ref := kernelReference(typ, x, y)
				kernelCompare(t, fmt.Sprintf("%s, by 4, column %d", name, c), got[c], want[c])
				kernelCompare(t, fmt.Sprintf("%s, generic, column %d, against C", name, c), want[c], ref)
				one := kernelDotQ(q, dm, y, yd[c:], ys[c:], f.q8_1)
				kernelCompare(t, fmt.Sprintf("%s, one column %d", name, c), one, kernelDotQGeneric(q, dm, y, yd[c:], ys[c:], f.q8_1))
				kernelCompare(t, fmt.Sprintf("%s, one column %d, against C", name, c), one, ref)
			}
		}
	}
}

// TestKernelTile runs gw_mul_mat_tile in a module's memory, above 2 GiB,
// where an address is a negative int32. Columns share rows of x in runs,
// as a block that crosses matrices of a batch does.
func TestKernelTile(t *testing.T) {
	mem, err := NewMemory()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := mem.Close(); err != nil {
			t.Error(err)
		}
	})
	// Wasm's pages are 64 KiB, and its memory at most 65536 of them.
	const base, page, pages = 0x8000_0000 + 0x10000, 1 << 16, 1 << 16
	if mem.Grow(base/page+512-mem.Grow(0, pages), pages) < 0 {
		t.Fatal("cannot grow the memory past 2 GiB")
	}
	k := newKernels(mem)
	b := mem.Bytes()
	rng := rand.New(rand.NewPCG(5, 6))

	for _, types := range [][2]int32{
		{kernelTypeF16, kernelTypeF16}, {kernelTypeF16, kernelTypeF32},
		{kernelTypeQ4_0, kernelTypeQ8_0}, {kernelTypeQ4_1, kernelTypeQ8_1}, {kernelTypeQ5_0, kernelTypeQ8_0},
		{kernelTypeQ5_1, kernelTypeQ8_1}, {kernelTypeQ8_0, kernelTypeQ8_0},
	} {
		typ, ytyp := types[0], types[1]
		for _, shape := range []struct{ n, nr0, nr1 int }{{64, 1, 1}, {64, 17, 3}, {96, 5, 4}, {384, 3, 5}, {64, 64, 16}} {
			name := fmt.Sprintf("types %d and %d, %+v", typ, ytyp, shape)
			xRow, yCol := 2*shape.n, 2*shape.n
			if ytyp == kernelTypeF32 {
				yCol = 4 * shape.n
			}
			if typ != kernelTypeF16 {
				f := kernelFormats[typ]
				yBlock, _ := kernelYValues(f.q8_1)
				xRow, yCol = shape.n/32*f.xBlock, shape.n/32*yBlock
			}
			// Two matrices of x, and the columns use the first, then the
			// second, then the first again.
			at := base
			mats := [2]int{at, at + shape.nr0*xRow}
			at += 2 * shape.nr0 * xRow
			for i := range 2 * shape.nr0 {
				copy(b[mats[0]+i*xRow:], kernelRandomRow(rng, typ, shape.n))
			}
			xsp, ysp, dsp := at, at+64, at+128
			at += 192
			var xs [16]int
			for c := range shape.nr1 {
				xs[c] = mats[min(c*3/max(shape.nr1, 1), 2)%2]
				binary.LittleEndian.PutUint32(b[xsp+4*c:], uint32(xs[c]))
				binary.LittleEndian.PutUint32(b[ysp+4*c:], uint32(at))
				if ytyp == kernelTypeF32 {
					for i := range shape.n {
						binary.LittleEndian.PutUint32(b[at+4*i:], math.Float32bits(rng.Float32()*2-1))
					}
				} else {
					copy(b[at:], kernelRandomColumn(rng, typ, shape.n))
				}
				at += yCol
				binary.LittleEndian.PutUint32(b[dsp+4*c:], uint32(at))
				at += 4 * shape.nr0
			}

			k.Xgw_mul_mat_tile(typ, ytyp, int32(shape.n), int32(shape.nr0), int32(shape.nr1), int32(uint32(xsp)), int32(xRow), int32(uint32(ysp)), int32(uint32(dsp)))

			for c := range shape.nr1 {
				y := int(binary.LittleEndian.Uint32(b[ysp+4*c:]))
				d := int(binary.LittleEndian.Uint32(b[dsp+4*c:]))
				for r := range shape.nr0 {
					row := b[xs[c]+r*xRow:][:xRow]
					got := math.Float32frombits(binary.LittleEndian.Uint32(b[d+4*r:]))
					want := kernelReference(typ, row, b[y:][:yCol])
					if ytyp == kernelTypeF32 {
						want = kernelReferenceF32(row, b[y:][:yCol])
					}
					kernelCompare(t, fmt.Sprintf("%s, column %d, row %d", name, c, r), got, want)
				}
			}
		}
	}
}

// kernelReferenceF32 is kernelReference for a float16 row and a float32
// column.
func kernelReferenceF32(x, y []byte) float32 {
	var sum float64
	for i := 0; i < len(x)/2; i++ {
		sum += float64(kernelHalfAt(x[2*i:])) * float64(math.Float32frombits(binary.LittleEndian.Uint32(y[4*i:])))
	}
	return float32(sum)
}

// kernelGenericDecoders are the generic counterparts of kernelFormats'.
var kernelGenericDecoders = map[int32]func(q []byte, dm []float32, src []byte){
	kernelTypeQ4_0: kernelDecodeQ4_0Generic,
	kernelTypeQ4_1: kernelDecodeQ4_1Generic,
	kernelTypeQ5_0: kernelDecodeQ5_0Generic,
	kernelTypeQ5_1: kernelDecodeQ5_1Generic,
	kernelTypeQ8_0: kernelDecodeQ8_0,
}

// kernelReference is ggml's C for a dot product of a row of x and a column
// of y, summed in float64. The type -1 is not ggml's: it marks q8_1 in
// kernelHalfFields.
func kernelReference(typ int32, x, y []byte) float32 {
	if typ == kernelTypeF16 {
		var sum float64
		for i := 0; i < len(x); i += 2 {
			sum += float64(kernelHalfAt(x[i:])) * float64(kernelHalfAt(y[i:]))
		}
		return float32(sum)
	}
	f := kernelFormats[typ]
	yBlock, yOff := kernelYValues(f.q8_1)
	var sum float64
	for ib := range len(x) / f.xBlock {
		xb, yb := x[ib*f.xBlock:(ib+1)*f.xBlock], y[ib*yBlock:(ib+1)*yBlock]
		var v [32]int
		switch typ {
		case kernelTypeQ4_0, kernelTypeQ4_1:
			qs, sub := xb[2:18], 8
			if typ == kernelTypeQ4_1 {
				qs, sub = xb[4:20], 0
			}
			for j := range 16 {
				v[j], v[j+16] = int(qs[j]&0xf)-sub, int(qs[j]>>4)-sub
			}
		case kernelTypeQ5_0, kernelTypeQ5_1:
			qh, qs, sub := binary.LittleEndian.Uint32(xb[2:]), xb[6:22], 16
			if typ == kernelTypeQ5_1 {
				qh, qs, sub = binary.LittleEndian.Uint32(xb[4:]), xb[8:24], 0
			}
			for j := range 16 {
				v[j] = int(qs[j]&0xf) | int(qh>>j&1)<<4 - sub
				v[j+16] = int(qs[j]>>4) | int(qh>>(j+16)&1)<<4 - sub
			}
		case kernelTypeQ8_0:
			for j := range 32 {
				v[j] = int(int8(xb[2+j]))
			}
		}
		sumi := 0
		for j := range 32 {
			sumi += v[j] * int(int8(yb[yOff+j]))
		}
		sum += float64(sumi) * float64(kernelHalfAt(xb)) * float64(kernelHalfAt(yb))
		if f.q8_1 {
			sum += float64(kernelHalfAt(xb[2:])) * float64(kernelHalfAt(yb[2:]))
		}
	}
	return float32(sum)
}

func kernelCompare(t *testing.T, name string, got, want float32) {
	t.Helper()
	if d := math.Abs(float64(got - want)); d > 1e-4*math.Max(1, math.Abs(float64(want))) {
		t.Errorf("%s: got %v, want %v", name, got, want)
	}
}

// kernelHalfFields is where a block of the type keeps float16 scales:
// d, and the m or s of q4_1, q5_1 and q8_1 (-1).
func kernelHalfFields(typ int32) []int {
	switch typ {
	case kernelTypeQ4_1, kernelTypeQ5_1, -1:
		return []int{0, 2}
	}
	return []int{0}
}

// kernelRandomBlocks is nb blocks of random bytes, with float16 scales of
// the size of weights at the given offsets. Random bits there would make
// NaNs.
func kernelRandomBlocks(rng *rand.Rand, nb, block int, halves []int) []byte {
	b := make([]byte, nb*block)
	for i := range b {
		b[i] = byte(rng.Uint32())
	}
	for ib := range nb {
		for _, off := range halves {
			binary.LittleEndian.PutUint16(b[ib*block+off:], kernelFloat32ToHalf(float32(rng.NormFloat64()*0.01)))
		}
	}
	return b
}

func kernelRandomHalves(rng *rand.Rand, n int) []byte {
	b := make([]byte, 2*n)
	for i := range n {
		binary.LittleEndian.PutUint16(b[2*i:], kernelFloat32ToHalf(float32(rng.NormFloat64())))
	}
	return b
}

// kernelRandomRow and kernelRandomColumn are a row of x and a column of y
// for a tile of the type.
func kernelRandomRow(rng *rand.Rand, typ int32, n int) []byte {
	if typ == kernelTypeF16 {
		return kernelRandomHalves(rng, n)
	}
	return kernelRandomBlocks(rng, n/32, kernelFormats[typ].xBlock, kernelHalfFields(typ))
}

func kernelRandomColumn(rng *rand.Rand, typ int32, n int) []byte {
	if typ == kernelTypeF16 {
		return kernelRandomHalves(rng, n)
	}
	if kernelFormats[typ].q8_1 {
		return kernelColumnValues(kernelRandomBlocks(rng, n/32, kernelBlockQ8_1, kernelHalfFields(-1)))
	}
	return kernelColumnValues(kernelRandomBlocks(rng, n/32, kernelBlockQ8_0, kernelHalfFields(kernelTypeQ8_0)))
}

// kernelColumnValues keeps a column of y within gw_quantize_row's ±127,
// which the vector code counts on: it turns each -128 into -127.
func kernelColumnValues(b []byte) []byte {
	for i, v := range b {
		if v == 0x80 {
			b[i] = 0x81
		}
	}
	return b
}

// kernelFloat32ToHalf rounds f to a nearby float16, for test data. It
// keeps to normal values, which is all the tests need.
func kernelFloat32ToHalf(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int(b>>23&0xff) - 127 + 15
	switch {
	case exp <= 0:
		return sign
	case exp >= 31:
		return sign | 0x7bff
	}
	return sign | uint16(exp)<<10 | uint16(b>>13&0x3ff)
}

// TestKernelNarrow checks the int16 sums of the quantized kernels at the
// limits of q5_1 (x = 31), q5_0 (x = -16) and the q4 types, with y at -128
// and 127, and of q8_0 (x = -128), with y at gw_quantize_row's -127 and 127.
func TestKernelNarrow(t *testing.T) {
	for _, tt := range []struct {
		xv, yv int8
		narrow bool
	}{
		{31, -128, true}, {31, 127, true}, {-16, -128, true}, {-16, 127, true},
		{15, -128, true}, {15, 127, true}, {-8, -128, true}, {-8, 127, true},
		{-128, -127, false}, {-128, 127, false}, {127, 127, false}, {127, -127, false},
	} {
		xv, yv := tt.xv, tt.yv
		for _, q8_1 := range []bool{false, true} {
			nb := 4
			block, off := kernelYValues(q8_1)
			q := make([]byte, 32*nb)
			for i := range q {
				q[i] = byte(xv)
			}
			dm := make([]float32, 2*nb)
			for i := range dm {
				dm[i] = 1
			}
			var cols [4][]byte
			for c := range cols {
				cols[c] = make([]byte, nb*block)
				for ib := range nb {
					for j := range 32 {
						cols[c][ib*block+off+j] = byte(yv)
					}
				}
			}
			yd, ys := make([]float32, nb*kernelScaleStride), make([]float32, nb*kernelScaleStride)
			for i := range yd {
				yd[i], ys[i] = 1, 0
			}
			want := float32(int(xv) * int(yv) * 32 * nb)
			got := kernelDotQx4(q, dm, cols, yd, ys, q8_1, tt.narrow)
			g0, g1 := kernelDotQ2x4(q, q, dm, dm, cols, yd, ys, q8_1, tt.narrow)
			for c := range got {
				if got[c] != want || g0[c] != want || g1[c] != want {
					t.Errorf("x=%d y=%d q8_1=%v: got %v, %v and %v, want %v", xv, yv, q8_1, got[c], g0[c], g1[c], want)
				}
			}
			if one := kernelDotQ(q, dm, cols[0], yd, ys, q8_1); one != want {
				t.Errorf("x=%d y=%d q8_1=%v, one column: got %v, want %v", xv, yv, q8_1, one, want)
			}
		}
	}
}

// TestKernelExpSum compares kernelExpSum, its generic version and
// float64's exp, at odd lengths, with values down to masked ones.
func TestKernelExpSum(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, n := range []int{1, 7, 8, 33, 1500} {
		w := make([]float32, n)
		for i := range w {
			switch rng.IntN(8) {
			case 0:
				w[i] = float32(math.Inf(-1))
			case 1:
				w[i] = -90 - rng.Float32()*10
			default:
				w[i] = rng.Float32()*40 - 30
			}
		}
		w[rng.IntN(n)] = 10
		const maxW = 10
		if n == 33 {
			// NaN stays NaN in both, as in ggml.
			nan := append([]float32{float32(math.NaN())}, w...)
			if s, g := kernelExpSum(nan, maxW), kernelExpSumGeneric(append([]float32(nil), nan...), maxW); !math.IsNaN(s) || !math.IsNaN(g) {
				t.Errorf("a NaN gave sums %v and %v, want NaN", s, g)
			}
		}
		got, gen := append([]float32(nil), w...), append([]float32(nil), w...)
		sum, sumGen := kernelExpSum(got, maxW), kernelExpSumGeneric(gen, maxW)
		var want float64
		for i, v := range w {
			e := math.Exp(float64(v) - maxW)
			want += e
			kernelCompare(t, fmt.Sprintf("n=%d, exp(%v)", n, v), got[i], float32(e))
			kernelCompare(t, fmt.Sprintf("n=%d, generic exp(%v)", n, v), gen[i], float32(e))
		}
		kernelCompare(t, fmt.Sprintf("n=%d, sum", n), float32(sum), float32(want))
		kernelCompare(t, fmt.Sprintf("n=%d, generic sum", n), float32(sumGen), float32(want))
	}
}

// TestKernelSoftMax runs gw_soft_max in place, as ggml may, with each kind
// of mask.
func TestKernelSoftMax(t *testing.T) {
	mem, err := NewMemory()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := mem.Close(); err != nil {
			t.Error(err)
		}
	})
	k := newKernels(mem)
	b := mem.Bytes()
	rng := rand.New(rand.NewPCG(9, 10))
	const n, src, mask = 77, 0x1000, 0x2000
	const scale, slope = 0.125, 0.5
	for maskType := range 3 { // none, float32, float16
		x, m := make([]float32, n), make([]float32, n)
		for i := range n {
			x[i] = rng.Float32()*20 - 10
			binary.LittleEndian.PutUint32(b[src+4*i:], math.Float32bits(x[i]))
			switch maskType {
			case 1:
				m[i] = rng.Float32() * 4
				binary.LittleEndian.PutUint32(b[mask+4*i:], math.Float32bits(m[i]))
			case 2:
				h := kernelRandomColumn(rng, kernelTypeF16, 2)[:2]
				copy(b[mask+2*i:], h)
				m[i] = kernelHalfAt(h)
			}
		}
		maskp, f16 := int32(mask), int32(0)
		switch maskType {
		case 0:
			maskp = 0
		case 2:
			f16 = 1
		}
		k.Xgw_soft_max(n, src, src, maskp, f16, scale, slope)
		want := make([]float64, n)
		var sum float64
		for i := range n {
			want[i] = math.Exp(float64(x[i])*scale + slope*float64(m[i]))
			sum += want[i]
		}
		for i := range n {
			got := math.Float32frombits(binary.LittleEndian.Uint32(b[src+4*i:]))
			kernelCompare(t, fmt.Sprintf("mask %d, value %d", maskType, i), got, float32(want[i]/sum))
		}
	}
}

// TestKernelHalfOf checks the rounding to float16 against every float16:
// each converts back to itself, and a value halfway between two goes to
// the one with an even last bit.
func TestKernelHalfOf(t *testing.T) {
	for h := range 1 << 16 {
		f := kernelHalf[h]
		if math.IsNaN(float64(f)) {
			if got := kernelHalfOf(f); got&0x7C00 != 0x7C00 || got&0x3FF == 0 {
				t.Errorf("NaN %#04x: got %#04x, want a NaN", h, got)
			}
			continue
		}
		if got := kernelHalfOf(f); got != uint16(h) {
			t.Errorf("%v (%#04x): got %#04x", f, h, got)
		}
		next := uint16(h) + 1
		if h&0x7FFF >= 0x7C00 || next&0x7FFF > 0x7C00 || next&0x7FFF == 0 {
			continue // infinities and NaNs have no neighbor to be halfway to
		}
		mid := float32((float64(f) + float64(kernelHalf[next])) / 2)
		want := uint16(h)
		if want&1 != 0 {
			want = next
		}
		if got := kernelHalfOf(mid); got != want {
			t.Errorf("halfway between %#04x and %#04x: got %#04x, want %#04x", h, next, got, want)
		}
	}
}

// TestKernelQuantizeRow quantizes rows at odd addresses, and checks each
// value within half a step of its block, and q8_1's sums.
func TestKernelQuantizeRow(t *testing.T) {
	mem, err := NewMemory()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := mem.Close(); err != nil {
			t.Error(err)
		}
	})
	k := newKernels(mem)
	b := mem.Bytes()
	rng := rand.New(rand.NewPCG(11, 12))
	const n, x, y = 96, 0x1004, 0x3001
	for _, typ := range []int32{kernelTypeQ8_0, kernelTypeQ8_1} {
		v := make([]float32, n)
		for i := range v {
			v[i] = (rng.Float32()*2 - 1) * float32(i%7+1)
			binary.LittleEndian.PutUint32(b[x+4*i:], math.Float32bits(v[i]))
		}
		// A zero, and a block of zeros.
		v[40] = 0
		binary.LittleEndian.PutUint32(b[x+4*40:], 0)
		for i := 64; i < 96; i++ {
			v[i] = 0
			binary.LittleEndian.PutUint32(b[x+4*i:], 0)
		}
		k.Xgw_quantize_row(typ, x, y, n)
		block, off := kernelYValues(typ == kernelTypeQ8_1)
		for ib := range n / 32 {
			yb := b[y+ib*block:][:block]
			d := kernelHalfAt(yb)
			sum := 0
			for j := range 32 {
				q := int8(yb[off+j])
				sum += int(q)
				if got, want := float64(q)*float64(d), float64(v[32*ib+j]); math.Abs(got-want) > float64(d)/2+1e-3*math.Abs(want) {
					t.Errorf("type %d, value %d: %v, want %v", typ, 32*ib+j, got, want)
				}
			}
			// s is float16, of d before its rounding: within half its last
			// bit, at most 2^-11 of it.
			var amax float32
			for _, f := range v[32*ib : 32*ib+32] {
				amax = max(amax, float32(math.Abs(float64(f))))
			}
			if got, want := kernelHalfAt(yb[2:]), float32(sum)*(amax/127); typ == kernelTypeQ8_1 && math.Abs(float64(got-want)) > math.Abs(float64(want))/2048 {
				t.Errorf("block %d: s = %v, want %v", ib, got, want)
			}
		}
	}
}
