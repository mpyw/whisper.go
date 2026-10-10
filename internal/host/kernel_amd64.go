//go:build goexperiment.simd

package host

import (
	"encoding/binary"
	"simd/archsimd"
)

// kernelAVX2 reports whether the CPU runs the vector code. Without AVX2
// and FMA, the generic code runs instead.
var kernelAVX2 = archsimd.X86.AVX2() && archsimd.X86.FMA()

// kernelHalfConverter holds the constants of a float16 conversion, made
// once per call rather than once per vector.
type kernelHalfConverter struct {
	expFrac, sign, infNaN, allOnes archsimd.Uint32x8
	scale                          archsimd.Float32x8
}

func kernelHalfConsts() kernelHalfConverter {
	return kernelHalfConverter{
		expFrac: archsimd.BroadcastUint32x8(0x7fff),
		sign:    archsimd.BroadcastUint32x8(0x8000),
		infNaN:  archsimd.BroadcastUint32x8(0x7c00 << 13),
		allOnes: archsimd.BroadcastUint32x8(0x7f800000),
		scale:   archsimd.BroadcastFloat32x8(0x1p112),
	}
}

// convert converts 8 float16 values, zero-extended to 32 bits, as
// kernel_arm64.go does.
func (c kernelHalfConverter) convert(h archsimd.Uint32x8) archsimd.Float32x8 {
	em := h.And(c.expFrac).ShiftAllLeft(13)
	bits := em.BitsToFloat32().Mul(c.scale).ToBits().Or(h.And(c.sign).ShiftAllLeft(16))
	return bits.Or(c.allOnes.Masked(em.GreaterEqual(c.infNaN))).BitsToFloat32()
}

// kernelDot16 is the products of 16 pairs of int8, added in pairs into 8
// lanes of int32.
func kernelDot16(x, y archsimd.Int8x16) archsimd.Int32x8 {
	return x.ExtendToInt16().DotProductPairs(y.ExtendToInt16())
}

func kernelSum8(v archsimd.Int32x8) int32 {
	var lanes [8]int32
	v.StoreArray(&lanes)
	var sum int32
	for _, x := range lanes {
		sum += x
	}
	return sum
}

// kernelLoad8 loads p, of 8 values. Loads through array pointers have no
// bounds checks of their own: the caller slices once for several.
func kernelLoad8(p []float32) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Array((*[8]float32)(p))
}

// kernelLoadInt8 loads p, of 16 values, as kernelLoad8 does.
func kernelLoadInt8(p []byte) archsimd.Int8x16 {
	return archsimd.LoadUint8x16Array((*[16]byte)(p)).BitsToInt8()
}

// kernelFifthBits spreads the low 16 bits of qh to 16 bytes: 0x10 for a set
// bit, 0 for a clear one.
func kernelFifthBits(qh uint32) archsimd.Uint8x16 {
	var b [16]uint8
	binary.LittleEndian.PutUint64(b[0:], kernelSpread[qh&0xff])
	binary.LittleEndian.PutUint64(b[8:], kernelSpread[qh>>8&0xff])
	return archsimd.LoadUint8x16Array(&b)
}

// kernelDotF32 sums 32 products at a time, in four float32 vectors.
func kernelDotF32(x, y []float32) float32 {
	if !kernelAVX2 {
		return kernelDotF32Generic(x, y)
	}
	n := min(len(x), len(y))
	x, y = x[:n], y[:n]
	var acc0, acc1, acc2, acc3 archsimd.Float32x8
	i := 0
	for ; i+32 <= n; i += 32 {
		xa, ya := (*[32]float32)(x[i:i+32]), (*[32]float32)(y[i:i+32])
		acc0 = kernelLoad8(xa[0:8]).MulAdd(kernelLoad8(ya[0:8]), acc0)
		acc1 = kernelLoad8(xa[8:16]).MulAdd(kernelLoad8(ya[8:16]), acc1)
		acc2 = kernelLoad8(xa[16:24]).MulAdd(kernelLoad8(ya[16:24]), acc2)
		acc3 = kernelLoad8(xa[24:32]).MulAdd(kernelLoad8(ya[24:32]), acc3)
	}
	var lanes [8]float32
	acc0.Add(acc1).Add(acc2.Add(acc3)).StoreArray(&lanes)
	var sum float32
	for _, v := range lanes {
		sum += v
	}
	for ; i < n; i++ {
		sum += x[i] * y[i]
	}
	return sum
}

// kernelFloatRow copies 16 float32 values at a time.
func kernelFloatRow(dst []float32, src []byte) {
	if !kernelAVX2 {
		kernelFloatRowGeneric(dst, src)
		return
	}
	n := len(dst)
	src = src[:4*n]
	i := 0
	for ; i+16 <= n; i += 16 {
		s, d := (*[64]byte)(src[4*i:4*i+64]), (*[16]float32)(dst[i:i+16])
		for k := 0; k < 2; k++ {
			v := archsimd.LoadUint8x32Array((*[32]byte)(s[32*k : 32*k+32])).ReshapeToUint32s().BitsToFloat32()
			v.StoreArray((*[8]float32)(d[8*k : 8*k+8]))
		}
	}
	kernelFloatRowGeneric(dst[i:], src[4*i:])
}

// kernelHalfRow converts 8 float16 values at a time.
func kernelHalfRow(dst []float32, src []byte) {
	if !kernelAVX2 {
		kernelHalfRowGeneric(dst, src)
		return
	}
	c := kernelHalfConsts()
	i := 0
	n := len(dst)
	src = src[:2*n]
	for ; i+8 <= n; i += 8 {
		h := archsimd.LoadUint8x16Array((*[16]byte)(src[2*i : 2*i+16])).ReshapeToUint16s()
		c.convert(h.ExtendToUint32()).StoreArray((*[8]float32)(dst[i : i+8]))
	}
	for ; i < len(dst); i++ {
		dst[i] = kernelHalf[binary.LittleEndian.Uint16(src[2*i:])]
	}
}

// kernelNibbles splits 16 bytes into their low and high 4 bits. x86 has
// no shift of bytes: the high ones shift as 16-bit lanes, then are masked.
func kernelNibbles(qs archsimd.Uint8x16) (lo, hi archsimd.Uint8x16) {
	nibble := archsimd.BroadcastUint8x16(0x0f)
	return qs.And(nibble), qs.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(nibble)
}

// kernelDecodeQ4_0 unpacks q4_0 blocks, as kernelDecodeQ4_0Generic does.
func kernelDecodeQ4_0(q []byte, dm []float32, src []byte) {
	if !kernelAVX2 {
		kernelDecodeQ4_0Generic(q, dm, src)
		return
	}
	eight := archsimd.BroadcastUint8x16(8)
	for ib := range len(src) / kernelBlockQ4_0 {
		xb := src[ib*kernelBlockQ4_0 : (ib+1)*kernelBlockQ4_0]
		lo, hi := kernelNibbles(archsimd.LoadUint8x16(xb[2:18]))
		lo.Sub(eight).Store(q[32*ib:])
		hi.Sub(eight).Store(q[32*ib+16:])
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), 0
	}
}

// kernelDecodeQ4_1 unpacks q4_1 blocks, as kernelDecodeQ4_1Generic does.
func kernelDecodeQ4_1(q []byte, dm []float32, src []byte) {
	if !kernelAVX2 {
		kernelDecodeQ4_1Generic(q, dm, src)
		return
	}
	for ib := range len(src) / kernelBlockQ4_1 {
		xb := src[ib*kernelBlockQ4_1 : (ib+1)*kernelBlockQ4_1]
		lo, hi := kernelNibbles(archsimd.LoadUint8x16(xb[4:20]))
		lo.Store(q[32*ib:])
		hi.Store(q[32*ib+16:])
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), kernelHalfAt(xb[2:])
	}
}

// kernelDecodeQ5_0 unpacks q5_0 blocks, as kernelDecodeQ5_0Generic does.
// The subtraction wraps, which leaves the int8 values.
func kernelDecodeQ5_0(q []byte, dm []float32, src []byte) {
	if !kernelAVX2 {
		kernelDecodeQ5_0Generic(q, dm, src)
		return
	}
	sixteen := archsimd.BroadcastUint8x16(16)
	for ib := range len(src) / kernelBlockQ5_0 {
		xb := src[ib*kernelBlockQ5_0 : (ib+1)*kernelBlockQ5_0]
		qh := binary.LittleEndian.Uint32(xb[2:])
		lo, hi := kernelNibbles(archsimd.LoadUint8x16(xb[6:22]))
		lo.Or(kernelFifthBits(qh)).Sub(sixteen).Store(q[32*ib:])
		hi.Or(kernelFifthBits(qh >> 16)).Sub(sixteen).Store(q[32*ib+16:])
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), 0
	}
}

// kernelDecodeQ5_1 unpacks q5_1 blocks, as kernelDecodeQ5_1Generic does.
func kernelDecodeQ5_1(q []byte, dm []float32, src []byte) {
	if !kernelAVX2 {
		kernelDecodeQ5_1Generic(q, dm, src)
		return
	}
	for ib := range len(src) / kernelBlockQ5_1 {
		xb := src[ib*kernelBlockQ5_1 : (ib+1)*kernelBlockQ5_1]
		qh := binary.LittleEndian.Uint32(xb[4:])
		lo, hi := kernelNibbles(archsimd.LoadUint8x16(xb[8:24]))
		lo.Or(kernelFifthBits(qh)).Store(q[32*ib:])
		hi.Or(kernelFifthBits(qh >> 16)).Store(q[32*ib+16:])
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), kernelHalfAt(xb[2:])
	}
}

// kernelDotQ is kernelDotQGeneric, for one column.
func kernelDotQ(q []byte, dm []float32, y []byte, yd, ys []float32, q8_1 bool) float32 {
	if !kernelAVX2 {
		return kernelDotQGeneric(q, dm, y, yd, ys, q8_1)
	}
	block, off := kernelYValues(q8_1)
	var sum float32
	for ib := range len(y) / block {
		yb := y[ib*block+off : (ib+1)*block]
		x0 := archsimd.LoadUint8x16(q[32*ib:]).BitsToInt8()
		x1 := archsimd.LoadUint8x16(q[32*ib+16:]).BitsToInt8()
		sumi := kernelSum8(kernelDot16(x0, archsimd.LoadUint8x16(yb[0:16]).BitsToInt8()).
			Add(kernelDot16(x1, archsimd.LoadUint8x16(yb[16:32]).BitsToInt8())))
		at := ib * kernelScaleStride
		sum += kernelScaleQ(dm[2*ib], dm[2*ib+1], yd[at], ys[at], sumi, q8_1)
	}
	return sum
}

// kernelDotQx4 multiplies an unpacked row with 4 columns. Each block of the
// row is loaded and widened once, and the 4 columns' sums and scales are
// one vector. Named sums stay in registers, where an array of vectors
// would go through the stack. It ignores narrow, which only
// kernel_arm64.go uses.
func kernelDotQx4(q []byte, dm []float32, cols [4][]byte, yd, ys []float32, q8_1, _ bool) [4]float32 {
	if !kernelAVX2 {
		return kernelDotQx4Generic(q, dm, cols, yd, ys, q8_1, false)
	}
	block, off := kernelYValues(q8_1)
	var acc archsimd.Float32x4
	nb := len(cols[0]) / block
	y0, y1, y2, y3 := cols[0][:nb*block], cols[1][:nb*block], cols[2][:nb*block], cols[3][:nb*block]
	q = q[:32*nb]
	for ib := range nb {
		xq := (*[32]byte)(q[32*ib : 32*ib+32])
		x0, x1 := kernelLoadInt8(xq[0:16]).ExtendToInt16(), kernelLoadInt8(xq[16:32]).ExtendToInt16()
		lo := ib*block + off
		b0, b1, b2, b3 := (*[32]byte)(y0[lo:lo+32]), (*[32]byte)(y1[lo:lo+32]), (*[32]byte)(y2[lo:lo+32]), (*[32]byte)(y3[lo:lo+32])
		s0 := x0.DotProductPairs(kernelLoadInt8(b0[0:16]).ExtendToInt16()).Add(x1.DotProductPairs(kernelLoadInt8(b0[16:32]).ExtendToInt16()))
		s1 := x0.DotProductPairs(kernelLoadInt8(b1[0:16]).ExtendToInt16()).Add(x1.DotProductPairs(kernelLoadInt8(b1[16:32]).ExtendToInt16()))
		s2 := x0.DotProductPairs(kernelLoadInt8(b2[0:16]).ExtendToInt16()).Add(x1.DotProductPairs(kernelLoadInt8(b2[16:32]).ExtendToInt16()))
		s3 := x0.DotProductPairs(kernelLoadInt8(b3[0:16]).ExtendToInt16()).Add(x1.DotProductPairs(kernelLoadInt8(b3[16:32]).ExtendToInt16()))
		// Pairs twice, within each 128 bits: then the low half holds the
		// sums of each column's low lanes, and the high half of its high
		// ones. Lane c of their sum is the sum of column c's.
		p := s0.ConcatAddPairsGrouped(s1).ConcatAddPairsGrouped(s2.ConcatAddPairsGrouped(s3))
		sumi := p.GetLo().Add(p.GetHi()).ConvertToFloat32()
		at := ib * kernelScaleStride
		d := archsimd.BroadcastFloat32x4(dm[2*ib]).Mul(archsimd.LoadFloat32x4(yd[at:]))
		part := sumi.Mul(d)
		if q8_1 {
			part = part.Add(archsimd.BroadcastFloat32x4(dm[2*ib+1]).Mul(archsimd.LoadFloat32x4(ys[at:])))
		}
		acc = acc.Add(part)
	}
	var sums [4]float32
	acc.StoreArray(&sums)
	return sums
}

// kernelDotF32x4 is kernelDotF32 with 4 columns: each vector of x is loaded
// once for all 4. Two vectors a step keep 8 sums in flight, since each
// waits for the last multiply-add of its own.
func kernelDotF32x4(x []float32, ys [4][]float32) [4]float32 {
	if !kernelAVX2 {
		return kernelDotF32x4Generic(x, ys)
	}
	n := len(x)
	y0, y1, y2, y3 := ys[0][:n], ys[1][:n], ys[2][:n], ys[3][:n]
	// Named, not an array: an array of vectors goes through the stack.
	var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Float32x8
	i := 0
	for ; i+16 <= n; i += 16 {
		xx := (*[16]float32)(x[i : i+16])
		q0, q1, q2, q3 := (*[16]float32)(y0[i:i+16]), (*[16]float32)(y1[i:i+16]), (*[16]float32)(y2[i:i+16]), (*[16]float32)(y3[i:i+16])
		xa, xb := kernelLoad8(xx[0:8]), kernelLoad8(xx[8:16])
		a0 = xa.MulAdd(kernelLoad8(q0[0:8]), a0)
		a1 = xa.MulAdd(kernelLoad8(q1[0:8]), a1)
		a2 = xa.MulAdd(kernelLoad8(q2[0:8]), a2)
		a3 = xa.MulAdd(kernelLoad8(q3[0:8]), a3)
		a4 = xb.MulAdd(kernelLoad8(q0[8:16]), a4)
		a5 = xb.MulAdd(kernelLoad8(q1[8:16]), a5)
		a6 = xb.MulAdd(kernelLoad8(q2[8:16]), a6)
		a7 = xb.MulAdd(kernelLoad8(q3[8:16]), a7)
	}
	var sums [4]float32
	acc := [4]archsimd.Float32x8{a0.Add(a4), a1.Add(a5), a2.Add(a6), a3.Add(a7)}
	for c, y := range [4][]float32{y0, y1, y2, y3} {
		var lanes [8]float32
		acc[c].StoreArray(&lanes)
		for _, v := range lanes {
			sums[c] += v
		}
		for j := i; j < n; j++ {
			sums[c] += x[j] * y[j]
		}
	}
	return sums
}

// kernelDotF32x2x4 is kernelDotF32x4 for 2 rows: each vector of a column
// is loaded once for both. One vector a step keeps the 8 sums, the 2
// rows and the 4 columns in 14 of the 16 registers.
func kernelDotF32x2x4(x0, x1 []float32, ys [4][]float32) (r0, r1 [4]float32) {
	if !kernelAVX2 {
		return kernelDotF32x2x4Generic(x0, x1, ys)
	}
	n := len(x0)
	x1 = x1[:n]
	y0, y1, y2, y3 := ys[0][:n], ys[1][:n], ys[2][:n], ys[3][:n]
	var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Float32x8
	i := 0
	for ; i+8 <= n; i += 8 {
		xa, xb := kernelLoad8(x0[i:i+8]), kernelLoad8(x1[i:i+8])
		c0, c1, c2, c3 := kernelLoad8(y0[i:i+8]), kernelLoad8(y1[i:i+8]), kernelLoad8(y2[i:i+8]), kernelLoad8(y3[i:i+8])
		a0 = xa.MulAdd(c0, a0)
		a1 = xa.MulAdd(c1, a1)
		a2 = xa.MulAdd(c2, a2)
		a3 = xa.MulAdd(c3, a3)
		a4 = xb.MulAdd(c0, a4)
		a5 = xb.MulAdd(c1, a5)
		a6 = xb.MulAdd(c2, a6)
		a7 = xb.MulAdd(c3, a7)
	}
	acc := [8]archsimd.Float32x8{a0, a1, a2, a3, a4, a5, a6, a7}
	for c, y := range [4][]float32{y0, y1, y2, y3} {
		var l0, l1 [8]float32
		acc[c].StoreArray(&l0)
		acc[c+4].StoreArray(&l1)
		for k := range l0 {
			r0[c] += l0[k]
			r1[c] += l1[k]
		}
		for j := i; j < n; j++ {
			r0[c] += x0[j] * y[j]
			r1[c] += x1[j] * y[j]
		}
	}
	return r0, r1
}

// kernelExpSum is kernelExpSumGeneric, 8 values at a time.
func kernelExpSum(w []float32, max float32) float64 {
	if !kernelAVX2 {
		return kernelExpSumGeneric(w, max)
	}
	var (
		r     = archsimd.BroadcastFloat32x8(0x1.8p23)
		log2e = archsimd.BroadcastFloat32x8(0x1.715476p+0)
		ln2hi = archsimd.BroadcastFloat32x8(-0x1.62e4p-1)
		ln2lo = archsimd.BroadcastFloat32x8(-0x1.7f7d1cp-20)
		one   = archsimd.BroadcastFloat32x8(1).ToBits()
		c1    = archsimd.BroadcastFloat32x8(0x1.ffffecp-1)
		c2    = archsimd.BroadcastFloat32x8(0x1.fffdb6p-2)
		c3    = archsimd.BroadcastFloat32x8(0x1.555e66p-3)
		c4    = archsimd.BroadcastFloat32x8(0x1.573e2ep-5)
		c5    = archsimd.BroadcastFloat32x8(0x1.0e4020p-7)
		lo    = archsimd.BroadcastFloat32x8(kernelExpMin)
		m     = archsimd.BroadcastFloat32x8(max)
	)
	var sum float64
	i := 0
	for ; i+8 <= len(w); i += 8 {
		p := (*[8]float32)(w[i : i+8])
		x := kernelLoad8(p[:]).Sub(m)
		z := x.MulAdd(log2e, r)
		n := z.Sub(r)
		b := n.MulAdd(ln2lo, n.MulAdd(ln2hi, x))
		k := z.ToBits().ShiftAllLeft(23).Add(one).BitsToFloat32()
		u := b.Mul(b)
		j := c1.Mul(b).Add(c3.MulAdd(b, c2).Add(c5.MulAdd(b, c4).Mul(u)).Mul(u))
		// NaN stays NaN, as in kernelExp and ggml.
		e := j.MulAdd(k, k).Masked(x.GreaterEqual(lo).Or(x.IsNaN()))
		e.StoreArray(p)
		var lanes [8]float32
		e.StoreArray(&lanes)
		var part float32
		for _, v := range lanes {
			part += v
		}
		sum += float64(part)
	}
	return sum + kernelExpSumGeneric(w[i:], max)
}

func kernelScaleMax(w, m []float32, scale, slope float32) float32 {
	return kernelScaleMaxGeneric(w, m, scale, slope)
}

func kernelScaleStore(dst []byte, w []float32, s float32) { kernelScaleStoreGeneric(dst, w, s) }

func kernelDotQ2x4(qa, qb []byte, dma, dmb []float32, cols [4][]byte, yd, ys []float32, q8_1, narrow bool) (r0, r1 [4]float32) {
	return kernelDotQx4(qa, dma, cols, yd, ys, q8_1, narrow), kernelDotQx4(qb, dmb, cols, yd, ys, q8_1, narrow)
}

// kernelQuantPairs: kernelDotQ2x4 here is two calls, unmeasured as pairs.
const kernelQuantPairs = false
