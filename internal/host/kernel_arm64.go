//go:build goexperiment.simd

package host

import (
	"encoding/binary"
	"math"
	"simd/archsimd"
)

// kernelHalfConverter holds the constants of a float16 conversion, made
// once per call rather than once per vector.
type kernelHalfConverter struct {
	expFrac, sign, infNaN, allOnes archsimd.Uint32x4
	scale                          archsimd.Float32x4
}

func kernelHalfConsts() kernelHalfConverter {
	return kernelHalfConverter{
		expFrac: archsimd.BroadcastUint32x4(0x7fff),
		sign:    archsimd.BroadcastUint32x4(0x8000),
		infNaN:  archsimd.BroadcastUint32x4(0x7c00 << 13),
		allOnes: archsimd.BroadcastUint32x4(0x7f800000),
		scale:   archsimd.BroadcastFloat32x4(0x1p112),
	}
}

// convert converts 4 float16 values, zero-extended to 32 bits. The exponent
// and the fraction, shifted into a float32's places, are the value times
// 2^-112, subnormals included. Infinities and NaNs get the all-ones
// exponent instead.
func (c kernelHalfConverter) convert(h archsimd.Uint32x4) archsimd.Float32x4 {
	em := h.And(c.expFrac).ShiftAllLeft(13)
	bits := em.BitsToFloat32().Mul(c.scale).ToBits().Or(h.And(c.sign).ShiftAllLeft(16))
	return bits.Or(c.allOnes.Masked(em.GreaterEqual(c.infNaN))).BitsToFloat32()
}

// kernelDot16 is the products of 16 pairs of int8, summed into 4 lanes. A
// product fits in int16, and a sum in int32.
func kernelDot16(x, y archsimd.Int8x16) archsimd.Int32x4 {
	lo := x.MulWidenLo(y)
	hi := x.HiToLo().MulWidenLo(y.HiToLo())
	return lo.ExtendLo4ToInt32().Add(lo.HiToLo().ExtendLo4ToInt32()).
		Add(hi.ExtendLo4ToInt32()).Add(hi.HiToLo().ExtendLo4ToInt32())
}

// kernelFifthBits spreads the low 16 bits of qh to 16 bytes: 0x10 for a set
// bit, 0 for a clear one.
func kernelFifthBits(qh uint32) archsimd.Uint8x16 {
	var b [16]uint8
	binary.LittleEndian.PutUint64(b[0:], kernelSpread[qh&0xff])
	binary.LittleEndian.PutUint64(b[8:], kernelSpread[qh>>8&0xff])
	return archsimd.LoadUint8x16Array(&b)
}

// kernelDotF32 sums 16 products at a time, in four float32 vectors.
func kernelDotF32(x, y []float32) float32 {
	n := min(len(x), len(y))
	x, y = x[:n], y[:n]
	var acc0, acc1, acc2, acc3 archsimd.Float32x4
	i := 0
	for ; i+16 <= n; i += 16 {
		xa, ya := (*[16]float32)(x[i:i+16]), (*[16]float32)(y[i:i+16])
		acc0 = kernelLoad4(xa[0:4]).MulAdd(kernelLoad4(ya[0:4]), acc0)
		acc1 = kernelLoad4(xa[4:8]).MulAdd(kernelLoad4(ya[4:8]), acc1)
		acc2 = kernelLoad4(xa[8:12]).MulAdd(kernelLoad4(ya[8:12]), acc2)
		acc3 = kernelLoad4(xa[12:16]).MulAdd(kernelLoad4(ya[12:16]), acc3)
	}
	var lanes [4]float32
	acc0.Add(acc1).Add(acc2.Add(acc3)).StoreArray(&lanes)
	sum := lanes[0] + lanes[1] + lanes[2] + lanes[3]
	for ; i < n; i++ {
		sum += x[i] * y[i]
	}
	return sum
}

// kernelLoad4 loads p, of 4 values. Loads through array pointers have no
// bounds checks of their own: the caller slices once for several.
func kernelLoad4(p []float32) archsimd.Float32x4 {
	return archsimd.LoadFloat32x4Array((*[4]float32)(p))
}

// kernelFloatRow copies 16 float32 values at a time.
func kernelFloatRow(dst []float32, src []byte) {
	n := len(dst)
	src = src[:4*n]
	i := 0
	for ; i+16 <= n; i += 16 {
		s, d := (*[64]byte)(src[4*i:4*i+64]), (*[16]float32)(dst[i:i+16])
		for k := 0; k < 4; k++ {
			v := archsimd.LoadUint8x16Array((*[16]byte)(s[16*k : 16*k+16])).ReshapeToUint32s().BitsToFloat32()
			v.StoreArray((*[4]float32)(d[4*k : 4*k+4]))
		}
	}
	kernelFloatRowGeneric(dst[i:], src[4*i:])
}

// kernelHalfRow converts 8 float16 values at a time.
func kernelHalfRow(dst []float32, src []byte) {
	c := kernelHalfConsts()
	i := 0
	n := len(dst)
	src = src[:2*n]
	for ; i+8 <= n; i += 8 {
		h := archsimd.LoadUint8x16Array((*[16]byte)(src[2*i : 2*i+16])).ReshapeToUint16s()
		d := (*[8]float32)(dst[i : i+8])
		c.convert(h.ExtendLo4ToUint32()).StoreArray((*[4]float32)(d[0:4]))
		c.convert(h.HiToLo().ExtendLo4ToUint32()).StoreArray((*[4]float32)(d[4:8]))
	}
	for ; i < len(dst); i++ {
		dst[i] = kernelHalf[binary.LittleEndian.Uint16(src[2*i:])]
	}
}

// kernelDecodeQ4_0 unpacks q4_0 blocks, as kernelDecodeQ4_0Generic does.
func kernelDecodeQ4_0(q []byte, dm []float32, src []byte) {
	nibble, eight := archsimd.BroadcastUint8x16(0x0f), archsimd.BroadcastUint8x16(8)
	for ib := range len(src) / kernelBlockQ4_0 {
		xb := src[ib*kernelBlockQ4_0 : (ib+1)*kernelBlockQ4_0]
		qs := archsimd.LoadUint8x16(xb[2:18])
		qs.And(nibble).Sub(eight).Store(q[32*ib:])
		qs.ShiftAllRight(4).Sub(eight).Store(q[32*ib+16:])
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), 0
	}
}

// kernelDecodeQ4_1 unpacks q4_1 blocks, as kernelDecodeQ4_1Generic does.
func kernelDecodeQ4_1(q []byte, dm []float32, src []byte) {
	nibble := archsimd.BroadcastUint8x16(0x0f)
	for ib := range len(src) / kernelBlockQ4_1 {
		xb := src[ib*kernelBlockQ4_1 : (ib+1)*kernelBlockQ4_1]
		qs := archsimd.LoadUint8x16(xb[4:20])
		qs.And(nibble).Store(q[32*ib:])
		qs.ShiftAllRight(4).Store(q[32*ib+16:])
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), kernelHalfAt(xb[2:])
	}
}

// kernelDecodeQ5_0 unpacks q5_0 blocks, as kernelDecodeQ5_0Generic does.
// The subtraction wraps, which leaves the int8 values.
func kernelDecodeQ5_0(q []byte, dm []float32, src []byte) {
	nibble, sixteen := archsimd.BroadcastUint8x16(0x0f), archsimd.BroadcastUint8x16(16)
	for ib := range len(src) / kernelBlockQ5_0 {
		xb := src[ib*kernelBlockQ5_0 : (ib+1)*kernelBlockQ5_0]
		qh := binary.LittleEndian.Uint32(xb[2:])
		qs := archsimd.LoadUint8x16(xb[6:22])
		qs.And(nibble).Or(kernelFifthBits(qh)).Sub(sixteen).Store(q[32*ib:])
		qs.ShiftAllRight(4).Or(kernelFifthBits(qh >> 16)).Sub(sixteen).Store(q[32*ib+16:])
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), 0
	}
}

// kernelDecodeQ5_1 unpacks q5_1 blocks, as kernelDecodeQ5_1Generic does.
func kernelDecodeQ5_1(q []byte, dm []float32, src []byte) {
	nibble := archsimd.BroadcastUint8x16(0x0f)
	for ib := range len(src) / kernelBlockQ5_1 {
		xb := src[ib*kernelBlockQ5_1 : (ib+1)*kernelBlockQ5_1]
		qh := binary.LittleEndian.Uint32(xb[4:])
		qs := archsimd.LoadUint8x16(xb[8:24])
		qs.And(nibble).Or(kernelFifthBits(qh)).Store(q[32*ib:])
		qs.ShiftAllRight(4).Or(kernelFifthBits(qh >> 16)).Store(q[32*ib+16:])
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), kernelHalfAt(xb[2:])
	}
}

// kernelDotQ is kernelDotQGeneric, for one column. Each block's integer
// sums are scaled in their 4 lanes, and the lanes are added once, at the end.
func kernelDotQ(q []byte, dm []float32, y []byte, yd, ys []float32, q8_1 bool) float32 {
	block, off := kernelYValues(q8_1)
	nb := len(y) / block
	q = q[:32*nb]
	var acc archsimd.Float32x4
	var mins float32
	for ib := range nb {
		xq := (*[32]byte)(q[32*ib : 32*ib+32])
		lo := ib*block + off
		yb := (*[32]byte)(y[lo : lo+32])
		sumi := kernelDot16(kernelLoadInt8(xq[0:16]), kernelLoadInt8(yb[0:16])).
			Add(kernelDot16(kernelLoadInt8(xq[16:32]), kernelLoadInt8(yb[16:32])))
		at := ib * kernelScaleStride
		acc = sumi.ConvertToFloat32().MulAdd(archsimd.BroadcastFloat32x4(dm[2*ib]*yd[at]), acc)
		if q8_1 {
			mins += dm[2*ib+1] * ys[at]
		}
	}
	var lanes [4]float32
	acc.StoreArray(&lanes)
	return lanes[0] + lanes[1] + lanes[2] + lanes[3] + mins
}

// kernelDotQx4 multiplies an unpacked row with 4 columns. Each block of the
// row is loaded once, and the 4 columns' sums and scales are one vector.
// Named sums and accumulators stay in registers, where arrays of vectors
// would go through the stack. narrow is for the q4 and q5 types, whose
// values are below 32: their products add up in int16, and widen once.
func kernelDotQx4(q []byte, dm []float32, cols [4][]byte, yd, ys []float32, q8_1, narrow bool) [4]float32 {
	block, off := kernelYValues(q8_1)
	var acc archsimd.Float32x4
	nb := len(cols[0]) / block
	y0, y1, y2, y3 := cols[0][:nb*block], cols[1][:nb*block], cols[2][:nb*block], cols[3][:nb*block]
	q = q[:32*nb]
	for ib := range nb {
		xq := (*[32]byte)(q[32*ib : 32*ib+32])
		x0, x1 := kernelLoadInt8(xq[0:16]), kernelLoadInt8(xq[16:32])
		lo := ib*block + off
		b0, b1, b2, b3 := (*[32]byte)(y0[lo:lo+32]), (*[32]byte)(y1[lo:lo+32]), (*[32]byte)(y2[lo:lo+32]), (*[32]byte)(y3[lo:lo+32])
		var sumi archsimd.Float32x4
		if narrow {
			x0h, x1h := x0.HiToLo(), x1.HiToLo()
			t01 := kernelSum32Narrow(x0, x0h, x1, x1h, b0).ConcatAddPairs(kernelSum32Narrow(x0, x0h, x1, x1h, b1))
			t23 := kernelSum32Narrow(x0, x0h, x1, x1h, b2).ConcatAddPairs(kernelSum32Narrow(x0, x0h, x1, x1h, b3))
			w0, w1 := t01.ExtendLo4ToInt32(), t01.HiToLo().ExtendLo4ToInt32()
			w2, w3 := t23.ExtendLo4ToInt32(), t23.HiToLo().ExtendLo4ToInt32()
			sumi = w0.ConcatAddPairs(w1).ConcatAddPairs(w2.ConcatAddPairs(w3)).ConvertToFloat32()
		} else {
			s0 := kernelDot16(x0, kernelLoadInt8(b0[0:16])).Add(kernelDot16(x1, kernelLoadInt8(b0[16:32])))
			s1 := kernelDot16(x0, kernelLoadInt8(b1[0:16])).Add(kernelDot16(x1, kernelLoadInt8(b1[16:32])))
			s2 := kernelDot16(x0, kernelLoadInt8(b2[0:16])).Add(kernelDot16(x1, kernelLoadInt8(b2[16:32])))
			s3 := kernelDot16(x0, kernelLoadInt8(b3[0:16])).Add(kernelDot16(x1, kernelLoadInt8(b3[16:32])))
			sumi = s0.ConcatAddPairs(s1).ConcatAddPairs(s2.ConcatAddPairs(s3)).ConvertToFloat32()
		}
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

// kernelSum32Narrow sums the 32 products of x (x0, x1 and their high
// halves) and b into 8 int16 lanes, 4 products a lane. It is exact while
// |x| <= 32: 4 products of at most 32*128 fit in int16, and so do 8.
func kernelSum32Narrow(x0, x0h, x1, x1h archsimd.Int8x16, b *[32]byte) archsimd.Int16x8 {
	y0, y1 := kernelLoadInt8(b[0:16]), kernelLoadInt8(b[16:32])
	return x0.MulWidenLo(y0).Add(x0h.MulWidenLo(y0.HiToLo())).Add(x1.MulWidenLo(y1).Add(x1h.MulWidenLo(y1.HiToLo())))
}

// kernelLoadInt8 loads p, of 16 values, as kernelLoad4 does.
func kernelLoadInt8(p []byte) archsimd.Int8x16 {
	return archsimd.LoadUint8x16Array((*[16]byte)(p)).BitsToInt8()
}

// kernelDotF32x4 is kernelDotF32 with 4 columns: each vector of x is loaded
// once for all 4. Two vectors a step keep 8 sums in flight, since each
// waits for the last multiply-add of its own.
func kernelDotF32x4(x []float32, ys [4][]float32) [4]float32 {
	n := len(x)
	y0, y1, y2, y3 := ys[0][:n], ys[1][:n], ys[2][:n], ys[3][:n]
	var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Float32x4
	i := 0
	for ; i+8 <= n; i += 8 {
		xx := (*[8]float32)(x[i : i+8])
		q0, q1, q2, q3 := (*[8]float32)(y0[i:i+8]), (*[8]float32)(y1[i:i+8]), (*[8]float32)(y2[i:i+8]), (*[8]float32)(y3[i:i+8])
		xa, xb := kernelLoad4(xx[0:4]), kernelLoad4(xx[4:8])
		a0 = xa.MulAdd(kernelLoad4(q0[0:4]), a0)
		a1 = xa.MulAdd(kernelLoad4(q1[0:4]), a1)
		a2 = xa.MulAdd(kernelLoad4(q2[0:4]), a2)
		a3 = xa.MulAdd(kernelLoad4(q3[0:4]), a3)
		a4 = xb.MulAdd(kernelLoad4(q0[4:8]), a4)
		a5 = xb.MulAdd(kernelLoad4(q1[4:8]), a5)
		a6 = xb.MulAdd(kernelLoad4(q2[4:8]), a6)
		a7 = xb.MulAdd(kernelLoad4(q3[4:8]), a7)
	}
	var sums [4]float32
	a0.Add(a4).ConcatAddPairs(a1.Add(a5)).ConcatAddPairs(a2.Add(a6).ConcatAddPairs(a3.Add(a7))).StoreArray(&sums)
	for c, y := range [4][]float32{y0, y1, y2, y3} {
		for j := i; j < n; j++ {
			sums[c] += x[j] * y[j]
		}
	}
	return sums
}

// kernelDotF32x2x4 is kernelDotF32x4 for 2 rows: each vector of a column
// is loaded once for both, and 8 sums are in flight.
func kernelDotF32x2x4(x0, x1 []float32, ys [4][]float32) (r0, r1 [4]float32) {
	n := len(x0)
	x1 = x1[:n]
	y0, y1, y2, y3 := ys[0][:n], ys[1][:n], ys[2][:n], ys[3][:n]
	var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Float32x4
	var b0, b1, b2, b3, b4, b5, b6, b7 archsimd.Float32x4
	i := 0
	for ; i+8 <= n; i += 8 {
		p0, p1 := (*[8]float32)(x0[i:i+8]), (*[8]float32)(x1[i:i+8])
		q0, q1, q2, q3 := (*[8]float32)(y0[i:i+8]), (*[8]float32)(y1[i:i+8]), (*[8]float32)(y2[i:i+8]), (*[8]float32)(y3[i:i+8])
		xa, xb := kernelLoad4(p0[0:4]), kernelLoad4(p1[0:4])
		c0, c1, c2, c3 := kernelLoad4(q0[0:4]), kernelLoad4(q1[0:4]), kernelLoad4(q2[0:4]), kernelLoad4(q3[0:4])
		a0 = xa.MulAdd(c0, a0)
		a1 = xa.MulAdd(c1, a1)
		a2 = xa.MulAdd(c2, a2)
		a3 = xa.MulAdd(c3, a3)
		a4 = xb.MulAdd(c0, a4)
		a5 = xb.MulAdd(c1, a5)
		a6 = xb.MulAdd(c2, a6)
		a7 = xb.MulAdd(c3, a7)
		xa, xb = kernelLoad4(p0[4:8]), kernelLoad4(p1[4:8])
		c0, c1, c2, c3 = kernelLoad4(q0[4:8]), kernelLoad4(q1[4:8]), kernelLoad4(q2[4:8]), kernelLoad4(q3[4:8])
		b0 = xa.MulAdd(c0, b0)
		b1 = xa.MulAdd(c1, b1)
		b2 = xa.MulAdd(c2, b2)
		b3 = xa.MulAdd(c3, b3)
		b4 = xb.MulAdd(c0, b4)
		b5 = xb.MulAdd(c1, b5)
		b6 = xb.MulAdd(c2, b6)
		b7 = xb.MulAdd(c3, b7)
	}
	a0.Add(b0).ConcatAddPairs(a1.Add(b1)).ConcatAddPairs(a2.Add(b2).ConcatAddPairs(a3.Add(b3))).StoreArray(&r0)
	a4.Add(b4).ConcatAddPairs(a5.Add(b5)).ConcatAddPairs(a6.Add(b6).ConcatAddPairs(a7.Add(b7))).StoreArray(&r1)
	for c, y := range [4][]float32{y0, y1, y2, y3} {
		for j := i; j < n; j++ {
			r0[c] += x0[j] * y[j]
			r1[c] += x1[j] * y[j]
		}
	}
	return r0, r1
}

// kernelExpSum is kernelExpSumGeneric, 4 values at a time.
func kernelExpSum(w []float32, max float32) float64 {
	var (
		r     = archsimd.BroadcastFloat32x4(0x1.8p23)
		log2e = archsimd.BroadcastFloat32x4(0x1.715476p+0)
		ln2hi = archsimd.BroadcastFloat32x4(-0x1.62e4p-1)
		ln2lo = archsimd.BroadcastFloat32x4(-0x1.7f7d1cp-20)
		one   = archsimd.BroadcastFloat32x4(1).ToBits()
		c1    = archsimd.BroadcastFloat32x4(0x1.ffffecp-1)
		c2    = archsimd.BroadcastFloat32x4(0x1.fffdb6p-2)
		c3    = archsimd.BroadcastFloat32x4(0x1.555e66p-3)
		c4    = archsimd.BroadcastFloat32x4(0x1.573e2ep-5)
		c5    = archsimd.BroadcastFloat32x4(0x1.0e4020p-7)
		lo    = archsimd.BroadcastFloat32x4(kernelExpMin)
		m     = archsimd.BroadcastFloat32x4(max)
	)
	var acc0, acc1 archsimd.Float64x2
	i := 0
	for ; i+4 <= len(w); i += 4 {
		p := (*[4]float32)(w[i : i+4])
		x := kernelLoad4(p[:]).Sub(m)
		z := x.MulAdd(log2e, r)
		n := z.Sub(r)
		b := n.MulAdd(ln2lo, n.MulAdd(ln2hi, x))
		k := z.ToBits().ShiftAllLeft(23).Add(one).BitsToFloat32()
		u := b.Mul(b)
		j := c5.MulAdd(b, c4).MulAdd(u, c3.MulAdd(b, c2)).MulAdd(u, c1.Mul(b))
		// NaN stays NaN, as in kernelExp and ggml.
		e := j.MulAdd(k, k).Masked(x.Less(lo).Not())
		e.StoreArray(p)
		acc0 = acc0.Add(e.ConvertLo2ToFloat64())
		acc1 = acc1.Add(e.HiToLo().ConvertLo2ToFloat64())
	}
	acc0 = acc0.Add(acc1)
	return acc0.GetElem(0) + acc0.GetElem(1) + kernelExpSumGeneric(w[i:], max)
}

// kernelScaleMax is kernelScaleMaxGeneric, 8 values at a time.
func kernelScaleMax(w, m []float32, scale, slope float32) float32 {
	n := len(w)
	sc, sl := archsimd.BroadcastFloat32x4(scale), archsimd.BroadcastFloat32x4(slope)
	ninf := float32(math.Inf(-1))
	mx0, mx1 := archsimd.BroadcastFloat32x4(ninf), archsimd.BroadcastFloat32x4(ninf)
	i := 0
	if m != nil {
		m = m[:n]
		for ; i+8 <= n; i += 8 {
			p, q := (*[8]float32)(w[i:i+8]), (*[8]float32)(m[i:i+8])
			v0 := kernelLoad4(p[0:4]).Mul(sc)
			v1 := kernelLoad4(p[4:8]).Mul(sc)
			v0 = kernelLoad4(q[0:4]).MulAdd(sl, v0)
			v1 = kernelLoad4(q[4:8]).MulAdd(sl, v1)
			v0.StoreArray((*[4]float32)(p[0:4]))
			v1.StoreArray((*[4]float32)(p[4:8]))
			mx0, mx1 = mx0.Max(v0), mx1.Max(v1)
		}
	} else {
		for ; i+8 <= n; i += 8 {
			p := (*[8]float32)(w[i : i+8])
			v0, v1 := kernelLoad4(p[0:4]).Mul(sc), kernelLoad4(p[4:8]).Mul(sc)
			v0.StoreArray((*[4]float32)(p[0:4]))
			v1.StoreArray((*[4]float32)(p[4:8]))
			mx0, mx1 = mx0.Max(v0), mx1.Max(v1)
		}
	}
	var tail []float32
	if m != nil {
		tail = m[i:]
	}
	return max(mx0.Max(mx1).ReduceMax(), kernelScaleMaxGeneric(w[i:], tail, scale, slope))
}

// kernelScaleStore is kernelScaleStoreGeneric, 8 values at a time.
func kernelScaleStore(dst []byte, w []float32, s float32) {
	n := len(w)
	dst = dst[:4*n]
	sv := archsimd.BroadcastFloat32x4(s)
	i := 0
	for ; i+8 <= n; i += 8 {
		p, d := (*[8]float32)(w[i:i+8]), (*[32]byte)(dst[4*i:4*i+32])
		kernelLoad4(p[0:4]).Mul(sv).ToBits().ReshapeToUint8s().StoreArray((*[16]byte)(d[0:16]))
		kernelLoad4(p[4:8]).Mul(sv).ToBits().ReshapeToUint8s().StoreArray((*[16]byte)(d[16:32]))
	}
	kernelScaleStoreGeneric(dst[4*i:], w[i:], s)
}

// kernelDotQ2x4 is kernelDotQx4 for 2 rows: each block of a column is
// loaded once for both. Only the narrow sums share it.
func kernelDotQ2x4(qa, qb []byte, dma, dmb []float32, cols [4][]byte, yd, ys []float32, q8_1, narrow bool) (r0, r1 [4]float32) {
	if !narrow {
		return kernelDotQ2x4Wide(qa, qb, dma, dmb, cols, yd, ys, q8_1)
	}
	block, off := kernelYValues(q8_1)
	var acc0, acc1 archsimd.Float32x4
	nb := len(cols[0]) / block
	y0, y1, y2, y3 := cols[0][:nb*block], cols[1][:nb*block], cols[2][:nb*block], cols[3][:nb*block]
	qa, qb = qa[:32*nb], qb[:32*nb]
	dma, dmb = dma[:2*nb], dmb[:2*nb]
	for ib := range nb {
		xa, xb := (*[32]byte)(qa[32*ib:32*ib+32]), (*[32]byte)(qb[32*ib:32*ib+32])
		a0, a1 := kernelLoadInt8(xa[0:16]), kernelLoadInt8(xa[16:32])
		c0, c1 := kernelLoadInt8(xb[0:16]), kernelLoadInt8(xb[16:32])
		a0h, a1h, c0h, c1h := a0.HiToLo(), a1.HiToLo(), c0.HiToLo(), c1.HiToLo()
		lo := ib*block + off
		b0, b1, b2, b3 := (*[32]byte)(y0[lo:lo+32]), (*[32]byte)(y1[lo:lo+32]), (*[32]byte)(y2[lo:lo+32]), (*[32]byte)(y3[lo:lo+32])
		pa0, pb0 := kernelSum32Narrow2(a0, a0h, a1, a1h, c0, c0h, c1, c1h, b0)
		pa1, pb1 := kernelSum32Narrow2(a0, a0h, a1, a1h, c0, c0h, c1, c1h, b1)
		pa2, pb2 := kernelSum32Narrow2(a0, a0h, a1, a1h, c0, c0h, c1, c1h, b2)
		pa3, pb3 := kernelSum32Narrow2(a0, a0h, a1, a1h, c0, c0h, c1, c1h, b3)
		at := ib * kernelScaleStride
		yv := archsimd.LoadFloat32x4(yd[at:])
		ta01, ta23 := pa0.ConcatAddPairs(pa1), pa2.ConcatAddPairs(pa3)
		sa := ta01.ExtendLo4ToInt32().ConcatAddPairs(ta01.HiToLo().ExtendLo4ToInt32()).
			ConcatAddPairs(ta23.ExtendLo4ToInt32().ConcatAddPairs(ta23.HiToLo().ExtendLo4ToInt32())).ConvertToFloat32()
		tb01, tb23 := pb0.ConcatAddPairs(pb1), pb2.ConcatAddPairs(pb3)
		sb := tb01.ExtendLo4ToInt32().ConcatAddPairs(tb01.HiToLo().ExtendLo4ToInt32()).
			ConcatAddPairs(tb23.ExtendLo4ToInt32().ConcatAddPairs(tb23.HiToLo().ExtendLo4ToInt32())).ConvertToFloat32()
		pa := sa.Mul(archsimd.BroadcastFloat32x4(dma[2*ib]).Mul(yv))
		pb := sb.Mul(archsimd.BroadcastFloat32x4(dmb[2*ib]).Mul(yv))
		if q8_1 {
			sv := archsimd.LoadFloat32x4(ys[at:])
			pa = pa.Add(archsimd.BroadcastFloat32x4(dma[2*ib+1]).Mul(sv))
			pb = pb.Add(archsimd.BroadcastFloat32x4(dmb[2*ib+1]).Mul(sv))
		}
		acc0, acc1 = acc0.Add(pa), acc1.Add(pb)
	}
	acc0.StoreArray(&r0)
	acc1.StoreArray(&r1)
	return r0, r1
}

// kernelSum32Narrow2 is kernelSum32Narrow for 2 rows, a and c.
func kernelSum32Narrow2(a0, a0h, a1, a1h, c0, c0h, c1, c1h archsimd.Int8x16, b *[32]byte) (pa, pc archsimd.Int16x8) {
	y0, y1 := kernelLoadInt8(b[0:16]), kernelLoadInt8(b[16:32])
	y0h, y1h := y0.HiToLo(), y1.HiToLo()
	pa = a0.MulWidenLo(y0).Add(a0h.MulWidenLo(y0h)).Add(a1.MulWidenLo(y1).Add(a1h.MulWidenLo(y1h)))
	pc = c0.MulWidenLo(y0).Add(c0h.MulWidenLo(y0h)).Add(c1.MulWidenLo(y1).Add(c1h.MulWidenLo(y1h)))
	return pa, pc
}

// kernelDotQ2x4Wide is kernelDotQ2x4 for values of all of int8, as q8_0's.
func kernelDotQ2x4Wide(qa, qb []byte, dma, dmb []float32, cols [4][]byte, yd, ys []float32, q8_1 bool) (r0, r1 [4]float32) {
	block, off := kernelYValues(q8_1)
	var acc0, acc1 archsimd.Float32x4
	nb := len(cols[0]) / block
	y0, y1, y2, y3 := cols[0][:nb*block], cols[1][:nb*block], cols[2][:nb*block], cols[3][:nb*block]
	qa, qb = qa[:32*nb], qb[:32*nb]
	dma, dmb = dma[:2*nb], dmb[:2*nb]
	for ib := range nb {
		xa, xb := (*[32]byte)(qa[32*ib:32*ib+32]), (*[32]byte)(qb[32*ib:32*ib+32])
		a0, a1 := kernelLoadInt8(xa[0:16]), kernelLoadInt8(xa[16:32])
		c0, c1 := kernelLoadInt8(xb[0:16]), kernelLoadInt8(xb[16:32])
		lo := ib*block + off
		b0, b1, b2, b3 := (*[32]byte)(y0[lo:lo+32]), (*[32]byte)(y1[lo:lo+32]), (*[32]byte)(y2[lo:lo+32]), (*[32]byte)(y3[lo:lo+32])
		u0, u1 := kernelLoadInt8(b0[0:16]), kernelLoadInt8(b0[16:32])
		sa0, sb0 := kernelDot16(a0, u0).Add(kernelDot16(a1, u1)), kernelDot16(c0, u0).Add(kernelDot16(c1, u1))
		u0, u1 = kernelLoadInt8(b1[0:16]), kernelLoadInt8(b1[16:32])
		sa1, sb1 := kernelDot16(a0, u0).Add(kernelDot16(a1, u1)), kernelDot16(c0, u0).Add(kernelDot16(c1, u1))
		u0, u1 = kernelLoadInt8(b2[0:16]), kernelLoadInt8(b2[16:32])
		sa2, sb2 := kernelDot16(a0, u0).Add(kernelDot16(a1, u1)), kernelDot16(c0, u0).Add(kernelDot16(c1, u1))
		u0, u1 = kernelLoadInt8(b3[0:16]), kernelLoadInt8(b3[16:32])
		sa3, sb3 := kernelDot16(a0, u0).Add(kernelDot16(a1, u1)), kernelDot16(c0, u0).Add(kernelDot16(c1, u1))
		sa := sa0.ConcatAddPairs(sa1).ConcatAddPairs(sa2.ConcatAddPairs(sa3)).ConvertToFloat32()
		sb := sb0.ConcatAddPairs(sb1).ConcatAddPairs(sb2.ConcatAddPairs(sb3)).ConvertToFloat32()
		at := ib * kernelScaleStride
		yv := archsimd.LoadFloat32x4(yd[at:])
		pa := sa.Mul(archsimd.BroadcastFloat32x4(dma[2*ib]).Mul(yv))
		pb := sb.Mul(archsimd.BroadcastFloat32x4(dmb[2*ib]).Mul(yv))
		if q8_1 {
			sv := archsimd.LoadFloat32x4(ys[at:])
			pa = pa.Add(archsimd.BroadcastFloat32x4(dma[2*ib+1]).Mul(sv))
			pb = pb.Add(archsimd.BroadcastFloat32x4(dmb[2*ib+1]).Mul(sv))
		}
		acc0, acc1 = acc0.Add(pa), acc1.Add(pb)
	}
	acc0.StoreArray(&r0)
	acc1.StoreArray(&r1)
	return r0, r1
}

// kernelQuantPairs: two quantized rows at a time share the loads of y.
const kernelQuantPairs = true
