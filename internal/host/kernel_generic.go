//declscope:namespace kernel

package host

import (
	"encoding/binary"
	"math"
)

// kernelDotF32Generic is the dot product of x and y.
func kernelDotF32Generic(x, y []float32) float32 {
	var sum float32
	for i, v := range x[:min(len(x), len(y))] {
		sum += v * y[i]
	}
	return sum
}

// kernelDotF32x4Generic is kernelDotF32Generic with 4 columns, in one
// pass: each value of x is read once, and the 4 sums do not wait for each
// other. Each column adds in the same order as alone.
func kernelDotF32x4Generic(x []float32, ys [4][]float32) [4]float32 {
	n := len(x)
	y0, y1, y2, y3 := ys[0][:n], ys[1][:n], ys[2][:n], ys[3][:n]
	var s0, s1, s2, s3 float32
	for i, v := range x {
		s0 += v * y0[i]
		s1 += v * y1[i]
		s2 += v * y2[i]
		s3 += v * y3[i]
	}
	return [4]float32{s0, s1, s2, s3}
}

// kernelHalfRowGeneric converts the float16 values in src into dst.
func kernelHalfRowGeneric(dst []float32, src []byte) {
	for i := range dst {
		dst[i] = kernelHalf[binary.LittleEndian.Uint16(src[2*i:])]
	}
}

// kernelFloatRowGeneric copies little-endian float32 values.
func kernelFloatRowGeneric(dst []float32, src []byte) {
	src = src[:4*len(dst)]
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[4*i:]))
	}
}

// The decoders unpack a row of blocks: their 32 values as int8, one per
// byte of q, and their d and m, or 0, in dm.

func kernelDecodeQ4_0Generic(q []byte, dm []float32, src []byte) {
	for ib := range len(src) / kernelBlockQ4_0 {
		xb, out := src[ib*kernelBlockQ4_0:(ib+1)*kernelBlockQ4_0], q[32*ib:32*ib+32]
		for j, v := range xb[2:18] {
			out[j], out[j+16] = v&0xf-8, v>>4-8
		}
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), 0
	}
}

func kernelDecodeQ4_1Generic(q []byte, dm []float32, src []byte) {
	for ib := range len(src) / kernelBlockQ4_1 {
		xb, out := src[ib*kernelBlockQ4_1:(ib+1)*kernelBlockQ4_1], q[32*ib:32*ib+32]
		for j, v := range xb[4:20] {
			out[j], out[j+16] = v&0xf, v>>4
		}
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), kernelHalfAt(xb[2:])
	}
}

func kernelDecodeQ5_0Generic(q []byte, dm []float32, src []byte) {
	for ib := range len(src) / kernelBlockQ5_0 {
		xb, out := src[ib*kernelBlockQ5_0:(ib+1)*kernelBlockQ5_0], q[32*ib:32*ib+32]
		qh := binary.LittleEndian.Uint32(xb[2:])
		for j, v := range xb[6:22] {
			out[j] = v&0xf | byte(qh>>j<<4)&0x10 - 16
			out[j+16] = v>>4 | byte(qh>>(j+12))&0x10 - 16
		}
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), 0
	}
}

func kernelDecodeQ5_1Generic(q []byte, dm []float32, src []byte) {
	for ib := range len(src) / kernelBlockQ5_1 {
		xb, out := src[ib*kernelBlockQ5_1:(ib+1)*kernelBlockQ5_1], q[32*ib:32*ib+32]
		qh := binary.LittleEndian.Uint32(xb[4:])
		for j, v := range xb[8:24] {
			out[j] = v&0xf | byte(qh>>j<<4)&0x10
			out[j+16] = v>>4 | byte(qh>>(j+12))&0x10
		}
		dm[2*ib], dm[2*ib+1] = kernelHalfAt(xb), kernelHalfAt(xb[2:])
	}
}

// kernelDotQGeneric is the dot product of a row that a decoder unpacked
// and a column of q8_0 or q8_1 blocks. yd and ys hold the column's scales,
// as kernelScalesQ8 lays them out.
func kernelDotQGeneric(q []byte, dm []float32, y []byte, yd, ys []float32, q8_1 bool) float32 {
	block, off := kernelYValues(q8_1)
	var sum float32
	for ib := range len(y) / block {
		xq, yq := q[32*ib:32*ib+32], y[ib*block+off:(ib+1)*block]
		var sumi int32
		for j := range 32 {
			sumi += int32(int8(xq[j])) * int32(int8(yq[j]))
		}
		at := ib * kernelScaleStride
		sum += kernelScaleQ(dm[2*ib], dm[2*ib+1], yd[at], ys[at], sumi, q8_1)
	}
	return sum
}

// kernelDotQx4Generic is kernelDotQGeneric with 4 columns, in one pass
// over each block, as kernelDotF32x4Generic does.
// narrow is kernelDotQx4's, which changes nothing here.
func kernelDotQx4Generic(q []byte, dm []float32, cols [4][]byte, yd, ys []float32, q8_1, _ bool) [4]float32 {
	block, off := kernelYValues(q8_1)
	var sums [4]float32
	for ib := range len(cols[0]) / block {
		xq := q[32*ib : 32*ib+32]
		y0 := cols[0][ib*block+off : (ib+1)*block]
		y1 := cols[1][ib*block+off : (ib+1)*block]
		y2 := cols[2][ib*block+off : (ib+1)*block]
		y3 := cols[3][ib*block+off : (ib+1)*block]
		var s0, s1, s2, s3 int32
		for j, v := range xq {
			x := int32(int8(v))
			s0 += x * int32(int8(y0[j]))
			s1 += x * int32(int8(y1[j]))
			s2 += x * int32(int8(y2[j]))
			s3 += x * int32(int8(y3[j]))
		}
		at := ib * kernelScaleStride
		for c, sumi := range [4]int32{s0, s1, s2, s3} {
			sums[c] += kernelScaleQ(dm[2*ib], dm[2*ib+1], yd[at+c], ys[at+c], sumi, q8_1)
		}
	}
	return sums
}

// kernelDotF32x2x4Generic is kernelDotF32x4Generic for 2 rows.
func kernelDotF32x2x4Generic(x0, x1 []float32, ys [4][]float32) (r0, r1 [4]float32) {
	return kernelDotF32x4Generic(x0, ys), kernelDotF32x4Generic(x1, ys)
}

// kernelExpSumGeneric replaces each value v of w by exp(v - max), and
// returns their sum. max is at least each v, as in a softmax.
func kernelExpSumGeneric(w []float32, max float32) float64 {
	var sum float64
	for i, v := range w {
		w[i] = kernelExp(v - max)
		sum += float64(w[i])
	}
	return sum
}

// kernelExp is ggml's ggml_v_expf for x at most 0, one value at a time:
// exp(x) as 2^n times a polynomial of the rest. Below kernelExpMin, the
// exponent would be under float32's, and it returns 0.
func kernelExp(x float32) float32 {
	if x < kernelExpMin {
		return 0
	}
	const r = 0x1.8p23
	z := x*0x1.715476p+0 + r
	n := z - r
	b := x - n*0x1.62e4p-1 - n*0x1.7f7d1cp-20
	k := math.Float32frombits(math.Float32bits(z)<<23 + math.Float32bits(1))
	u := b * b
	j := 0x1.ffffecp-1*b + (0x1.fffdb6p-2+0x1.555e66p-3*b+(0x1.573e2ep-5+0x1.0e4020p-7*b)*u)*u
	return k + j*k
}

// kernelExpMin is where kernelExp returns 0: exp of it is about 2^-125.
const kernelExpMin = -86.6

// kernelScaleMaxGeneric sets each w[i] to w[i]*scale + slope*m[i], or
// w[i]*scale with no m, and returns the largest.
func kernelScaleMaxGeneric(w, m []float32, scale, slope float32) float32 {
	maxW := float32(math.Inf(-1))
	if m != nil {
		m = m[:len(w)]
	}
	for i, v := range w {
		v *= scale
		if m != nil {
			v += slope * m[i]
		}
		w[i] = v
		maxW = max(maxW, v)
	}
	return maxW
}

// kernelScaleStoreGeneric writes each w[i]*s to dst, as little-endian
// float32.
func kernelScaleStoreGeneric(dst []byte, w []float32, s float32) {
	dst = dst[:4*len(w)]
	for i, v := range w {
		binary.LittleEndian.PutUint32(dst[4*i:], math.Float32bits(v*s))
	}
}
