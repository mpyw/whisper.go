//declscope:namespace kernel

//go:build !goexperiment.simd || !(arm64 || amd64)

package host

func kernelDotF32(x, y []float32) float32 { return kernelDotF32Generic(x, y) }
func kernelHalfRow(dst []float32, src []byte) {
	kernelHalfRowGeneric(dst, src)
}

func kernelFloatRow(dst []float32, src []byte) { kernelFloatRowGeneric(dst, src) }

func kernelExpSum(w []float32, max float32) float64 { return kernelExpSumGeneric(w, max) }

func kernelDotF32x4(x []float32, ys [4][]float32) [4]float32 {
	return kernelDotF32x4Generic(x, ys)
}

func kernelDecodeQ4_0(q []byte, dm []float32, src []byte) { kernelDecodeQ4_0Generic(q, dm, src) }
func kernelDecodeQ4_1(q []byte, dm []float32, src []byte) { kernelDecodeQ4_1Generic(q, dm, src) }
func kernelDecodeQ5_0(q []byte, dm []float32, src []byte) { kernelDecodeQ5_0Generic(q, dm, src) }
func kernelDecodeQ5_1(q []byte, dm []float32, src []byte) { kernelDecodeQ5_1Generic(q, dm, src) }

func kernelDotQ(q []byte, dm []float32, y []byte, yd, ys []float32, q8_1 bool) float32 {
	return kernelDotQGeneric(q, dm, y, yd, ys, q8_1)
}

func kernelDotQx4(q []byte, dm []float32, cols [4][]byte, yd, ys []float32, q8_1, narrow bool) [4]float32 {
	return kernelDotQx4Generic(q, dm, cols, yd, ys, q8_1, narrow)
}

func kernelDotF32x2x4(x0, x1 []float32, ys [4][]float32) (r0, r1 [4]float32) {
	return kernelDotF32x2x4Generic(x0, x1, ys)
}

func kernelScaleMax(w, m []float32, scale, slope float32) float32 {
	return kernelScaleMaxGeneric(w, m, scale, slope)
}

func kernelScaleStore(dst []byte, w []float32, s float32) { kernelScaleStoreGeneric(dst, w, s) }

func kernelDotQ2x4(qa, qb []byte, dma, dmb []float32, cols [4][]byte, yd, ys []float32, q8_1, narrow bool) (r0, r1 [4]float32) {
	return kernelDotQx4Generic(qa, dma, cols, yd, ys, q8_1, narrow), kernelDotQx4Generic(qb, dmb, cols, yd, ys, q8_1, narrow)
}

// kernelQuantPairs: the generic code gains nothing from 2 quantized rows at a time.
const kernelQuantPairs = false
