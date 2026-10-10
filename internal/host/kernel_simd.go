//declscope:namespace kernel

//go:build goexperiment.simd && (arm64 || amd64)

package host

// kernelSpread maps a byte to 8 bytes, 0x10 where the byte has a bit set.
// The vector code builds q5_1's fifth bits with it.
var kernelSpread = func() *[256]uint64 {
	var t [256]uint64
	for v := range t {
		for bit := range 8 {
			if v>>bit&1 != 0 {
				t[v] |= 0x10 << (8 * bit)
			}
		}
	}
	return &t
}()
