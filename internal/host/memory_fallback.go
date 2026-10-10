//declscope:namespace memory

//go:build !unix && !windows

package host

import "errors"

// memoryReserve fails: without a way to reserve address space, the memory
// would move as it grows, under threads that read it.
func memoryReserve(int) ([]byte, error) {
	return nil, errors.New("host: this platform cannot reserve address space")
}

func memoryCommit([]byte, int, int) error { return nil }

func memoryRelease([]byte) error { return nil }
