//declscope:namespace memory

//go:build unix

package host

import "golang.org/x/sys/unix"

// memoryReserve maps size bytes with no access. Such a mapping takes address
// space only, so reserving 4 GiB commits nothing.
func memoryReserve(size int) ([]byte, error) {
	return unix.Mmap(-1, 0, size, unix.PROT_NONE, unix.MAP_PRIVATE|unix.MAP_ANON)
}

// memoryCommit makes b[from:to] readable and writable.
func memoryCommit(b []byte, from, to int) error {
	if from == to {
		return nil
	}
	return unix.Mprotect(b[from:to], unix.PROT_READ|unix.PROT_WRITE)
}

func memoryRelease(b []byte) error {
	if cap(b) == 0 {
		return nil
	}
	return unix.Munmap(b[:cap(b)])
}
