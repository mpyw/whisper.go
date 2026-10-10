package host

import (
	"unsafe"

	"golang.org/x/sys/windows"
)

// memoryReserve reserves size bytes of address space and commits none.
func memoryReserve(size int) ([]byte, error) {
	p, err := windows.VirtualAlloc(0, uintptr(size), windows.MEM_RESERVE, windows.PAGE_NOACCESS)
	if err != nil {
		return nil, err
	}
	// p is address space Go does not manage, so no GC object moves under it.
	return unsafe.Slice((*byte)(unsafe.Add(nil, p)), size), nil
}

// memoryCommit commits b[from:to].
func memoryCommit(b []byte, from, to int) error {
	if from == to {
		return nil
	}
	_, err := windows.VirtualAlloc(uintptr(unsafe.Pointer(&b[from])), uintptr(to-from), windows.MEM_COMMIT, windows.PAGE_READWRITE)
	return err
}

func memoryRelease(b []byte) error {
	if cap(b) == 0 {
		return nil
	}
	return windows.VirtualFree(uintptr(unsafe.Pointer(unsafe.SliceData(b))), 0, windows.MEM_RELEASE)
}
