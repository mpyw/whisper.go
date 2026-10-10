// Package host implements what whisper.wasm imports: its shared linear
// memory, part of WASI preview 1, wasi-threads, and ggml's kernels.
package host

import (
	"errors"
	"strconv"
	"sync"

	whisperwasm "github.com/mpyw/whisper.go-wasm"
)

const (
	memoryPageSize = 1 << 16
	// memoryMaxPages is the 4 GiB a 32-bit address reaches.
	memoryMaxPages = 1 << 16
)

// Memory is the linear memory every thread of a module shares. Its whole
// address space is reserved up front, and its slice spans all of it from
// the start. So the slice never changes while threads read it: Grow only
// commits more pages, and counts them.
//
// A page past the committed ones faults when touched. Every goroutine that
// runs the module turns faults into panics, which are traps.
type Memory struct {
	buf     []byte
	waiters sync.Map

	mu    sync.Mutex
	pages int64 // committed
}

// NewMemory reserves the address space and commits the initial pages.
func NewMemory() (*Memory, error) {
	if strconv.IntSize < 64 {
		return nil, errors.New("host: the module's 4 GiB of address space needs a 64-bit host")
	}
	size := uint64(memoryMaxPages) * memoryPageSize // a variable: 1<<32 overflows a 32-bit int
	b, err := memoryReserve(int(size))
	if err != nil {
		return nil, err
	}
	m := &Memory{buf: b}
	if m.Grow(whisperwasm.InitialPages, memoryMaxPages) < 0 {
		return nil, errors.Join(errors.New("host: cannot commit the initial memory"), m.Close())
	}
	return m, nil
}

// Bytes is the whole reservation. Only its committed pages may be touched.
func (m *Memory) Bytes() []byte { return m.buf }

// Slice is the whisperwasm.Memory method that generated code reads the
// memory through.
func (m *Memory) Slice() *[]byte { return &m.buf }

// Waiters is the whisperwasm.Memory method for memory.atomic.wait and
// notify.
func (m *Memory) Waiters() *sync.Map { return &m.waiters }

// Grow commits delta more pages, and returns the old size in pages, or -1.
// The module's memory.size is Grow(0).
func (m *Memory) Grow(delta, maxPages int64) int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	old := m.pages
	n := old + delta
	if delta < 0 || n > min(maxPages, memoryMaxPages) {
		return -1
	}
	if err := memoryCommit(m.buf, int(old*memoryPageSize), int(n*memoryPageSize)); err != nil {
		return -1
	}
	m.pages = n
	return old
}

// Close releases the address space. No thread may use the memory after.
func (m *Memory) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	b := m.buf
	m.buf, m.pages = nil, 0
	return memoryRelease(b)
}

// MemoryAddr is the offset of a module address in Bytes. The module passes
// addresses as i32, so one at 2 GiB or above is negative until read as
// unsigned.
func MemoryAddr(p int32) int { return int(uint32(p)) }

var _ whisperwasm.Memory = (*Memory)(nil)
