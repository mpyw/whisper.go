// Package host implements what whisper.wasm imports: its shared linear
// memory, part of WASI preview 1, and wasi-threads.
package host

import (
	"errors"
	"math"
	"sync"

	whisperwasm "github.com/mpyw/whisper.go-wasm"
)

const (
	memoryPageSize = 1 << 16
	// memoryMaxPages is the 4 GiB a 32-bit address reaches.
	memoryMaxPages = 1 << 16
)

// Memory is the linear memory every thread of a module shares. Its address
// space is reserved in full up front, so it never moves while threads read it.
type Memory struct {
	mu      sync.Mutex
	buf     []byte
	waiters sync.Map
}

// NewMemory reserves the address space and commits the initial pages.
func NewMemory() (*Memory, error) {
	// A 32-bit host cannot address 4 GiB. Reserve what an int reaches.
	size := uint64(memoryMaxPages) * memoryPageSize
	size = min(size, uint64(math.MaxInt)&^(memoryPageSize-1))
	b, err := memoryReserve(int(size))
	if err != nil {
		return nil, err
	}
	m := &Memory{buf: b[:0]}
	if m.Grow(whisperwasm.InitialPages, memoryMaxPages) < 0 {
		return nil, errors.Join(errors.New("host: cannot commit the initial memory"), m.Close())
	}
	return m, nil
}

// Bytes is the memory as it is now. It stays valid after Grow, but does not
// reach the pages Grow adds.
func (m *Memory) Bytes() []byte { return m.buf }

// Slice is the whisperwasm.Memory method generated code reads the memory through.
func (m *Memory) Slice() *[]byte { return &m.buf }

// Waiters is the whisperwasm.Memory method for memory.atomic.wait and notify.
func (m *Memory) Waiters() *sync.Map { return &m.waiters }

// Grow commits delta more pages, and returns the old size in pages, or -1.
func (m *Memory) Grow(delta, maxPages int64) int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	old := int64(len(m.buf) / memoryPageSize)
	n := old + delta
	if delta < 0 || n > min(maxPages, int64(cap(m.buf)/memoryPageSize)) {
		return -1
	}
	if err := memoryCommit(m.buf[:cap(m.buf)], len(m.buf), int(n*memoryPageSize)); err != nil {
		return -1
	}
	m.buf = m.buf[:n*memoryPageSize]
	return old
}

// Close releases the address space. No thread may use the memory after.
func (m *Memory) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	b := m.buf[:cap(m.buf)]
	m.buf = nil
	return memoryRelease(b)
}

var _ whisperwasm.Memory = (*Memory)(nil)
