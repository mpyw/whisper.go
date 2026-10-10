package host

import (
	"runtime/debug"
	"sync"
	"sync/atomic"

	whisperwasm "github.com/mpyw/whisper.go-wasm"
)

// Threads implements wasi-threads. Each thread is a new instance of the
// module over the same memory, run on its own goroutine.
//
// A trap in a thread is not recovered. The thread that started it would
// wait for it at a barrier forever, so ending the program is the lesser harm.
type Threads struct {
	env  *Env
	wasi *WASI
	next atomic.Uint32
	live sync.WaitGroup
}

// NewThreads returns the "wasi" module for instances over env and wasi.
func NewThreads(env *Env, wasi *WASI) *Threads {
	return &Threads{env: env, wasi: wasi}
}

// Xthread_spawn_6s4jie starts a thread, and returns its ID. wasi-libc keeps
// the thread's stack and TLS in arg.
func (t *Threads) Xthread_spawn_6s4jie(arg int32) int32 {
	id := t.nextID()
	m := whisperwasm.New(t.env, t.wasi, t)
	t.live.Go(func() {
		// A page past the committed memory faults: make it a panic, as a
		// bounds check would have.
		debug.SetPanicOnFault(true)
		m.Xwasi_thread_start(id, arg)
	})
	return id
}

// nextID is the ID of a new thread. wasi-threads' IDs are 1 to 0x1FFFFFFF.
// ggml starts threads for every graph, hundreds a second, so IDs come round
// again: at most 31 are alive at once, and each is joined before long.
func (t *Threads) nextID() int32 {
	return int32((t.next.Add(1)-1)%threadMaxID) + 1
}

// threadMaxID is the largest thread ID of wasi-threads.
const threadMaxID = 0x1FFFFFFF

// Wait returns once every thread has returned. A thread still writes to
// the memory after the one that joins it wakes, so the memory may be
// released only after Wait.
func (t *Threads) Wait() { t.live.Wait() }

var _ whisperwasm.Xwasi = (*Threads)(nil)
