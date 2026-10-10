package host

import (
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
	next atomic.Int32
}

// NewThreads returns the "wasi" module for instances over env and wasi.
func NewThreads(env *Env, wasi *WASI) *Threads {
	return &Threads{env: env, wasi: wasi}
}

// Xthread_spawn_6s4jie starts a thread, and returns its ID. wasi-libc keeps
// the thread's stack and TLS in arg.
func (t *Threads) Xthread_spawn_6s4jie(arg int32) int32 {
	id := t.next.Add(1)
	m := whisperwasm.New(t.env, t.wasi, t)
	go m.Xwasi_thread_start(id, arg)
	return id
}

var _ whisperwasm.Xwasi = (*Threads)(nil)
