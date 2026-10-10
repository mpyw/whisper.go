package host

import (
	"context"
	"sync/atomic"

	whisperwasm "github.com/mpyw/whisper.go-wasm"
)

// Env is whisper.wasm's "env" module: its memory, whether the run in
// progress is canceled, and ggml's kernels.
type Env struct {
	*Kernels
	mem *Memory
	ctx atomic.Pointer[context.Context]
}

// NewEnv returns the env module over mem.
func NewEnv(mem *Memory) *Env { return &Env{Kernels: newKernels(mem), mem: mem} }

// SetContext makes gw_aborted report ctx's cancellation.
func (e *Env) SetContext(ctx context.Context) { e.ctx.Store(&ctx) }

// Xmemory is the module's shared memory.
func (e *Env) Xmemory() whisperwasm.Memory { return e.mem }

// Xgw_aborted is called between ggml's operations, from a graph's first
// thread, and before each pass of the encoder.
func (e *Env) Xgw_aborted() int32 {
	if p := e.ctx.Load(); p != nil && (*p).Err() != nil {
		return 1
	}
	return 0
}

var _ whisperwasm.Xenv = (*Env)(nil)
