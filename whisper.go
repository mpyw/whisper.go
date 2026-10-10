// Package whisper transcribes speech with whisper.cpp, in pure Go.
//
// whisper.cpp is built for WebAssembly and translated to Go source by
// wasm2go, so there is no cgo and no Wasm runtime. Each Model owns one
// instance of it, with its own memory.
package whisper

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"runtime"
	"sync"
	"time"

	whisperwasm "github.com/mpyw/whisper.go-wasm"

	"github.com/mpyw/whisper.go/internal/host"
)

// hostABIVersion is the whisperwasm.ABIVersion this host implements.
const hostABIVersion = 1

// SampleRate is the rate Transcribe expects: 16 kHz mono.
const SampleRate = 16000

// ErrClosed is returned by a Model after Close, or after a trap.
var ErrClosed = errors.New("whisper: model is closed")

// Options configures a Model.
type Options struct {
	// Log receives whisper.cpp's log. Nil discards it.
	Log io.Writer
}

// TranscribeOptions configures one Transcribe call.
type TranscribeOptions struct {
	// Language is a code such as "en" or "ja". Empty detects it with a
	// multilingual model, and means "en" with an English-only one.
	Language string
	// Threads is how many threads the model runs on. Zero means GOMAXPROCS.
	Threads int
}

// Result is a transcription.
type Result struct {
	// Language is the language transcribed, as given or detected.
	Language string
	Segments []Segment
}

// Segment is a span of the transcription.
type Segment struct {
	Start, End time.Duration
	Text       string
}

// Model is a loaded whisper.cpp model. Its methods may be called from any
// goroutine, and calls to Transcribe run one at a time.
type Model struct {
	mu  sync.Mutex
	mem *host.Memory
	env *host.Env
	mod *whisperwasm.Module
	ctx int32
}

// New loads a model in whisper.cpp's ggml format, such as ggml-base.bin.
func New(model []byte, opts *Options) (_ *Model, err error) {
	if whisperwasm.ABIVersion != hostABIVersion {
		return nil, fmt.Errorf("whisper: github.com/mpyw/whisper.go-wasm has ABI version %d, but this whisper.go implements %d; use matching versions", whisperwasm.ABIVersion, hostABIVersion)
	}
	if opts == nil {
		opts = &Options{}
	}
	mem, err := host.NewMemory()
	if err != nil {
		return nil, fmt.Errorf("whisper: %w", err)
	}
	m := &Model{mem: mem, env: host.NewEnv(mem)}
	defer func() {
		if err != nil {
			err = errors.Join(err, m.mem.Close())
		}
	}()
	wasi := host.NewWASI(mem, opts.Log)
	err = m.call(func() {
		m.mod = whisperwasm.New(m.env, wasi, host.NewThreads(m.env, wasi))
		m.mod.X_initialize()
		p := m.alloc(model)
		if p == 0 {
			return
		}
		m.ctx = m.mod.Xgw_init(p, int32(len(model)))
		m.mod.Xgw_free(p)
	})
	if err != nil {
		return nil, err
	}
	if m.ctx == 0 {
		return nil, errors.New("whisper: cannot load the model; see Options.Log")
	}
	return m, nil
}

// Transcribe transcribes pcm, mono samples at SampleRate in [-1, 1].
// Canceling ctx stops the run between two of ggml's operations.
func (m *Model) Transcribe(ctx context.Context, pcm []float32, opts *TranscribeOptions) (*Result, error) {
	if opts == nil {
		opts = &TranscribeOptions{}
	}
	threads := opts.Threads
	if threads <= 0 {
		threads = runtime.GOMAXPROCS(0)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.mod == nil {
		return nil, ErrClosed
	}
	m.env.SetContext(ctx)
	defer m.env.SetContext(context.Background())

	b := make([]byte, 4*len(pcm))
	for i, v := range pcm {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	var rc int32
	var res Result
	err := m.call(func() {
		samples, lang := m.alloc(b), m.alloc(append([]byte(opts.Language), 0))
		defer m.mod.Xgw_free(samples)
		defer m.mod.Xgw_free(lang)
		if samples == 0 && len(b) > 0 || lang == 0 {
			rc = -1
			return
		}
		rc = m.mod.Xgw_full(m.ctx, samples, int32(len(pcm)), int32(threads), lang)
		if rc != 0 {
			return
		}
		res.Language = m.str(m.mod.Xgw_lang(m.ctx))
		for i := range m.mod.Xgw_n_segments(m.ctx) {
			res.Segments = append(res.Segments, Segment{
				// whisper.cpp counts time in units of 10 ms.
				Start: time.Duration(m.mod.Xgw_segment_t0(m.ctx, i)) * 10 * time.Millisecond,
				End:   time.Duration(m.mod.Xgw_segment_t1(m.ctx, i)) * 10 * time.Millisecond,
				Text:  m.str(m.mod.Xgw_segment_text(m.ctx, i)),
			})
		}
	})
	switch {
	case err != nil:
		return nil, err
	case rc != 0 && ctx.Err() != nil:
		return nil, ctx.Err()
	case rc != 0:
		return nil, fmt.Errorf("whisper: whisper_full returned %d; see Options.Log", rc)
	}
	return &res, nil
}

// Close frees the model. Calls after it return ErrClosed.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.mod == nil {
		return nil
	}
	err := m.call(func() { m.mod.Xgw_free_ctx(m.ctx) })
	m.mod = nil
	return errors.Join(err, m.mem.Close())
}

// call runs f, which calls into the module. A trap there, such as ggml's
// abort, panics in the generated code. The module's memory may then be
// inconsistent, so the model is closed.
func (m *Model) call(f func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			m.mod = nil
			err = fmt.Errorf("whisper: the model trapped and is closed: %v", r)
		}
	}()
	f()
	return nil
}

// alloc copies b into the module's memory, and returns its address, or 0.
func (m *Model) alloc(b []byte) int32 {
	p := m.mod.Xgw_malloc(int32(len(b)))
	if p != 0 {
		copy(m.mem.Bytes()[p:], b)
	}
	return p
}

// str reads the NUL-terminated string at p.
func (m *Model) str(p int32) string {
	b := m.mem.Bytes()[p:]
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}
