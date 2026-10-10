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
	"runtime/debug"
	"strings"
	"time"

	whisperwasm "github.com/mpyw/whisper.go-wasm"

	"github.com/mpyw/whisper.go/internal/host"
)

// hostABIVersion is the whisperwasm.ABIVersion this host implements.
const hostABIVersion = 2

// Both differences are unsigned constants, so a whisper.go-wasm of another
// ABI version fails to compile: "constant overflows uint".
const (
	_ uint = whisperwasm.ABIVersion - hostABIVersion
	_ uint = hostABIVersion - whisperwasm.ABIVersion
)

// maxThreads is the most threads gw_full runs on (ABI.md).
const maxThreads = 32

// gw_full's results of its own (ABI.md): a language whisper.cpp does not
// know, and audio that the module's memory has no room for.
const (
	gwUnknownLanguage = -100
	gwNoRoom          = -101
)

// SampleRate is the rate Transcribe expects: 16 kHz mono.
const SampleRate = 16000

// ErrClosed is returned by a Model after Close, or after a trap. The error
// of the call that trapped wraps it too.
var ErrClosed = errors.New("whisper: model is closed")

// ErrTooLong is Transcribe's error for audio the module's memory cannot
// hold. The model stays usable: split the audio, and transcribe the parts.
var ErrTooLong = errors.New("whisper: the audio is too long for the module's memory; split it")

// Options configures a Model.
type Options struct {
	// Log receives whisper.cpp's log. Nil discards it.
	Log io.Writer
}

// TranscribeOptions configures one Transcribe call.
type TranscribeOptions struct {
	// Language is a code such as "en" or "ja", or a lowercase name such as
	// "japanese". Empty, or "auto", detects it
	// with a multilingual model. An English-only model transcribes English
	// whatever it says.
	Language string
	// Threads is how many threads the model runs on. Zero, a negative
	// number, or more than GOMAXPROCS means GOMAXPROCS, and 32 is the most. ggml's threads spin
	// while they wait for each other, so a thread without a CPU of its own
	// slows all of them. Each takes 8 MiB of the module's memory.
	Threads int
}

// Result is a transcription.
type Result struct {
	// Language is the code of the language transcribed, given or detected.
	Language string
	Segments []Segment
}

// Segment is a span of the transcription.
type Segment struct {
	Start, End time.Duration
	Text       string
}

// Model is a loaded whisper.cpp model. Its methods may be called from any
// goroutine. Calls to Transcribe run one at a time, and Close waits for the
// one in progress.
type Model struct {
	sem     chan struct{} // holds one value while a call runs
	mem     *host.Memory
	env     *host.Env
	threads *host.Threads
	wasi    *host.WASI
	mod     *whisperwasm.Module // nil once closed or trapped
	ctx     int32
}

// New loads a model in whisper.cpp's ggml format, such as ggml-base.bin.
func New(model []byte, opts *Options) (_ *Model, err error) {
	// The module's sizes are int32, and its memory 4 GiB.
	if len(model) > math.MaxInt32 {
		return nil, errors.New("whisper: the model is larger than the module's memory")
	}
	if opts == nil {
		opts = &Options{}
	}
	mem, err := host.NewMemory()
	if err != nil {
		return nil, fmt.Errorf("whisper: %w", err)
	}
	m := &Model{sem: make(chan struct{}, 1), mem: mem, env: host.NewEnv(mem)}
	m.wasi = host.NewWASI(mem, opts.Log)
	m.threads = host.NewThreads(m.env, m.wasi)
	err = m.call(func() {
		m.mod = whisperwasm.New(m.env, m.wasi, m.threads)
		m.mod.X_initialize()
		p := m.alloc(model)
		if p == 0 {
			return
		}
		m.ctx = m.mod.Xgw_init(p, int32(len(model)))
		m.mod.Xgw_free(p)
	})
	if err != nil {
		// The trap frees the memory itself: see call.
		return nil, err
	}
	if m.ctx == 0 {
		return nil, errors.Join(errors.New("whisper: cannot load the model; see Options.Log"), m.release())
	}
	return m, nil
}

// Transcribe transcribes pcm, mono samples at SampleRate in [-1, 1].
// Samples outside are clipped, and NaN reads as 0. No samples give an
// empty Result, with no language, whatever opts says. Under 100 ms give no
// segments, and no language if it was to be detected.
// Canceling ctx stops the run within one of ggml's operations, and its
// error is the result.
func (m *Model) Transcribe(ctx context.Context, pcm []float32, opts *TranscribeOptions) (*Result, error) {
	// The module's sizes are int32: this is about 9 hours. With a large
	// model, its memory runs out before, which gw_full reports as gwNoRoom.
	if len(pcm) > math.MaxInt32/4 {
		return nil, ErrTooLong
	}
	if opts == nil {
		opts = &TranscribeOptions{}
	}
	// The module reads the language up to a NUL, so "en\x00xx" was "en".
	if strings.IndexByte(opts.Language, 0) >= 0 {
		return nil, fmt.Errorf("whisper: unknown language %q", opts.Language)
	}
	threads := min(runtime.GOMAXPROCS(0), maxThreads)
	if opts.Threads > 0 {
		threads = min(threads, opts.Threads)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	select {
	case m.sem <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-m.sem }()
	if m.mod == nil {
		return nil, ErrClosed
	}
	// ctx may have ended while this call waited for the one before.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// whisper.cpp makes no mel spectrogram for no samples, and would read
	// the previous call's.
	if len(pcm) == 0 {
		return &Result{}, nil
	}
	m.env.SetContext(ctx)
	defer m.env.SetContext(context.Background())

	var rc int32
	var res Result
	err := m.call(func() {
		samples := m.mod.Xgw_malloc(int32(4 * len(pcm)))
		lang := m.alloc(append([]byte(opts.Language), 0))
		if samples == 0 || lang == 0 {
			m.mod.Xgw_free(samples)
			m.mod.Xgw_free(lang)
			rc = gwNoRoom
			return
		}
		b := m.mem.Bytes()[host.MemoryAddr(samples):]
		for i, v := range pcm {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
		}
		// Not deferred: after a trap, the module must not be called again.
		rc = m.mod.Xgw_full(m.ctx, samples, int32(len(pcm)), int32(threads), lang)
		m.mod.Xgw_free(samples)
		m.mod.Xgw_free(lang)
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
	// A cancel before a pass of the encoder ends whisper_full early, with
	// success and part of the result. So ctx decides, whatever rc says.
	switch {
	case err != nil:
		return nil, err
	case ctx.Err() != nil:
		return nil, ctx.Err()
	case rc == gwUnknownLanguage:
		return nil, fmt.Errorf("whisper: unknown language %q", opts.Language)
	case rc == gwNoRoom:
		return nil, ErrTooLong
	case rc != 0:
		return nil, fmt.Errorf("whisper: whisper_full returned %d; see Options.Log", rc)
	}
	return &res, nil
}

// Close frees the model and its memory. Calls after it return ErrClosed.
//
// After a trap, Close frees nothing: the memory goes once the threads the
// trap left in the module have ended, which one waiting for good never does.
func (m *Model) Close() error {
	m.sem <- struct{}{}
	defer func() { <-m.sem }()
	if m.mod == nil {
		return nil
	}
	if err := m.call(func() { m.mod.Xgw_free_ctx(m.ctx) }); err != nil {
		return err
	}
	return m.release()
}

// release waits for the module's threads, then frees its memory.
func (m *Model) release() error {
	m.mod = nil
	m.threads.Wait()
	return m.mem.Close()
}

// call runs f, which calls into the module. A trap there, such as ggml's
// abort, panics in the generated code. A fault on an uncommitted page is a
// panic too. The module's memory may then be inconsistent, so the model is
// closed.
func (m *Model) call(f func()) (err error) {
	defer debug.SetPanicOnFault(debug.SetPanicOnFault(true))
	defer func() {
		if r := recover(); r != nil {
			m.mod = nil
			m.wasi.Stop()
			// The memory goes once every thread has ended. Only this
			// goroutine starts threads, so none starts after. One that
			// waits in the module for good keeps the memory, as before.
			go func() {
				m.threads.Wait()
				_ = m.mem.Close() // nothing is left to report the error to
			}()
			err = fmt.Errorf("%w: the model trapped: %v", ErrClosed, r)
		}
	}()
	f()
	return nil
}

// alloc copies b into the module's memory, and returns its address, or 0.
func (m *Model) alloc(b []byte) int32 {
	p := m.mod.Xgw_malloc(int32(len(b)))
	if p != 0 {
		copy(m.mem.Bytes()[host.MemoryAddr(p):], b)
	}
	return p
}

// str reads the NUL-terminated string at p.
func (m *Model) str(p int32) string {
	b := m.mem.Bytes()[host.MemoryAddr(p):]
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}
