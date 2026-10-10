package whisper_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mpyw/whisper.go"
)

// The model the tests run: tiny.en, quantized to 5 bits, 32 MB. It is
// downloaded once, at a fixed revision, and checked against its digest.
const (
	testModelURL    = "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.en-q5_1.bin"
	testModelSHA256 = "c77c5766f1cef09b6b7d47f21b546cbddd4157886b3b5d6d4f709e91e66c7c2b"
)

// testModel returns the model's bytes. WHISPER_TEST_MODEL names a model on
// disk, read as it is, such as another one to benchmark. Otherwise the
// pinned one is kept in the user cache directory.
func testModel(t testing.TB) []byte {
	t.Helper()
	if path := os.Getenv("WHISPER_TEST_MODEL"); path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	dir, err := os.UserCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "whisper.go-test", testModelSHA256+".bin")
	if b, err := os.ReadFile(path); err == nil && testDigest(b) == testModelSHA256 {
		return b
	}
	b := testDownload(t)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return b
}

func testDownload(t testing.TB) []byte {
	t.Helper()
	resp, err := http.Get(testModelURL)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }() // the body was read in full, or the test failed
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET %s: %s", testModelURL, resp.Status)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if d := testDigest(b); d != testModelSHA256 {
		t.Fatalf("GET %s: SHA-256 %s, want %s", testModelURL, d, testModelSHA256)
	}
	return b
}

func testDigest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// testWAV reads a 16-bit mono WAV file at whisper.SampleRate.
func testWAV(t testing.TB, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if binary.LittleEndian.Uint32(b[24:]) != whisper.SampleRate || binary.LittleEndian.Uint16(b[22:]) != 1 {
		t.Fatalf("%s is not mono at %d Hz", path, whisper.SampleRate)
	}
	i := 12
	for string(b[i:i+4]) != "data" {
		i += 8 + int(binary.LittleEndian.Uint32(b[i+4:]))
	}
	data := b[i+8 : i+8+int(binary.LittleEndian.Uint32(b[i+4:]))]
	pcm := make([]float32, len(data)/2)
	for j := range pcm {
		pcm[j] = float32(int16(binary.LittleEndian.Uint16(data[2*j:]))) / 32768
	}
	return pcm
}

func testNew(t testing.TB) *whisper.Model {
	t.Helper()
	m, err := whisper.New(testModel(t), &whisper.Options{Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	})
	return m
}

func TestTranscribe(t *testing.T) {
	m := testNew(t)
	res, err := m.Transcribe(context.Background(), testWAV(t, "testdata/jfk.wav"), &whisper.TranscribeOptions{Threads: 4})
	if err != nil {
		t.Fatal(err)
	}
	if res.Language != "en" {
		t.Errorf("Language = %q, want en", res.Language)
	}
	var text string
	var end time.Duration
	for _, s := range res.Segments {
		if s.Start < end || s.End < s.Start {
			t.Errorf("segment out of order: %+v after %v", s, end)
		}
		end = s.End
		text += s.Text
	}
	const want = "ask not what your country can do for you"
	if !strings.Contains(text, want) {
		t.Errorf("text = %q, want it to contain %q", text, want)
	}
	if end < 10*time.Second || end > 11*time.Second {
		t.Errorf("the last segment ends at %v, want about 11 s", end)
	}
}

func TestTranscribeCanceled(t *testing.T) {
	m := testNew(t)
	pcm := testWAV(t, "testdata/jfk.wav")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Transcribe(ctx, pcm, nil); !errors.Is(err, context.Canceled) {
		t.Errorf("before the run: err = %v, want context.Canceled", err)
	}

	// One thread takes several seconds, so the deadline falls inside the run.
	ctx, cancel = context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := m.Transcribe(ctx, pcm, &whisper.TranscribeOptions{Threads: 1}); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("during the run: err = %v, want context.DeadlineExceeded", err)
	}
	if d := time.Since(start); d > 3*time.Second*testSlowdown {
		t.Errorf("the canceled run took %v", d)
	}

	// The model is still usable.
	res, err := m.Transcribe(context.Background(), pcm[:3*whisper.SampleRate], &whisper.TranscribeOptions{Threads: 4})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Segments) == 0 {
		t.Error("no segments after a canceled run")
	}
}

// testCanceledAfter is a context canceled from its nth call to Err on.
// The module asks for Err before each pass of the encoder, so this cancels
// at a chosen point, without a race against time.
type testCanceledAfter struct {
	context.Context
	n     int32
	calls atomic.Int32
}

func (c *testCanceledAfter) Err() error {
	if c.calls.Add(1) >= c.n {
		return context.Canceled
	}
	return nil
}

// TestTranscribeCanceledAtEncoder cancels when the first pass of the
// encoder begins, where whisper.cpp stops with success and no segments.
func TestTranscribeCanceledAtEncoder(t *testing.T) {
	m := testNew(t)
	pcm := testWAV(t, "testdata/jfk.wav")
	for n := int32(2); n <= 4; n++ {
		ctx := &testCanceledAfter{Context: context.Background(), n: n}
		if res, err := m.Transcribe(ctx, pcm, &whisper.TranscribeOptions{Threads: 2}); !errors.Is(err, context.Canceled) {
			t.Errorf("canceled at Err call %d: res = %+v, err = %v, want context.Canceled", n, res, err)
		}
	}
}

// TestTranscribeCanceledDuringMel cancels while the mel spectrogram of an
// hour of audio is made, which takes about 4 s. Its threads must notice.
func TestTranscribeCanceledDuringMel(t *testing.T) {
	m := testNew(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	time.AfterFunc(20*time.Millisecond, cancel)
	start := time.Now()
	if _, err := m.Transcribe(ctx, make([]float32, 3600*whisper.SampleRate), &whisper.TranscribeOptions{Threads: 4}); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if d := time.Since(start); d > 1500*time.Millisecond*testSlowdown {
		t.Errorf("the canceled run took %v", d)
	}
}

// TestTranscribeCanceledWhileWaiting cancels a call that waits for
// another one. It must return at once, not when the other ends.
func TestTranscribeCanceledWhileWaiting(t *testing.T) {
	m := testNew(t)
	pcm := testWAV(t, "testdata/jfk.wav")
	started := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		// One thread takes several seconds. Its result is not the test's.
		ctx := &testStarted{Context: context.Background(), started: started}
		_, _ = m.Transcribe(ctx, pcm, &whisper.TranscribeOptions{Threads: 1})
	}()
	<-started
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := m.Transcribe(ctx, pcm[:whisper.SampleRate], nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want context.DeadlineExceeded", err)
	}
	if d := time.Since(start); d > time.Second*testSlowdown {
		t.Errorf("the waiting call returned after %v", d)
	}
	<-done
}

// testStarted is a context that reports the module's first call to Err,
// once the run holds the model.
type testStarted struct {
	context.Context
	started chan struct{}
	once    atomic.Bool
}

func (c *testStarted) Err() error {
	if !c.once.Swap(true) {
		close(c.started)
	}
	return nil
}

// testTrapAfter is a context whose nth call to Err panics: in the module,
// in the middle of a graph, where a trap of ggml's would be.
type testTrapAfter struct {
	context.Context
	n     int32
	calls atomic.Int32
}

func (c *testTrapAfter) Err() error {
	if c.calls.Add(1) == c.n {
		panic("trap")
	}
	return nil
}

// TestTranscribeTrap traps while ggml's threads wait for the caller. The
// model closes, and the threads that spin end rather than spin forever.
func TestTranscribeTrap(t *testing.T) {
	m := testNew(t)
	before := runtime.NumGoroutine()
	ctx := &testTrapAfter{Context: context.Background(), n: 20}
	if _, err := m.Transcribe(ctx, testWAV(t, "testdata/jfk.wav"), &whisper.TranscribeOptions{Threads: 8}); !errors.Is(err, whisper.ErrClosed) {
		t.Fatalf("err = %v, want ErrClosed", err)
	}
	for deadline := time.Now().Add(5 * time.Second * testSlowdown); runtime.NumGoroutine() > before; time.Sleep(10 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("%d goroutines, %d before the trap", runtime.NumGoroutine(), before)
		}
	}
	if _, err := m.Transcribe(context.Background(), make([]float32, whisper.SampleRate), nil); !errors.Is(err, whisper.ErrClosed) {
		t.Errorf("after the trap: err = %v, want ErrClosed", err)
	}
	if err := m.Close(); err != nil {
		t.Errorf("Close after the trap: %v", err)
	}
}

// TestTranscribeAuto checks that "auto" means "en" with an English-only
// model, as an empty language does.
func TestTranscribeAuto(t *testing.T) {
	m := testNew(t)
	res, err := m.Transcribe(context.Background(), testWAV(t, "testdata/jfk.wav"), &whisper.TranscribeOptions{Language: "auto", Threads: 4})
	if err != nil {
		t.Fatal(err)
	}
	if res.Language != "en" {
		t.Errorf("Language = %q, want en", res.Language)
	}
}

func TestTranscribeUnknownLanguage(t *testing.T) {
	m := testNew(t)
	_, err := m.Transcribe(context.Background(), testWAV(t, "testdata/jfk.wav"), &whisper.TranscribeOptions{Language: "xx"})
	if err == nil || !strings.Contains(err.Error(), `unknown language "xx"`) {
		t.Errorf("err = %v, want an unknown language", err)
	}
	// The model is still usable.
	if _, err := m.Transcribe(context.Background(), make([]float32, whisper.SampleRate), nil); err != nil {
		t.Error(err)
	}
}

// TestTranscribeEmpty checks that no samples give no segments, rather than
// the previous call's: whisper.cpp keeps its last mel spectrogram.
func TestTranscribeEmpty(t *testing.T) {
	m := testNew(t)
	if _, err := m.Transcribe(context.Background(), testWAV(t, "testdata/jfk.wav"), &whisper.TranscribeOptions{Threads: 4}); err != nil {
		t.Fatal(err)
	}
	res, err := m.Transcribe(context.Background(), nil, nil)
	if err != nil || len(res.Segments) != 0 {
		t.Errorf("res = %+v, err = %v, want no segments", res, err)
	}
}

// TestTranscribeTooLong passes 10 minutes of silence to a model whose
// memory has room for less. It must fail and leave the model usable.
func TestTranscribeTooLong(t *testing.T) {
	if testSlowdown > 1 {
		t.Skip("it takes a minute under the race detector, and runs nothing at once")
	}
	m := testNew(t)
	free := m.Fill(64 << 20)
	if _, err := m.Transcribe(context.Background(), make([]float32, 10*60*whisper.SampleRate), nil); !errors.Is(err, whisper.ErrTooLong) {
		t.Fatalf("err = %v, want ErrTooLong", err)
	}
	free()
	if _, err := m.Transcribe(context.Background(), make([]float32, whisper.SampleRate), nil); err != nil {
		t.Error(err)
	}
}

func TestNewNotAModel(t *testing.T) {
	m, err := whisper.New([]byte("not a ggml model"), nil)
	if err == nil {
		_ = m.Close() // the test failed already
		t.Fatal("New accepted bytes that are not a model")
	}
}

func TestClose(t *testing.T) {
	m, err := whisper.New(testModel(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if _, err := m.Transcribe(context.Background(), make([]float32, whisper.SampleRate), nil); !errors.Is(err, whisper.ErrClosed) {
		t.Errorf("Transcribe after Close: err = %v, want ErrClosed", err)
	}
}

func Example() {
	model, err := os.ReadFile("ggml-base.en.bin")
	if err != nil {
		panic(err)
	}
	m, err := whisper.New(model, nil)
	if err != nil {
		panic(err)
	}
	defer func() { _ = m.Close() }() // nothing to report at exit

	var pcm []float32 // mono, 16 kHz, in [-1, 1]
	res, err := m.Transcribe(context.Background(), pcm, &whisper.TranscribeOptions{Language: "en"})
	if err != nil {
		panic(err)
	}
	for _, s := range res.Segments {
		_, _ = fmt.Printf("[%v -> %v] %s\n", s.Start, s.End, s.Text) // a failed write to stdout is not the example's concern
	}
}

// BenchmarkTranscribe transcribes testdata/jfk.wav, 11 s of speech.
func BenchmarkTranscribe(b *testing.B) {
	m := testNew(b)
	pcm := testWAV(b, "testdata/jfk.wav")
	for _, threads := range []int{1, 4, 8} {
		b.Run(fmt.Sprintf("threads=%d", threads), func(b *testing.B) {
			for b.Loop() {
				if _, err := m.Transcribe(context.Background(), pcm, &whisper.TranscribeOptions{Threads: threads}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestNewTrapFreesMemory loads a model whose first word claims 2 GiB, which
// traps in gw_init. The memory of the model must not stay.
func TestNewTrapFreesMemory(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("it reads the RSS with ps")
	}
	b := testModel(t)
	d := append([]byte(nil), b...)
	// The magic and 11 hparams, the filters' sizes and values, the vocab size.
	nMel, nFFT := binary.LittleEndian.Uint32(d[48:]), binary.LittleEndian.Uint32(d[52:])
	binary.LittleEndian.PutUint32(d[56+4*nMel*nFFT+4:], 0x7fffffff)
	before := testRSS(t)
	for range 4 {
		if _, err := whisper.New(d, &whisper.Options{Log: io.Discard}); !errors.Is(err, whisper.ErrClosed) {
			t.Fatalf("err = %v, want a trap", err)
		}
	}
	for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(10 * time.Millisecond) {
		grew := testRSS(t) - before
		if grew < 512<<20 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the RSS grew %d MiB over 4 models that trapped", grew>>20)
		}
	}
}

// testRSS is the process's resident set, in bytes.
func testRSS(t *testing.T) int {
	t.Helper()
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		t.Fatal(err)
	}
	kb, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil {
		t.Fatal(err)
	}
	return kb << 10
}

// TestTranscribeLanguageNUL passes a language with a NUL inside. The module
// read it up to the NUL, as "en", and returned "en" for it.
func TestTranscribeLanguageNUL(t *testing.T) {
	m := testNew(t)
	if _, err := m.Transcribe(context.Background(), make([]float32, whisper.SampleRate), &whisper.TranscribeOptions{Language: "en\x00xx"}); err == nil {
		t.Error("Transcribe accepted a language with a NUL")
	}
}
