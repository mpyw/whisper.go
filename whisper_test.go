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
	"path/filepath"
	"strings"
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

// testModel returns the model's bytes. WHISPER_TEST_MODEL names a copy on
// disk. Otherwise it is kept in the user cache directory.
func testModel(t *testing.T) []byte {
	t.Helper()
	path := os.Getenv("WHISPER_TEST_MODEL")
	if path == "" {
		dir, err := os.UserCacheDir()
		if err != nil {
			t.Fatal(err)
		}
		path = filepath.Join(dir, "whisper.go-test", testModelSHA256+".bin")
	}
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

func testDownload(t *testing.T) []byte {
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
func testWAV(t *testing.T, path string) []float32 {
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

func testNew(t *testing.T) *whisper.Model {
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
	if d := time.Since(start); d > 3*time.Second {
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
