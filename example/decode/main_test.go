package main

import (
	"math"
	"testing"

	"github.com/mpyw/whisper.go"
)

// TestDecode decodes testdata/jfk.wav, already 16 kHz and mono: resampling
// it to the same rate keeps its length.
func TestDecode(t *testing.T) {
	pcm, err := decode("../../testdata/jfk.wav")
	if err != nil {
		t.Fatal(err)
	}
	if want := 176000; len(pcm) != want {
		t.Errorf("%d samples, want %d: 11 s at %d Hz", len(pcm), want, whisper.SampleRate)
	}
	var peak float64
	for _, v := range pcm {
		peak = max(peak, math.Abs(float64(v)))
	}
	if peak < 0.1 || peak > 1 {
		t.Errorf("peak %v, want speech within [-1, 1]", peak)
	}
}

func TestDecodeUnknown(t *testing.T) {
	if _, err := decode("main.go"); err == nil {
		t.Error("decoded a .go file")
	}
}
