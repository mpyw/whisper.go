package main

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
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

// writeWAV writes 1 s of 16-bit PCM at rate, whose channel c is sample(c,
// t) at t seconds, and returns its path. size is the data chunk's size, or
// 0 for the real one.
func writeWAV(t *testing.T, rate, channels int, size uint32, sample func(c int, t float64) float64) string {
	t.Helper()
	var data []byte
	for i := range rate {
		for c := range channels {
			data = binary.LittleEndian.AppendUint16(data, uint16(int16(math.Round(32767*sample(c, float64(i)/float64(rate))))))
		}
	}
	if size == 0 {
		size = uint32(len(data))
	}
	b := []byte("RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00")
	b = binary.LittleEndian.AppendUint16(b, 1) // PCM
	b = binary.LittleEndian.AppendUint16(b, uint16(channels))
	b = binary.LittleEndian.AppendUint32(b, uint32(rate))
	b = binary.LittleEndian.AppendUint32(b, uint32(rate*2*channels))
	b = binary.LittleEndian.AppendUint16(b, uint16(2*channels))
	b = binary.LittleEndian.AppendUint16(b, 16)
	b = append(b, "data"...)
	b = binary.LittleEndian.AppendUint32(b, size)
	b = append(b, data...)
	binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
	path := filepath.Join(t.TempDir(), "a.wav")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// sine is a sine of freq Hz, at amplitude 0.5.
func sine(freq float64) func(int, float64) float64 {
	return func(_ int, t float64) float64 { return 0.5 * math.Sin(2*math.Pi*freq*t) }
}

// rms is the root mean square of pcm, without 50 ms at each end.
func rms(pcm []float32) float64 {
	pcm = pcm[800 : len(pcm)-800]
	var sum float64
	for _, v := range pcm {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum / float64(len(pcm)))
}

// TestDecodeResample decodes 1 s of a 1 kHz sine at each rate, mono and
// stereo: it must stay 1 s of the same sine.
func TestDecodeResample(t *testing.T) {
	for _, rate := range []int{8000, 16000, 22050, 44100, 48000, 96000} {
		for _, channels := range []int{1, 2} {
			pcm, err := decode(writeWAV(t, rate, channels, 0, sine(1000)))
			if err != nil {
				t.Fatal(err)
			}
			if d := len(pcm) - whisper.SampleRate; d < -1 || d > 1 {
				t.Errorf("%d Hz, %d channels: %d samples, want %d", rate, channels, len(pcm), whisper.SampleRate)
			}
			if got, want := rms(pcm), 0.5/math.Sqrt2; math.Abs(got-want) > 0.01*want {
				t.Errorf("%d Hz, %d channels: RMS %.4f, want %.4f", rate, channels, got, want)
			}
		}
	}
}

// TestDecodeMix decodes a stereo WAV with a sine on the left only, at a rate
// for each of to16k's paths: the mono is their mean, half the sine.
func TestDecodeMix(t *testing.T) {
	for _, rate := range []int{8000, 16000, 44100} {
		pcm, err := decode(writeWAV(t, rate, 2, 0, func(c int, t float64) float64 {
			if c == 1 {
				return 0
			}
			return sine(1000)(c, t)
		}))
		if err != nil {
			t.Fatal(err)
		}
		if got, want := rms(pcm), 0.25/math.Sqrt2; math.Abs(got-want) > 0.01*want {
			t.Errorf("%d Hz: RMS %.4f, want %.4f", rate, got, want)
		}
	}
}

// TestDecodeAlias decodes sines at 44.1 kHz. Those above 8 kHz, half of
// whisper.SampleRate, must go, not fold into the band below.
func TestDecodeAlias(t *testing.T) {
	for _, tt := range []struct {
		freq     float64
		min, max float64 // of the RMS, as a part of the input's
	}{
		{6000, 0.99, 1.01},
		{9000, 0, 0.001},
		{12000, 0, 0.001},
		{20000, 0, 0.001},
	} {
		pcm, err := decode(writeWAV(t, 44100, 1, 0, sine(tt.freq)))
		if err != nil {
			t.Fatal(err)
		}
		if got := rms(pcm) / (0.5 / math.Sqrt2); got < tt.min || got > tt.max {
			t.Errorf("%v Hz: RMS %.4f of the input's, want %v to %v", tt.freq, got, tt.min, tt.max)
		}
	}
}

// residual is what is left of pcm, without 50 ms at each end, once the
// sine of freq Hz is taken out, as a part of that sine.
func residual(pcm []float32, freq float64) float64 {
	pcm = pcm[800 : len(pcm)-800]
	var re, im float64
	for i, v := range pcm {
		a := 2 * math.Pi * freq * float64(i) / whisper.SampleRate
		re += float64(v) * math.Cos(a)
		im += float64(v) * math.Sin(a)
	}
	re, im = 2*re/float64(len(pcm)), 2*im/float64(len(pcm))
	var rest float64
	for i, v := range pcm {
		a := 2 * math.Pi * freq * float64(i) / whisper.SampleRate
		d := float64(v) - re*math.Cos(a) - im*math.Sin(a)
		rest += d * d
	}
	return math.Sqrt(2*rest/float64(len(pcm))) / math.Hypot(re, im)
}

// TestDecodeImage decodes a 3 kHz sine at rates below whisper.SampleRate.
// beep.Resample adds copies of it, at 5 kHz from 8 kHz and at 1975 Hz from
// 11.025 kHz, which must go.
func TestDecodeImage(t *testing.T) {
	for _, rate := range []int{8000, 11025, 12000} {
		pcm, err := decode(writeWAV(t, rate, 1, 0, sine(3000)))
		if err != nil {
			t.Fatal(err)
		}
		if got := residual(pcm, 3000); got > 0.001 {
			t.Errorf("%d Hz: %.4f of the 3 kHz sine is something else, want 0.001 or less", rate, got)
		}
	}
}

func TestDecodeErrors(t *testing.T) {
	for _, tt := range []struct {
		name string
		path string
	}{
		{"6 channels", writeWAV(t, 16000, 6, 0, sine(1000))},
		// ffmpeg writing to a pipe leaves the size at 0xFFFFFFFF.
		{"the data's size unknown", writeWAV(t, 16000, 1, 0xFFFFFFFF, sine(1000))},
	} {
		if pcm, err := decode(tt.path); err == nil {
			t.Errorf("%s: %d samples, want an error", tt.name, len(pcm))
		}
	}
}
