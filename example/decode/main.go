// Command decode transcribes an audio file: WAV, MP3, FLAC or Ogg Vorbis, at
// any sample rate, mono or stereo. beep decodes and resamples it, in pure
// Go, to the samples whisper.go takes.
//
//	mise run transcribe -- speech.mp3
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/gopxl/beep/v2"
	"github.com/gopxl/beep/v2/flac"
	"github.com/gopxl/beep/v2/mp3"
	"github.com/gopxl/beep/v2/vorbis"
	"github.com/gopxl/beep/v2/wav"

	"github.com/mpyw/whisper.go"
)

func main() {
	if len(os.Args) != 3 {
		log.Fatal("usage: go run . model.bin audio.(wav|mp3|flac|ogg)")
	}
	model, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	m, err := whisper.New(model, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()

	pcm, err := decode(os.Args[2])
	if err != nil {
		log.Fatal(err)
	}
	res, err := m.Transcribe(context.Background(), pcm, nil)
	if err != nil {
		log.Fatal(err)
	}
	for _, s := range res.Segments {
		fmt.Printf("[%v -> %v] %s\n", s.Start, s.End, s.Text)
	}
}

// decode reads an audio file, and returns its samples as whisper.go takes
// them: mono, at whisper.SampleRate, in [-1, 1].
func decode(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var s beep.StreamSeekCloser
	var format beep.Format
	switch ext := strings.ToLower(filepath.Ext(path)); ext {
	case ".wav":
		s, format, err = wav.Decode(f)
	case ".mp3":
		s, format, err = mp3.Decode(f)
	case ".flac":
		s, format, err = flac.Decode(f)
	case ".ogg":
		s, format, err = vorbis.Decode(f)
	default:
		err = fmt.Errorf("cannot decode %s files", ext)
	}
	if err != nil {
		_ = f.Close() // the error to report is the decoder's
		return nil, err
	}
	defer s.Close() // it closes f too
	// beep reads the first 2 channels of a WAV or FLAC file, and the front
	// left and right of Ogg Vorbis. Speech in a center channel would be lost.
	if format.NumChannels > 2 {
		return nil, fmt.Errorf("%d channels: mix them to mono with ffmpeg first", format.NumChannels)
	}

	r := beep.Resample(4, format.SampleRate, whisper.SampleRate, newLowPass(s, int(format.SampleRate)))
	var pcm []float32
	buf := make([][2]float64, 4096)
	for {
		n, ok := r.Stream(buf)
		for _, frame := range buf[:n] {
			pcm = append(pcm, float32(frame[0]))
		}
		if !ok {
			break
		}
	}
	if err := r.Err(); err != nil {
		return nil, err
	}
	// beep reads a WAV of unknown size, as ffmpeg writes to a pipe, or of
	// 2 GiB or more, as empty.
	if len(pcm) == 0 {
		return nil, errors.New("no samples")
	}
	return pcm, nil
}

// lowPass mixes a source to mono, and removes what lies above 8 kHz, half
// of whisper.SampleRate. beep.Resample filters nothing, so it would fold
// that into the speech. lowPass is a windowed-sinc filter: it passes up to
// 6.5 kHz, stops from 8.5 kHz, and delays nothing.
type lowPass struct {
	s    beep.Streamer
	h    []float64    // the filter, of 2*half+1 taps
	x    []float64    // the mono samples ahead, after the half before them
	buf  [][2]float64 // read from s
	done bool         // s has ended
}

// newLowPass filters s, sampled at rate. A rate up to whisper.SampleRate
// has nothing to remove, and is only mixed.
func newLowPass(s beep.Streamer, rate int) *lowPass {
	const cutoff, width = 7500, 2000 // Hz
	half := 0
	if rate > whisper.SampleRate {
		half = 11 * rate / (4 * width) // a Blackman window is 5.5 rate/width taps
	}
	fc := float64(cutoff) / float64(rate)
	h := make([]float64, 2*half+1)
	var sum float64
	for i := range h {
		t := float64(i - half)
		h[i] = 1
		if half > 0 {
			h[i] = 0.42 + 0.5*math.Cos(math.Pi*t/float64(half)) + 0.08*math.Cos(2*math.Pi*t/float64(half))
		}
		if t != 0 {
			h[i] *= math.Sin(2*math.Pi*fc*t) / (math.Pi * t)
		} else {
			h[i] *= 2 * fc
		}
		sum += h[i]
	}
	for i := range h {
		h[i] /= sum
	}
	return &lowPass{s: s, h: h, x: make([]float64, half), buf: make([][2]float64, 4096)}
}

func (f *lowPass) Stream(out [][2]float64) (int, bool) {
	half := len(f.h) / 2
	for !f.done && len(f.x) < len(out)+2*half {
		n, ok := f.s.Stream(f.buf)
		// beep gives every source as stereo: a mono one in both channels.
		// Their mean is the mono whisper.go takes.
		for _, frame := range f.buf[:n] {
			f.x = append(f.x, (frame[0]+frame[1])/2)
		}
		if !ok {
			f.done = true
			f.x = append(f.x, make([]float64, half)...)
		}
	}
	n := max(0, min(len(out), len(f.x)-2*half))
	for i := range n {
		var v float64
		w := f.x[i : i+len(f.h)]
		for k, hk := range f.h {
			v += hk * w[k]
		}
		out[i] = [2]float64{v, v}
	}
	f.x = f.x[n:]
	return n, n > 0
}

func (f *lowPass) Err() error { return f.s.Err() }
