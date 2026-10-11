// Command decode transcribes an audio file: WAV, MP3, FLAC or Ogg Vorbis, at
// any sample rate, mono or stereo. beep decodes it, and go-resample converts
// it to 16 kHz, both in pure Go.
//
//	mise run transcribe -- speech.mp3
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/gojargo/go-resample"
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

	pcm, err := to16k(s, int(format.SampleRate))
	if err != nil {
		return nil, err
	}
	// beep reads a WAV of unknown size, as ffmpeg writes to a pipe, or of
	// 2 GiB or more, as empty.
	if len(pcm) == 0 {
		return nil, errors.New("no samples")
	}
	return pcm, nil
}

// to16k reads s, sampled at rate, as mono at whisper.SampleRate. It
// resamples as it reads, so it never holds the samples at the old rate.
// SincBestQuality filters out what 16 kHz cannot hold, above 8 kHz. README.md
// lists what is left.
func to16k(s beep.Streamer, rate int) ([]float32, error) {
	r, err := resample.New(resample.SincBestQuality, 1)
	if err != nil {
		return nil, err
	}
	// ratio is how many samples come out for each one in: 16000/44100, about
	// 0.36, from 44.1 kHz, and 2 from 8 kHz.
	ratio := float64(whisper.SampleRate) / float64(rate)
	// buf takes one block from beep: 4096 frames, a left and a right sample
	// each. in holds the same block as mono, and out what go-resample makes
	// of it. All three are reused for every block.
	buf := make([][2]float64, 4096)
	in := make([]float32, 0, len(buf))
	out := make([]float32, 4096)
	var pcm []float32 // every sample at 16 kHz, the result
	for {
		// n is how many frames beep put in buf. ok is false once the file
		// has ended.
		n, ok := s.Stream(buf)
		// beep gives every source as stereo: a mono one in both channels.
		// The mean of left and right is the mono whisper.go takes.
		in = in[:0]
		for _, frame := range buf[:n] {
			left, right := frame[0], frame[1]
			in = append(in, float32((left+right)/2))
		}
		if rate == whisper.SampleRate {
			pcm = append(pcm, in...) // already 16 kHz: nothing to convert
		} else {
			d := resample.Data{
				In:           in,
				InputFrames:  len(in),
				Out:          out,
				OutputFrames: len(out), // room in out
				Ratio:        ratio,
				// The last block: give out every sample left, rather than
				// wait for samples after them.
				EndOfInput: !ok,
			}
			// Process keeps all of in, and fills out as far as it can.
			// OutputFramesGen says how far. When out is full, there may be
			// more: ask again, with no new input.
			for {
				if err := r.Process(&d); err != nil {
					return nil, err
				}
				pcm = append(pcm, out[:d.OutputFramesGen]...)
				if d.OutputFramesGen < len(out) {
					break // all of this block is out
				}
				d.InputFrames = 0 // in is kept already: do not pass it twice
			}
		}
		if !ok {
			return pcm, s.Err()
		}
	}
}
