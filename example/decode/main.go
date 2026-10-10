// Command decode transcribes an audio file: WAV, MP3, FLAC or Ogg Vorbis, at
// any sample rate, mono or stereo. beep decodes and resamples it, in pure
// Go, to the samples whisper.go takes.
//
//	mise run transcribe -- speech.mp3
package main

import (
	"context"
	"fmt"
	"log"
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

	// beep gives every source as stereo: a mono one in both channels.
	// Their mean is the mono whisper.go takes.
	r := beep.Resample(4, format.SampleRate, whisper.SampleRate, s)
	var pcm []float32
	buf := make([][2]float64, 4096)
	for {
		n, ok := r.Stream(buf)
		for _, frame := range buf[:n] {
			pcm = append(pcm, float32((frame[0]+frame[1])/2))
		}
		if !ok {
			break
		}
	}
	return pcm, r.Err()
}
