// Command jfk transcribes jfk.wav, a sample of whisper.cpp: 11 s of a
// speech by John F. Kennedy.
//
//	mise run transcribe
//
// The model and the WAV file may be given as arguments too:
//
//	go run . ggml-tiny.en-q5_1.bin jfk.wav
package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/mpyw/whisper.go"
)

func main() {
	modelPath, wavPath := "ggml-tiny.en-q5_1.bin", "jfk.wav"
	if len(os.Args) == 3 {
		modelPath, wavPath = os.Args[1], os.Args[2]
	}

	model, err := os.ReadFile(modelPath)
	if err != nil {
		log.Fatal(err)
	}
	m, err := whisper.New(model, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()

	wav, err := os.ReadFile(wavPath)
	if err != nil {
		log.Fatal(err)
	}
	pcm, err := readWAV(wav)
	if err != nil {
		log.Fatalf("%s: %v", wavPath, err)
	}

	res, err := m.Transcribe(context.Background(), pcm, &whisper.TranscribeOptions{Language: "en"})
	if err != nil {
		log.Fatal(err)
	}
	for _, s := range res.Segments {
		fmt.Printf("[%v -> %v] %s\n", s.Start, s.End, s.Text)
	}
}

// readWAV reads a WAV file of 16-bit samples, mono, at 16 kHz, and returns
// its samples in [-1, 1]. Convert other audio with ffmpeg first:
//
//	ffmpeg -i in.mp3 -ar 16000 -ac 1 -c:a pcm_s16le out.wav
func readWAV(b []byte) ([]float32, error) {
	if len(b) < 12 || !bytes.Equal(b[0:4], []byte("RIFF")) || !bytes.Equal(b[8:12], []byte("WAVE")) {
		return nil, errors.New("not a WAV file")
	}
	var format bool
	for b = b[12:]; len(b) >= 8; {
		id, size := string(b[0:4]), int(binary.LittleEndian.Uint32(b[4:8]))
		b = b[8:]
		if size > len(b) {
			return nil, fmt.Errorf("chunk %q is cut short", id)
		}
		chunk := b[:size]
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, errors.New("fmt chunk is cut short")
			}
			tag, channels := binary.LittleEndian.Uint16(chunk[0:]), binary.LittleEndian.Uint16(chunk[2:])
			rate, bits := binary.LittleEndian.Uint32(chunk[4:]), binary.LittleEndian.Uint16(chunk[14:])
			if tag != 1 || channels != 1 || rate != whisper.SampleRate || bits != 16 {
				return nil, fmt.Errorf("want 16-bit PCM, mono, at %d Hz: got format %d, %d channels, %d Hz, %d bits", whisper.SampleRate, tag, channels, rate, bits)
			}
			format = true
		case "data":
			if !format {
				return nil, errors.New("data before the fmt chunk")
			}
			pcm := make([]float32, size/2)
			for i := range pcm {
				pcm[i] = float32(int16(binary.LittleEndian.Uint16(chunk[2*i:]))) / 32768
			}
			return pcm, nil
		}
		// Chunks are padded to an even size.
		b = b[min(size+size%2, len(b)):]
	}
	return nil, errors.New("no data chunk")
}
