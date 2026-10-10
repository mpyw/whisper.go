// Command record records from the microphone until Enter, then transcribes
// what it heard.
//
//	mise run record
//
// ffmpeg does the recording, so whisper.go needs no audio library and no
// cgo. It writes the samples to stdout as whisper.go wants them: 16 kHz,
// mono, float32.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"runtime"
	"time"

	"github.com/mpyw/whisper.go"
)

func main() {
	modelPath := flag.String("model", "ggml-base-q5_1.bin", "the whisper.cpp model")
	lang := flag.String("lang", "auto", `the language, such as "en" or "ja", or "auto"`)
	format, input := defaultInput()
	flag.StringVar(&format, "format", format, "ffmpeg's input format (-f)")
	flag.StringVar(&input, "input", input, "ffmpeg's input (-i), such as a microphone")
	flag.Parse()
	if input == "" {
		log.Fatal(`give the microphone with -input, such as -input audio="Microphone (Realtek Audio)"; ffmpeg -list_devices true -f dshow -i dummy lists them`)
	}

	model, err := os.ReadFile(*modelPath)
	if err != nil {
		log.Fatal(err)
	}
	m, err := whisper.New(model, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()

	pcm, err := record(format, input)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "Recorded %v. Transcribing...\n", (time.Duration(len(pcm))*time.Second/whisper.SampleRate).Round(10*time.Millisecond))

	res, err := m.Transcribe(context.Background(), pcm, &whisper.TranscribeOptions{Language: *lang})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "Language: %s\n", res.Language)
	for _, s := range res.Segments {
		fmt.Printf("[%v -> %v] %s\n", s.Start, s.End, s.Text)
	}
}

// defaultInput is ffmpeg's default microphone on this OS. Windows has no
// default: its devices go by name.
func defaultInput() (format, input string) {
	switch runtime.GOOS {
	case "darwin":
		return "avfoundation", ":0"
	case "windows":
		return "dshow", ""
	default:
		return "pulse", "default" // PipeWire serves PulseAudio's clients too
	}
}

// record runs ffmpeg until Enter, and returns the samples it wrote.
func record(format, input string) ([]float32, error) {
	cmd := exec.Command("ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
		"-f", format, "-i", input, "-ar", "16000", "-ac", "1", "-f", "f32le", "-")
	cmd.Stderr = os.Stderr
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start ffmpeg: %w", err)
	}

	var buf bytes.Buffer
	done := make(chan error, 1)
	go func() {
		_, err := io.Copy(&buf, out)
		done <- err
	}()

	fmt.Fprintln(os.Stderr, "Recording. Press Enter to stop.")
	enter := make(chan struct{})
	go func() {
		_, _ = bufio.NewReader(os.Stdin).ReadString('\n') // any line, or EOF, stops it
		close(enter)
	}()

	select {
	case <-enter:
		// Kill works on every OS. The samples already written are kept.
		_ = cmd.Process.Kill() // it may have just exited by itself
		<-done
		_ = cmd.Wait() // killed: its error says only that
	case err := <-done:
		// ffmpeg stopped first: the input ended, or it failed.
		if err != nil {
			return nil, err
		}
		if err := cmd.Wait(); err != nil {
			return nil, fmt.Errorf("ffmpeg: %w", err)
		}
	}

	b := buf.Bytes()
	if len(b) < 4 {
		return nil, errors.New("ffmpeg recorded nothing: check -format and -input, and that this terminal may use the microphone")
	}
	pcm := make([]float32, len(b)/4)
	for i := range pcm {
		pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return pcm, nil
}
