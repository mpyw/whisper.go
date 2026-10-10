<div align="center">

<img src=".github/assets/icon.png" width="200" alt="A Go gopher cupping a paw to its ear, listening to sound waves">

# whisper.go

Transcribe speech from Go with [whisper.cpp](https://github.com/ggml-org/whisper.cpp), with no cgo.

[![Test](https://github.com/mpyw/whisper.go/actions/workflows/test.yml/badge.svg)](https://github.com/mpyw/whisper.go/actions/workflows/test.yml)
[![Coverage](https://codecov.io/gh/mpyw/whisper.go/graph/badge.svg)](https://codecov.io/gh/mpyw/whisper.go)
[![Go Reference](https://pkg.go.dev/badge/github.com/mpyw/whisper.go.svg)](https://pkg.go.dev/github.com/mpyw/whisper.go)

</div>

whisper.go runs whisper.cpp itself, so it reads the same models and writes the same text.
whisper.cpp is compiled to WebAssembly, and [wasm2go](https://github.com/ncruces/wasm2go) translates that to Go source.
That code is in [whisper.go-wasm](https://github.com/mpyw/whisper.go-wasm).
There is no cgo and no Wasm runtime, so `go build` is enough, for any OS.

> [!WARNING]
> This is an experiment. The API will change.

## Usage

```sh
go get github.com/mpyw/whisper.go
```

```go
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
```

The models are whisper.cpp's, in its ggml format.
Download them from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp).

> [!IMPORTANT]
> `Transcribe` takes samples, not a file.
> They must be mono, at 16 kHz (`whisper.SampleRate`), as float32 in [-1, 1].
> whisper.go reads no audio formats.
> `ffmpeg -i in.mp3 -ar 16000 -ac 1 -f f32le out.raw` writes such samples.

| Name | What it is |
| --- | --- |
| `New(model, opts)` | Loads a model. `Options.Log` receives whisper.cpp's log. Nil discards it. |
| `(*Model).Transcribe(ctx, pcm, opts)` | Transcribes the samples. Canceling `ctx` stops the run within one of ggml's operations. |
| `TranscribeOptions.Language` | A code such as `en` or `ja`. Empty, or `auto`, detects it with a multilingual model. An English-only model transcribes English whatever it says. |
| `TranscribeOptions.Threads` | Threads to run on. Zero means `GOMAXPROCS`, and 32 is the most. |
| `(*Model).Close()` | Frees the model and its memory |
| `ErrClosed` | Returned after `Close`, or after a trap |
| `ErrTooLong` | Returned for audio the module's memory cannot hold. The model stays usable. |

A `Model` may be used from any goroutine. Its `Transcribe` calls run one at a time.
For calls at the same time, load the model more than once.

## Performance

> [!TIP]
> Build with `GOEXPERIMENT=simd` for speed.
> Then whisper.go runs ggml's matrix products with NEON on arm64, and with AVX2 and FMA on amd64.
> Without it, or on other CPUs, the same code runs without SIMD, 4.1 to 5.9 times slower.

```sh
GOEXPERIMENT=simd go build ./...
```

whisper.cpp 1.9.5 with tiny.en, on `testdata/jfk.wav` (11 s of speech), on an Apple M2 Pro.

| Model | Threads | whisper.go | With `GOEXPERIMENT=simd` | Native whisper.cpp, CPU only |
| --- | --- | --- | --- | --- |
| f16 | 1 | 9.6 s | 1.73 s | 0.96 s |
| f16 | 8 | 1.34 s | 0.29 s | 0.26 s |
| q8_0 | 1 | 8.6 s | 1.75 s | 0.63 s |
| q8_0 | 8 | 1.20 s | 0.29 s | 0.17 s |
| q5_1 | 1 | 9.0 s | 1.51 s | 1.24 s |
| q5_1 | 8 | 1.25 s | 0.26 s | 0.24 s |

> [!NOTE]
> Loading the model took 0.03 to 0.1 s.
> Softmax and the quantization of activations run on the host too. The rest of ggml has no SIMD, since wasm2go has none.
> That is most of the gap to native whisper.cpp.
> `GOEXPERIMENT=simd` is experimental in Go 1.27, and its API may change.

## Limitations

| What | Why |
| --- | --- |
| 64-bit Unix and Windows only | The module's memory is 4 GiB of address space, reserved up front. It must never move while threads read it. Other platforms cannot reserve it, and `New` fails there. |
| A model takes about twice its size in memory | `New` copies the model into the module, which builds its tensors from the copy and keeps that memory. The 4 GiB limit holds models below about 2 GB. |
| Audio takes about 6 bytes a sample | The samples and their mel spectrogram, 7 with large-v3's 128 mel bands. One call takes at most about 9 hours, and less with a large model. Longer audio returns `ErrTooLong`, and the model stays usable. Split it. |
| Memory only grows | The module never gives memory back. A model keeps what its longest call took, until `Close`. |
| A trap closes the model, and leaks it | `GGML_ABORT`, and the errors whisper.cpp threw, trap. The error wraps `ErrClosed`, and so does every call after. `Close` frees nothing then, since the module's threads may still be inside it. Those that spin end, and those that wait stay. |
| A trap on a worker thread ends the program | The thread that waits for it would wait forever. |
| `Threads` above `GOMAXPROCS` | It is capped at `GOMAXPROCS`, and at 32. ggml's threads spin while they wait for each other, so a busy CPU slows every thread. |
| No GPU | whisper.cpp runs on its CPU backend only |

## License

whisper.go is under the [MIT License](LICENSE).
A program built with it contains whisper.cpp and other software, through whisper.go-wasm.
Their licenses are in `whisperwasm.Licenses`, and in [whisper.go-wasm's licenses/](https://github.com/mpyw/whisper.go-wasm/tree/main/licenses).
Whoever distributes such a program must pass them on.

`testdata/jfk.wav` comes from whisper.cpp's samples, under its MIT License.

The mascot is the Go gopher, designed by Renée French, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
