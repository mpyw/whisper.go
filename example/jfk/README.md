# Example: transcribe a WAV file

Transcribes [`testdata/jfk.wav`](../../testdata/jfk.wav), whisper.cpp's sample: 11 s of a speech by John F. Kennedy.

## Run

Needs [mise](https://mise.jdx.dev), Go, `bash` and `curl`.

```sh
mise run transcribe
```

```
[0s -> 7.74s]  And so my fellow Americans ask not what your country can do for you
[7.74s -> 10.58s]  ask what you can do for your country.
```

## Tasks

| Task | What it does |
| --- | --- |
| `mise run download` | Downloads the model (`ggml-tiny.en-q5_1.bin`, 32 MB) once, and checks its SHA-256 |
| `mise run transcribe` | Runs `download`, then `go run .` |

`mise.toml` sets `GOEXPERIMENT=simd`, which makes whisper.go 4 to 6 times faster.

## Your own audio

```sh
go run . path/to/model.bin path/to/audio.wav
```

The WAV file must be 16-bit PCM, mono, at 16 kHz. Convert other audio with ffmpeg:

```sh
ffmpeg -i in.mp3 -ar 16000 -ac 1 -c:a pcm_s16le out.wav
```

## Code

[`main.go`](main.go):

1. `whisper.New` loads the model.
2. `readWAV` reads the WAV file into float32 samples in [-1, 1].
3. `Transcribe` returns the segments, which it prints.

This is its own Go module. Its `go.mod` builds against this checkout of whisper.go, through `replace`.
