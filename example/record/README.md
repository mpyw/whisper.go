# Example: record and transcribe

Records from the microphone until you press Enter, then transcribes what it heard, in any of whisper's 99 languages.

## Run

Needs [mise](https://mise.jdx.dev), Go, `bash`, `curl` and [ffmpeg](https://ffmpeg.org).

```sh
mise run record
```

For example:

```
Recording. Press Enter to stop.

Recorded 4.2s. Transcribing...
Language: en
[0s -> 3.2s]  Hello, this is a test of whisper.go.
```

Pass flags after `--`:

```sh
mise run record -- -lang ja
```

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `-lang` | `auto` | The language, such as `en` or `ja`. `auto` detects it. |
| `-model` | `ggml-base-q5_1.bin` | The whisper.cpp model |
| `-format` | per OS, below | ffmpeg's input format (`-f`) |
| `-input` | per OS, below | ffmpeg's input (`-i`): the microphone |

| OS | `-format` | `-input` |
| --- | --- | --- |
| macOS | `avfoundation` | `:0`, the first microphone. `ffmpeg -f avfoundation -list_devices true -i ""` lists them. |
| Linux | `pulse` | `default`. PipeWire serves it too. With ALSA only, use `-format alsa -input default`. |
| Windows | `dshow` | None: give it, such as `-input audio="Microphone (Realtek Audio)"`. `ffmpeg -list_devices true -f dshow -i dummy` lists them. |

> [!NOTE]
> On macOS, the terminal needs permission to use the microphone: System Settings → Privacy & Security → Microphone.
> Without it, nothing is recorded.

## Tasks

| Task | What it does |
| --- | --- |
| `mise run download` | Downloads the model (`ggml-base-q5_1.bin`, 60 MB, multilingual) once, and checks its SHA-256 |
| `mise run record` | Runs `download`, then `go run .` with your flags |

`mise.toml` sets `GOEXPERIMENT=simd`, which makes whisper.go 4 to 6 times faster.

## Code

[`main.go`](main.go):

1. `whisper.New` loads the model.
2. `record` runs ffmpeg, which writes the samples to stdout as whisper.go wants them: 16 kHz, mono, float32. Enter kills ffmpeg, and the samples already written are kept.
3. `Transcribe` returns the language and the segments, which it prints.

ffmpeg does the recording, so the program needs no audio library and no cgo.

This is its own Go module. Its `go.mod` builds against this checkout of whisper.go, through `replace`.
