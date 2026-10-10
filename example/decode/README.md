# Example: decode and transcribe any audio file

Transcribes a WAV, MP3, FLAC or Ogg Vorbis file, at any sample rate, mono or stereo.
[beep](https://github.com/gopxl/beep) decodes and resamples it in pure Go, so there is still no cgo and no ffmpeg.

## Run

Needs [mise](https://mise.jdx.dev), Go, `bash` and `curl`.

```sh
mise run transcribe                  # testdata/jfk.wav
mise run transcribe -- speech.mp3    # your own file
```

## Formats

| Format | Supported | Decoder, and its license |
| --- | --- | --- |
| WAV, integer PCM of 8, 16 or 24 bits | Yes | beep's own `wav` (MIT) |
| WAV, 32-bit integer or float PCM | No | Convert with ffmpeg. A float WAV is float32 already: reading it yourself is short. |
| MP3 | Yes | [go-mp3](https://github.com/hajimehoshi/go-mp3) (Apache-2.0). It is archived, but beep still uses it. |
| FLAC | Yes | [mewkiz/flac](https://github.com/mewkiz/flac) (Unlicense) |
| Ogg Vorbis | Yes | [oggvorbis](https://github.com/jfreymuth/oggvorbis) (MIT) |
| AAC, Opus, video | No | Use ffmpeg: `ffmpeg -i in.m4a -ar 16000 -ac 1 -f f32le out.raw` |

> [!NOTE]
> MP3 shifts the times a little: its encoder adds silence at the start.
> jfk's first segment ended at 7.44 s from an MP3, and at 7.74 s from the WAV.

## Tasks

| Task | What it does |
| --- | --- |
| `mise run download` | Downloads the model (`ggml-tiny.en-q5_1.bin`, 32 MB) once, and checks its SHA-256 |
| `mise run transcribe -- FILE` | Runs `download`, then transcribes `FILE`, or `testdata/jfk.wav` without one |

`mise.toml` sets `GOEXPERIMENT=simd`, which makes whisper.go 4 to 6 times faster.

## Code

[`main.go`](main.go):

1. `decode` opens the file with beep's decoder for its extension.
2. `beep.Resample` converts it to 16 kHz.
3. beep gives every source as stereo, a mono one in both channels. The mean of the two is the mono sample.
4. `Transcribe` returns the segments, which it prints.

This is its own Go module. Its `go.mod` builds against this checkout of whisper.go, through `replace`.
