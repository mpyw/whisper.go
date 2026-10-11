# Example: decode and transcribe any audio file

Transcribes a WAV, MP3, FLAC or Ogg Vorbis file, at any sample rate, mono or stereo.
[beep](https://github.com/gopxl/beep) decodes it, and [go-resample](https://github.com/gojargo/go-resample) converts it to 16 kHz, both in pure Go, so there is still no cgo and no ffmpeg.

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
| WAV of unknown size, as ffmpeg writes to a pipe, or of 2 GiB or more | No | beep reads it as empty, which is an error here. Write the WAV to a file, or split it. |
| WAV, 32-bit integer or float PCM | No | Convert with ffmpeg. A float WAV is float32 already: reading it yourself is short. |
| MP3 | Yes | [go-mp3](https://github.com/hajimehoshi/go-mp3) (Apache-2.0). It is archived, but beep still uses it. |
| MP3 at 22.05 kHz and 32 kbit/s or less | No | go-mp3 decodes it wrong, with no error: jfk came out as other words. |
| MP3 at 8, 11.025 or 12 kHz (MPEG 2.5) | No | go-mp3 returns an error. |
| FLAC | Yes | [mewkiz/flac](https://github.com/mewkiz/flac) and mewkiz/pkg (Unlicense), with [icza/bitio](https://github.com/icza/bitio) (Apache-2.0) |
| Ogg Vorbis | Yes | [oggvorbis](https://github.com/jfreymuth/oggvorbis) and vorbis (MIT) |
| More than 2 channels | No | beep reads only 2, so `decode` returns an error. Ogg Vorbis hides the count: beep reads its front left and right, without the center. |
| AAC, Opus, video | No | Use ffmpeg: `ffmpeg -i in.m4a -ar 16000 -ac 1 -f f32le out.raw` |

Every decoder also uses [pkg/errors](https://github.com/pkg/errors) (BSD-2-Clause).
Resampling is [go-resample](https://github.com/gojargo/go-resample) (BSD-2-Clause), which follows [libsamplerate](https://github.com/libsndfile/libsamplerate) (BSD-2-Clause). A program built with it must carry its `LICENSE` and `NOTICE`.

> [!NOTE]
> go-mp3 keeps the silence the MP3 encoder adds, which ffmpeg removes. jfk's samples from an MP3 began 51 ms late, and ran 76 ms longer.
> whisper.cpp's times can move more than that: jfk's first segment ended at 7.44 s from an MP3, and at 7.74 s from the WAV.

## What resampling leaves

go-resample's `SincBestQuality` filters out what 16 kHz cannot hold, above 8 kHz, as it converts.
The worst false tone left, against a test tone, measured through `decode` in 100 Hz steps up to 90% of the lower of the two half rates:

| Source rate | Worst false tone |
| --- | --- |
| 8 or 12 kHz | -92 dB |
| 11.025 or 22.05 kHz | -82 dB |
| 16 kHz | None: it is only mixed |
| 32, 44.1, 48 or 96 kHz | -83 dB or less |

The test WAVs are 16-bit, whose own noise is near -90 dB.
At the edge of the band, from 44.1 kHz:

| Tone | Left |
| --- | --- |
| 7 kHz | All of it |
| 7.5 kHz | -1 dB |
| 7.8 kHz | -6 dB |
| 8.1 to 8.5 kHz | -18 to -54 dB, folded to 7.5 to 7.9 kHz |
| 8.7 kHz or more | -88 dB or less |

jfk transcribed the same from 11.025, 22.05 and 48 kHz WAVs, and from a 44.1 kHz FLAC. From 8 kHz, "Americans" became "America", and "American" when ffmpeg resampled the 8 kHz file instead: 8 kHz holds nothing above 4 kHz.

## Tasks

| Task | What it does |
| --- | --- |
| `mise run download` | Downloads the model (`ggml-tiny.en-q5_1.bin`, 32 MB) once, and checks its SHA-256 |
| `mise run transcribe -- FILE` | Runs `download`, then transcribes `FILE`, or `testdata/jfk.wav` without one |

`mise.toml` sets `GOEXPERIMENT=simd`, which makes whisper.go 4 to 6 times faster.

## Code

[`main.go`](main.go):

1. `decode` opens the file with beep's decoder for its extension.
2. `to16k` reads it in blocks. beep gives every source as stereo, a mono one in both channels, and `to16k` takes the mean of the two.
3. Unless the source is 16 kHz already, `to16k` passes each block to go-resample, and keeps what comes out. It never holds the samples at the old rate: 10 minutes of 48 kHz stereo took 3.6 s and 114 MB, on an Apple M2 Pro.
4. `Transcribe` returns the segments, which it prints.

This is its own Go module. Its `go.mod` builds against this checkout of whisper.go, through `replace`.
