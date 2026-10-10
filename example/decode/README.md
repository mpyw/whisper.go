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

> [!NOTE]
> go-mp3 keeps the silence the MP3 encoder adds, which ffmpeg removes. jfk's samples from an MP3 began 51 ms late, and ran 76 ms longer.
> whisper.cpp's times can move more than that: jfk's first segment ended at 7.44 s from an MP3, and at 7.74 s from the WAV.

## Tasks

| Task | What it does |
| --- | --- |
| `mise run download` | Downloads the model (`ggml-tiny.en-q5_1.bin`, 32 MB) once, and checks its SHA-256 |
| `mise run transcribe -- FILE` | Runs `download`, then transcribes `FILE`, or `testdata/jfk.wav` without one |

`mise.toml` sets `GOEXPERIMENT=simd`, which makes whisper.go 4 to 6 times faster.

## Code

[`main.go`](main.go):

1. `decode` opens the file with beep's decoder for its extension.
2. beep gives every source as stereo, a mono one in both channels. `lowPass` takes the mean of the two as the mono sample.
3. `beep.Resample` converts it to 16 kHz. It filters nothing, so `lowPass` filters around it:
   - From a higher rate, `lowPass` first removes what is above 8 kHz, half of 16 kHz, which `beep.Resample` would fold into the speech. This makes decoding about 4 times slower: 3.8 s for 10 minutes of 48 kHz stereo, on an Apple M2 Pro.
   - From a lower rate, `beep.Resample` adds copies of the speech above half that rate, which `lowPass` removes after.
4. A 16 kHz source passes through both unchanged.
5. `Transcribe` returns the segments, which it prints.

This is its own Go module. Its `go.mod` builds against this checkout of whisper.go, through `replace`.
