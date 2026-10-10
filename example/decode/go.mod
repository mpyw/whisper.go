module github.com/mpyw/whisper.go/example/decode

go 1.27.0

require (
	github.com/gopxl/beep/v2 v2.1.1
	github.com/mpyw/whisper.go v0.0.0
)

require (
	github.com/hajimehoshi/go-mp3 v0.3.4 // indirect
	github.com/icza/bitio v1.1.0 // indirect
	github.com/jfreymuth/oggvorbis v1.0.5 // indirect
	github.com/jfreymuth/vorbis v1.0.2 // indirect
	github.com/mewkiz/flac v1.0.12 // indirect
	github.com/mewkiz/pkg v0.0.0-20230226050401-4010bf0fec14 // indirect
	github.com/mpyw/whisper.go-wasm v0.1.1 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	golang.org/x/sys v0.49.0 // indirect
)

// The example builds against this checkout of whisper.go.
replace github.com/mpyw/whisper.go => ../..
