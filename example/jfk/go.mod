module github.com/mpyw/whisper.go/example/jfk

go 1.27.0

require github.com/mpyw/whisper.go v0.0.0

require (
	github.com/mpyw/whisper.go-wasm v0.1.1 // indirect
	golang.org/x/sys v0.49.0 // indirect
)

// The example builds against this checkout of whisper.go.
replace github.com/mpyw/whisper.go => ../..
