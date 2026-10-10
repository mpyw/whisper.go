# whisper.go

Speech-to-text for Go, running whisper.cpp from whisper.go-wasm with no cgo. See README.md for the API.

## Rules

- whisper.cpp lives in github.com/mpyw/whisper.go-wasm, as Go source that wasm2go translated. Its ABI.md is the contract with `whisper.go` and `internal/host`.
- `hostABIVersion` in `whisper.go` must equal `whisperwasm.ABIVersion`.
- whisper.go-wasm's branches hold no generated code. Test against a local checkout only after building it (`whisper.go-wasm-upgrade` skill).
- Check every error, or drop it with `_ =` and a comment saying why. errcheck runs with none of its default exclusions, in tests too.
- The library does not log. whisper.cpp's log goes to `Options.Log`, and everything else is a returned error.
- The module's memory never moves. `internal/host` reserves its whole address space, and grows inside it.

## Skills

Read the matching one in `.agents/skills/` before you start. Each lists the designs already rejected in its area, and why.

| Skill | When |
| --- | --- |
| `whisper.go-wasm-upgrade` | A new whisper.go-wasm release, a local checkout, or the host side of an ABI change |
| `declscope-authoring` | Adding, naming or moving any declaration |
| `declscope-adoption` | Changing declscope's configuration |

## Before you finish

```sh
go vet ./... && go test ./...
mise exec -- golangci-lint run ./...
mise exec -- declscope shrink ./...
mise exec -- declscope ./...
```

declscope runs with `.declscope.yaml` (`qualify: ondemand`, `exported: true`).
The tests download a 32 MB model once, into the user cache directory. `WHISPER_TEST_MODEL` names another copy.

## Rejected designs

| Idea | Why not |
| --- | --- |
| Transpile whisper.cpp to Go with ccgo | ccgo is C only. whisper.cpp and most of ggml are C++. |
| Rewrite Whisper's inference in Go | whisper.cpp's quantized models, decoding and timestamps would all have to be rewritten. Translating it keeps its output. |
| A growable `[]byte` for the module's memory | Growing could move it while other threads read it. The address space is reserved up front instead. |
| Recovering a trap on a worker thread | The thread that started the graph waits for it at a barrier, so recovering would hang the run. The program ends instead. |
| `SetContext(nil)` to stop watching a context | staticcheck rejects a nil Context. `Transcribe` restores `context.Background()` instead. |
