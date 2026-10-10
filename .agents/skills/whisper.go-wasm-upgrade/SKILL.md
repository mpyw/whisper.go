---
name: whisper.go-wasm-upgrade
description: Move whisper.go to a new whisper.go-wasm release, test whisper.go against a local whisper.go-wasm checkout, or implement the host side of an ABI change. Use when go.mod's whisper.go-wasm version changes, when hostABIVersion or an import of the module changes, or before a whisper.go-wasm release.
---

# Upgrade whisper.go-wasm

whisper.go-wasm's branches hold no generated code. Only its release tags carry `module.go`.

## Before a release: test against a local checkout

1. Build the checkout: `scripts/fetch.sh && scripts/build.sh` in `../whisper.go-wasm`.
2. Point Go at it with a `go.work` outside the repository, so that it is never committed:

   ```sh
   printf 'go 1.27.1\n\nuse (\n\t%s\n\t%s\n)\n' "$PWD" "$(cd ../whisper.go-wasm && pwd)" > /tmp/whisper.work
   GOWORK=/tmp/whisper.work go test ./...
   ```

   The `go` line must not be older than either `go.mod`. Give `use` clean absolute paths. With `..` in one, `declscope shrink` fails to load the module.

## After a release

```sh
go get github.com/mpyw/whisper.go-wasm@vX.Y.Z
go mod tidy
```

Then run everything in AGENTS.md's "Before you finish".

| If the release changed | Also |
| --- | --- |
| `ABIVersion` | Set `hostABIVersion` in `whisper.go` to it, and implement the host side (below) |
| `WhisperVersion` | README's Performance section names the version it measured |

## The host side of an ABI change

whisper.go-wasm's ABI.md is the contract. `internal/host` implements every import:

| Import module | Implemented by |
| --- | --- |
| `env`: the memory and `gw_aborted` | `memory.go`, `env.go` |
| `wasi_snapshot_preview1` | `wasi.go` |
| `wasi`: `thread-spawn` | `thread.go` |

The exports are called from `whisper.go`, inside `(*Model).call`, which turns a trap into an error.

## Rejected designs

| Idea | Why not |
| --- | --- |
| Committing the generated Go to this repository | It is 8.5 MB and 276,000 lines, and nearly every line changes with each rebuild. whisper.go-wasm's release tags carry it, on orphan commits. |
