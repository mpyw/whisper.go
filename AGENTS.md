# whisper.go

Speech-to-text for Go, running whisper.cpp from whisper.go-wasm with no cgo. See README.md for the API.

## Rules

- whisper.cpp lives in github.com/mpyw/whisper.go-wasm, as Go source that wasm2go translated. Its ABI.md is the contract with `whisper.go` and `internal/host`.
- `hostABIVersion` in `whisper.go` must equal `whisperwasm.ABIVersion`.
- whisper.go-wasm's branches hold no generated code. Test against a local checkout only after building it (`whisper.go-wasm-upgrade` skill).
- Check every error, or drop it with `_ =` and a comment saying why. errcheck runs with none of its default exclusions, in tests too.
- The library does not log. whisper.cpp's log goes to `Options.Log`, and everything else is a returned error.
- The module's memory never moves. `internal/host` reserves its whole address space, and grows inside it.
- Every kernel in `internal/host` has a generic version in `kernel_generic.go`. The vector versions, `kernel_arm64.go` and `kernel_amd64.go`, build only with `GOEXPERIMENT=simd`. `kernel_nosimd.go` maps the names otherwise. `TestKernelHalf`, `TestKernelQuantized` and `TestKernelTile` compare the two.
- In the vector code, keep accumulators in named variables, not arrays: the compiler sends an array of vectors through the stack. Load through array pointers, sliced once for several loads, so that each load has no bounds check.
- Test the vector code on both architectures. An amd64 test binary runs under Docker's `--platform linux/amd64` on an arm64 Mac.
- Release with the `Tag and Release` workflow: `gh workflow run tag_and_release.yml -f version=vX.Y.Z`. It tags main's head only once its tests passed, and never reuses a tag: Go caches a version forever.

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
GOEXPERIMENT=simd go vet ./... && GOEXPERIMENT=simd go test ./...
for d in example/*/; do (cd "$d" && go vet ./... && go test ./...); done
mise exec -- golangci-lint run ./...
GOEXPERIMENT=simd mise exec -- golangci-lint run ./...
mise exec -- declscope shrink ./...
mise exec -- declscope ./...
```

declscope runs with `.declscope.yaml` (`qualify: ondemand`, `exported: true`).
The tests download a 32 MB model once, into the user cache directory. `WHISPER_TEST_MODEL` names another model, which they read as it is.

## Rejected designs

| Idea | Why not |
| --- | --- |
| Transpile whisper.cpp to Go with ccgo | ccgo is C only. whisper.cpp and most of ggml are C++. |
| Rewrite Whisper's inference in Go | whisper.cpp's quantized models, decoding and timestamps would all have to be rewritten. Translating it keeps its output. |
| A growable `[]byte` for the module's memory | Growing could move it while other threads read it. The address space is reserved up front instead. |
| A slice as long as the committed pages | `Grow` changed the slice's length while threads read the slice: `go test -race` failed on every run. The slice spans the reservation from the start, and the committed pages are counted apart. A touch beyond them faults, and `debug.SetPanicOnFault` makes that a panic. |
| Addresses as `int32` in the host | The module passes them as i32, so one at 2 GiB or above was negative and panicked. `host.MemoryAddr` reads them as unsigned. |
| Checking `ctx.Err()` only when `gw_full` fails | A cancel before a pass of the encoder makes whisper.cpp stop with success, and `Transcribe` returned part of the result with no error. `ctx.Err()` decides first. |
| Freeing a trapped model's memory in `Close` | The trap may leave threads in the module, which would fault on freed pages, and `Close` would wait for them. A goroutine frees it once they end instead: `Stop` ends those that spin, and one that waits for good keeps it. |
| Recovering a trap on a worker thread | The thread that started the graph waits for it at a barrier, so recovering would hang the run. The program ends instead. |
| `SetContext(nil)` to stop watching a context | staticcheck rejects a nil Context. `Transcribe` restores `context.Background()` instead. |
| The portable `simd` package | It has no widening and no sum across lanes, which float16 and int8 need. `simd/archsimd` has both, one file per architecture. |
| `unsafe` to view the module's memory as `[]float32` | `archsimd.LoadUint8x16` and the `Reshape` methods read the bytes as vectors with no `unsafe`. |
| A dot product per column of a tile | Each row of x was loaded once per column. 4 columns at a time load it once for 4, and keep 8 sums in flight. With q5_1 and 8 threads, 0.76 s became 0.64 s. |
| A no-op `sched_yield` | It is 16% faster on an idle machine, but 5 times slower with 2 busy goroutines: Go preempts a spinning goroutine only every 10 ms. The time in `pthread_cond_signal` was `Gosched` waking another thread, so ggml yields only once every 4096 spins (whisper.go-wasm's `patches/0001`). |
| Reading F16's float32 columns from the module's memory in place | `kernelFloatRow` copies them to a scratch row first, for 2.7% of the time. Without the copy, q5_1 took 1.2% longer with 1 thread and 18% with 8. |
| float16 to float32 by arithmetic, in place of the `kernelHalf` table | With 1 thread, q5_1 took 3.1% longer, and q8_0 2.1%. |
| Returning vectors from `kernelDotF32x2x4` and storing the two rows in pairs | With 1 thread it was 1% slower, and with 8 threads up to 3% faster: no clear gain. |
