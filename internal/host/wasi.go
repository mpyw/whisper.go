package host

import (
	"encoding/binary"
	"fmt"
	"io"
	"runtime"
	"time"

	whisperwasm "github.com/mpyw/whisper.go-wasm"
)

// WASI errno values.
const (
	wasiSuccess    = 0
	wasiBadf       = 8
	wasiInval      = 28
	wasiNosys      = 52
	wasiNotcapable = 76
)

// WASI is the part of WASI preview 1 that whisper.wasm reaches. It has no
// file system, arguments or environment. Its stdout and stderr carry
// whisper.cpp's log.
type WASI struct {
	mem   *Memory
	log   io.Writer
	start time.Time
}

// NewWASI returns WASI over mem, writing the log to log. Nil discards it.
func NewWASI(mem *Memory, log io.Writer) *WASI {
	return &WASI{mem: mem, log: log, start: time.Now()}
}

// WASIExit is the panic of proc_exit. whisper.cpp calls it only through
// exit(), which a library linked as a reactor never reaches on purpose.
type WASIExit int32

func (e WASIExit) Error() string { return fmt.Sprintf("host: proc_exit(%d)", int32(e)) }

// Xclock_time_get answers the realtime and monotonic clocks. The CPU-time
// clocks follow the monotonic one: ggml times its work with them.
func (w *WASI) Xclock_time_get(id int32, _ int64, out int32) int32 {
	var ns int64
	switch id {
	case 0:
		ns = time.Now().UnixNano()
	case 1, 2, 3:
		ns = int64(time.Since(w.start))
	default:
		return wasiInval
	}
	binary.LittleEndian.PutUint64(w.mem.Bytes()[out:], uint64(ns))
	return wasiSuccess
}

// Xenviron_get writes nothing: the environment is empty.
func (w *WASI) Xenviron_get(_, _ int32) int32 { return wasiSuccess }

// Xenviron_sizes_get reports an empty environment.
func (w *WASI) Xenviron_sizes_get(count, size int32) int32 {
	b := w.mem.Bytes()
	binary.LittleEndian.PutUint32(b[count:], 0)
	binary.LittleEndian.PutUint32(b[size:], 0)
	return wasiSuccess
}

// Xfd_close closes nothing. Only the standard descriptors exist.
func (w *WASI) Xfd_close(fd int32) int32 {
	if fd >= 0 && fd <= 2 {
		return wasiSuccess
	}
	return wasiBadf
}

// Xfd_fdstat_get reports the standard descriptors as character devices.
func (w *WASI) Xfd_fdstat_get(fd, out int32) int32 {
	if fd < 0 || fd > 2 {
		return wasiBadf
	}
	b := w.mem.Bytes()[out : out+24]
	clear(b)
	b[0] = 2 // filetype: character_device
	binary.LittleEndian.PutUint64(b[8:], ^uint64(0))
	return wasiSuccess
}

// Xfd_fdstat_set_flags is not supported.
func (w *WASI) Xfd_fdstat_set_flags(_, _ int32) int32 { return wasiNosys }

// Xfd_prestat_get reports no preopened directories.
func (w *WASI) Xfd_prestat_get(_, _ int32) int32 { return wasiBadf }

// Xfd_prestat_dir_name has no directory to name.
func (w *WASI) Xfd_prestat_dir_name(_, _, _ int32) int32 { return wasiBadf }

// Xfd_read has nothing to read: stdin is closed.
func (w *WASI) Xfd_read(_, _, _, _ int32) int32 { return wasiBadf }

// Xfd_seek cannot seek a character device.
func (w *WASI) Xfd_seek(_ int32, _ int64, _, _ int32) int32 { return wasiBadf }

// Xfd_write writes stdout and stderr to the log.
func (w *WASI) Xfd_write(fd, iovs, n, written int32) int32 {
	if fd != 1 && fd != 2 {
		return wasiBadf
	}
	b := w.mem.Bytes()
	var total uint32
	for i := range n {
		p := binary.LittleEndian.Uint32(b[iovs+8*i:])
		l := binary.LittleEndian.Uint32(b[iovs+8*i+4:])
		if w.log != nil {
			// A log the caller cannot write is no reason to fail the run.
			_, _ = w.log.Write(b[p : p+l])
		}
		total += l
	}
	binary.LittleEndian.PutUint32(b[written:], total)
	return wasiSuccess
}

// Xpath_open opens nothing: there is no file system.
func (w *WASI) Xpath_open(_, _, _, _, _ int32, _, _ int64, _, _ int32) int32 {
	return wasiNotcapable
}

// Xproc_exit panics with WASIExit, which the caller of the export recovers.
func (w *WASI) Xproc_exit(code int32) { panic(WASIExit(code)) }

// Xsched_yield yields the goroutine: ggml spins on it while it waits.
func (w *WASI) Xsched_yield() int32 {
	runtime.Gosched()
	return wasiSuccess
}

var _ whisperwasm.Xwasi_snapshot_preview1 = (*WASI)(nil)
