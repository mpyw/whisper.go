package host

import (
	"bytes"
	"encoding/binary"
	"errors"
	"testing"
)

// wasiMemory is a new Memory, closed when t ends.
func wasiMemory(t *testing.T) *Memory {
	t.Helper()
	mem, err := NewMemory()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := mem.Close(); err != nil {
			t.Error(err)
		}
	})
	return mem
}

// TestWASIStubs checks the errno of each call whisper.cpp never makes. They
// are linked for wasi-libc, which answers its callers by them.
func TestWASIStubs(t *testing.T) {
	w := NewWASI(wasiMemory(t), nil)
	for _, tt := range []struct {
		name      string
		got, want int32
	}{
		{"environ_get", w.Xenviron_get(0x1000, 0x2000), wasiSuccess},
		{"fd_close(2)", w.Xfd_close(2), wasiSuccess},
		{"fd_close(3)", w.Xfd_close(3), wasiBadf},
		{"fd_close(-1)", w.Xfd_close(-1), wasiBadf},
		{"fd_fdstat_get(3)", w.Xfd_fdstat_get(3, 0x1000), wasiBadf},
		{"fd_fdstat_get(-1)", w.Xfd_fdstat_get(-1, 0x1000), wasiBadf},
		{"fd_fdstat_set_flags", w.Xfd_fdstat_set_flags(1, 0), wasiNosys},
		{"fd_prestat_get", w.Xfd_prestat_get(3, 0x1000), wasiBadf},
		{"fd_prestat_dir_name", w.Xfd_prestat_dir_name(3, 0x1000, 16), wasiBadf},
		{"fd_read", w.Xfd_read(0, 0x1000, 1, 0x2000), wasiBadf},
		{"fd_seek", w.Xfd_seek(1, 0, 0, 0x1000), wasiBadf},
		{"fd_write(3)", w.Xfd_write(3, 0x1000, 0, 0x2000), wasiBadf},
		{"path_open", w.Xpath_open(3, 0, 0x1000, 4, 0, 0, 0, 0, 0x2000), wasiNotcapable},
		{"clock_time_get(4)", w.Xclock_time_get(4, 0, 0x1000), wasiInval},
	} {
		if tt.got != tt.want {
			t.Errorf("%s = %d, want %d", tt.name, tt.got, tt.want)
		}
	}
}

// TestWASIFdstat checks that the standard descriptors are character devices
// with every right.
func TestWASIFdstat(t *testing.T) {
	mem := wasiMemory(t)
	w := NewWASI(mem, nil)
	b := mem.Bytes()[0x1000:][:24]
	for i := range b {
		b[i] = 0xff
	}
	if errno := w.Xfd_fdstat_get(1, 0x1000); errno != wasiSuccess {
		t.Fatalf("errno = %d", errno)
	}
	if b[0] != 2 || binary.LittleEndian.Uint16(b[2:]) != 0 || binary.LittleEndian.Uint64(b[8:]) != ^uint64(0) || binary.LittleEndian.Uint64(b[16:]) != 0 {
		t.Errorf("fdstat = %x", b)
	}
}

// TestWASIClock checks that each clock writes its time, and that the
// monotonic ones do not go back.
func TestWASIClock(t *testing.T) {
	mem := wasiMemory(t)
	w := NewWASI(mem, nil)
	b := mem.Bytes()
	var last uint64
	for _, id := range []int32{0, 1, 2, 3, 1} {
		if errno := w.Xclock_time_get(id, 0, 0x1000); errno != wasiSuccess {
			t.Fatalf("clock %d: errno = %d", id, errno)
		}
		ns := binary.LittleEndian.Uint64(b[0x1000:])
		switch {
		case id == 0 && ns < 1<<60:
			// 2^60 ns is 2006: the realtime clock is in Unix time.
			t.Errorf("realtime = %d", ns)
		case id != 0 && ns < last:
			t.Errorf("clock %d went back: %d after %d", id, ns, last)
		}
		if id != 0 {
			last = ns
		}
	}
}

// TestWASIWrite checks that stdout and stderr go to the log, iovec by
// iovec, and that a nil log still counts what was written.
func TestWASIWrite(t *testing.T) {
	mem := wasiMemory(t)
	b := mem.Bytes()
	copy(b[0x3000:], "hello, world")
	binary.LittleEndian.PutUint32(b[0x1000:], 0x3000)
	binary.LittleEndian.PutUint32(b[0x1004:], 5)
	binary.LittleEndian.PutUint32(b[0x1008:], 0x3005)
	binary.LittleEndian.PutUint32(b[0x100c:], 7)
	var log bytes.Buffer
	for _, w := range []*WASI{NewWASI(mem, &log), NewWASI(mem, nil)} {
		for _, fd := range []int32{1, 2} {
			if errno := w.Xfd_write(fd, 0x1000, 2, 0x2000); errno != wasiSuccess {
				t.Fatalf("fd %d: errno = %d", fd, errno)
			}
			if n := binary.LittleEndian.Uint32(b[0x2000:]); n != 12 {
				t.Errorf("fd %d: wrote %d, want 12", fd, n)
			}
		}
	}
	if got := log.String(); got != "hello, worldhello, world" {
		t.Errorf("log = %q", got)
	}
}

// TestWASIExit checks that proc_exit panics with its code.
func TestWASIExit(t *testing.T) {
	w := NewWASI(wasiMemory(t), nil)
	defer func() {
		err, _ := recover().(error)
		if e := WASIExit(0); !errors.As(err, &e) || e != 3 || err.Error() != "host: proc_exit(3)" {
			t.Errorf("recovered %v", err)
		}
	}()
	w.Xproc_exit(3)
}

// TestWASIEnviron checks that the environment is empty.
func TestWASIEnviron(t *testing.T) {
	mem := wasiMemory(t)
	b := mem.Bytes()
	for i := range 8 {
		b[0x1000+i] = 0xff
	}
	if errno := NewWASI(mem, nil).Xenviron_sizes_get(0x1000, 0x1004); errno != wasiSuccess {
		t.Fatalf("errno = %d", errno)
	}
	if n, size := binary.LittleEndian.Uint32(b[0x1000:]), binary.LittleEndian.Uint32(b[0x1004:]); n != 0 || size != 0 {
		t.Errorf("%d variables in %d bytes", n, size)
	}
}

// TestWASIStop checks that sched_yield returns, and ends its goroutine after
// Stop.
func TestWASIStop(t *testing.T) {
	w := NewWASI(wasiMemory(t), nil)
	if errno := w.Xsched_yield(); errno != wasiSuccess {
		t.Fatalf("errno = %d", errno)
	}
	w.Stop()
	returned := make(chan bool, 1)
	go func() {
		defer func() { returned <- false }()
		w.Xsched_yield()
		returned <- true
	}()
	if <-returned {
		t.Error("sched_yield returned after Stop")
	}
}
