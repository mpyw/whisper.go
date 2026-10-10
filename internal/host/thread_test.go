package host

import "testing"

// TestThreadID checks that thread IDs stay in wasi-threads' range when the
// counter wraps, as it does after 2^29 threads.
func TestThreadID(t *testing.T) {
	var th Threads
	th.next.Store(threadMaxID - 2)
	for _, want := range []int32{threadMaxID - 1, threadMaxID, 1, 2} {
		if got := th.nextID(); got != want {
			t.Errorf("got %d, want %d", got, want)
		}
	}
	th.next.Store(1<<32 - 1)
	if got := th.nextID(); got < 1 || got > threadMaxID {
		t.Errorf("after the uint32 wraps: got %d", got)
	}
}
