//declscope:namespace whisper

//go:build race

package whisper_test

// testSlowdown is how much slower the code runs: the race detector makes
// it about ten times slower, and a time limit must allow for that.
const testSlowdown = 20
