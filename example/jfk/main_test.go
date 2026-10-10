package main

import (
	"encoding/binary"
	"slices"
	"testing"
)

// wavFile is a WAV file of chunks, with RIFF's size filled in.
func wavFile(chunks ...[]byte) []byte {
	b := []byte("RIFF\x00\x00\x00\x00WAVE")
	for _, c := range chunks {
		b = append(b, c...)
	}
	binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
	return b
}

// wavChunk is a chunk of id. size is what its header says, and body what
// follows, padded to an even size.
func wavChunk(id string, size uint32, body []byte) []byte {
	b := binary.LittleEndian.AppendUint32([]byte(id), size)
	b = append(b, body...)
	if len(body)%2 == 1 {
		b = append(b, 0)
	}
	return b
}

// wavFormat is a fmt chunk of 16-bit PCM, as WAVE_FORMAT_EXTENSIBLE if
// extensible.
func wavFormat(channels uint16, rate uint32, extensible bool) []byte {
	b := make([]byte, 16, 40)
	binary.LittleEndian.PutUint16(b[0:], 1)
	binary.LittleEndian.PutUint16(b[2:], channels)
	binary.LittleEndian.PutUint32(b[4:], rate)
	binary.LittleEndian.PutUint32(b[8:], rate*2*uint32(channels))
	binary.LittleEndian.PutUint16(b[12:], 2*channels)
	binary.LittleEndian.PutUint16(b[14:], 16)
	if extensible {
		binary.LittleEndian.PutUint16(b[0:], 0xFFFE)
		b = binary.LittleEndian.AppendUint16(b, 22) // the extension's size
		b = binary.LittleEndian.AppendUint16(b, 16) // valid bits
		b = binary.LittleEndian.AppendUint32(b, 4)  // the channel mask
		// The subformat: KSDATAFORMAT_SUBTYPE_PCM.
		b = append(b, 1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71)
	}
	return wavChunk("fmt ", uint32(len(b)), b)
}

func TestReadWAV(t *testing.T) {
	samples := []byte{0x00, 0x00, 0x00, 0x40, 0x00, 0x80, 0xff, 0x7f} // 0, 0.5, -1, 32767/32768
	want := []float32{0, 0.5, -1, 32767.0 / 32768}
	format := wavFormat(1, 16000, false)
	data := wavChunk("data", uint32(len(samples)), samples)
	for _, tt := range []struct {
		name string
		file []byte
		want []float32 // nil for an error
	}{
		{"plain", wavFile(format, data), want},
		{"extensible", wavFile(wavFormat(1, 16000, true), data), want},
		{"a chunk of odd size first", wavFile(format, wavChunk("LIST", 3, []byte("abc")), data), want},
		{"the data's size unknown, as ffmpeg writes to a pipe", wavFile(format, wavChunk("data", 0xFFFFFFFF, samples)), want},
		{"cut short in the data", wavFile(format, wavChunk("data", 100, samples)), want},
		{"stereo", wavFile(wavFormat(2, 16000, false), data), nil},
		{"44.1 kHz", wavFile(wavFormat(1, 44100, false), data), nil},
		{"no fmt chunk", wavFile(data), nil},
		{"no data chunk", wavFile(format), nil},
		{"cut short in the fmt chunk", wavFile(format[:12]), nil},
		{"a fmt chunk under 16 bytes", wavFile(wavChunk("fmt ", 14, format[8:22]), data), nil},
		{"not a WAV file", []byte("ID3\x04 an MP3 file"), nil},
	} {
		got, err := readWAV(tt.file)
		switch {
		case tt.want == nil && err == nil:
			t.Errorf("%s: got %v, want an error", tt.name, got)
		case tt.want != nil && err != nil:
			t.Errorf("%s: %v", tt.name, err)
		case !slices.Equal(got, tt.want):
			t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
		}
	}
}
