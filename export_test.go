//declscope:namespace whisper

package whisper

// Fill takes the module's memory but for less than room bytes, so that a
// test can run out of it. free gives the memory back.
func (m *Model) Fill(room int32) (free func()) {
	m.sem <- struct{}{}
	defer func() { <-m.sem }()
	var held []int32
	_ = m.call(func() { // a trap closes the model, which the test sees next
		for size := int32(1 << 30); size >= room; size /= 2 {
			for p := m.mod.Xgw_malloc(size); p != 0; p = m.mod.Xgw_malloc(size) {
				held = append(held, p)
			}
		}
	})
	return func() {
		m.sem <- struct{}{}
		defer func() { <-m.sem }()
		_ = m.call(func() { // as above
			for _, p := range held {
				m.mod.Xgw_free(p)
			}
		})
	}
}

// Room is how many bytes the module's memory can still give, in blocks of
// 1 MiB or more. It takes them all, then gives them back.
func (m *Model) Room() (n int64) {
	m.sem <- struct{}{}
	defer func() { <-m.sem }()
	_ = m.call(func() { // a trap closes the model, which the test sees next
		var held []int32
		for size := int32(1 << 30); size >= 1<<20; size /= 2 {
			for p := m.mod.Xgw_malloc(size); p != 0; p = m.mod.Xgw_malloc(size) {
				held = append(held, p)
				n += int64(size)
			}
		}
		for _, p := range held {
			m.mod.Xgw_free(p)
		}
	})
	return n
}
