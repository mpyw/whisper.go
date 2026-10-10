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
