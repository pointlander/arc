// Copyright 2010 The Arc Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"sort"
)

type rotation struct {
	int
	s []uint8
}

type Rotations []rotation

func (r Rotations) Len() int {
	return len(r)
}

func less(a, b rotation) bool {
	la, lb, ia, ib := len(a.s), len(b.s), a.int, b.int
	for {
		if x, y := a.s[ia], b.s[ib]; x != y {
			return x < y
		}
		ia, ib = ia+1, ib+1
		if ia == la {
			ia = 0
		}
		if ib == lb {
			ib = 0
		}
		if ia == a.int && ib == b.int {
			break
		}
	}
	return false
}

func (r Rotations) Less(i, j int) bool {
	return less(r[i], r[j])
}

func (r Rotations) Swap(i, j int) {
	r[i], r[j] = r[j], r[i]
}

func merge(left, right, out Rotations) {
	for len(left) > 0 && len(right) > 0 {
		if less(left[0], right[0]) {
			out[0], left = left[0], left[1:]
		} else {
			out[0], right = right[0], right[1:]
		}
		out = out[1:]
	}
	copy(out, left)
	copy(out, right)
}

func psort(in Rotations, s chan<- bool) {
	if len(in) < 1024 {
		sort.Sort(in)
		s <- true
		return
	}

	l, r, split := make(chan bool), make(chan bool), len(in)/2
	left, right := in[:split], in[split:]
	go psort(left, l)
	go psort(right, r)
	_, _ = <-l, <-r
	out := make(Rotations, len(in))
	merge(left, right, out)
	copy(in, out)
	s <- true
}

func BijectiveBurrowsWheelerCoder(block []byte) []byte {
	var lyndon Lyndon
	var rotations Rotations
	wait := make(chan bool)
	buffer := make([]uint8, len(block))

	copy(buffer, block)
	lyndon.Factor(buffer)

	/* rotate */
	if length := len(block); cap(rotations) < length {
		rotations = make(Rotations, length)
	} else {
		rotations = rotations[:length]
	}
	r := 0
	for _, word := range lyndon.Words {
		for i, _ := range word {
			rotations[r], r = rotation{i, word}, r+1
		}
	}

	go psort(rotations, wait)
	<-wait

	/* output the last character of each rotation */
	for i, j := range rotations {
		if j.int == 0 {
			j.int = len(j.s)
		}
		block[i] = j.s[j.int-1]
	}
	return block
}
