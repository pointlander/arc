// Copyright 2024 The Arc Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/pointlander/gradient/tf64"
)

type Example struct {
	Input  [][]byte `json:"input"`
	Output [][]byte `json:"output"`
}

type Set struct {
	Test  []Example `json:"test"`
	Train []Example `json:"train"`
}

func main() {
	dirs, err := os.ReadDir("ARC-AGI/data/training/")
	if err != nil {
		panic(err)
	}
	sets := make([]Set, len(dirs))
	for i, dir := range dirs {
		data, err := os.ReadFile("ARC-AGI/data/training/" + dir.Name())
		if err != nil {
			panic(err)
		}
		err = json.Unmarshal(data, &sets[i])
		if err != nil {
			panic(err)
		}
	}
	fmt.Println("loaded", len(sets))
	test, train := 0, 0
	for _, set := range sets {
		test += len(set.Test)
		train += len(set.Train)
	}
	fmt.Println("test", test)
	fmt.Println("train", train)

	x, y := 0, 0
	for _, value := range sets[0].Train {
		if len(value.Input) > y {
			y = len(value.Input)
		}
		if len(value.Input[0]) > x {
			x = len(value.Input[0])
		}
		if len(value.Output) > y {
			y = len(value.Output)
		}
		if len(value.Output[0]) > x {
			x = len(value.Output[0])
		}
	}
	size := x * y * 10
	input := tf64.NewV(size, len(sets[0].Train))
	input.X = input.X[:cap(input.X)]

	set := tf64.NewSet()
	set.Add("q", size, size)
	set.Add("k", size, size)
	set.Add("v", size, size)

	q := tf64.Mul(set.Get("q"), input.Meta())
	k := tf64.Mul(set.Get("k"), input.Meta())
	v := tf64.Mul(set.Get("v"), input.Meta())
	attention := tf64.T(tf64.Mul(tf64.Softmax(tf64.Mul(q, k)), tf64.T(v)))
	output := tf64.Entropy(tf64.Softmax(attention))
	output(func(a *tf64.V) bool {
		fmt.Println(a.S)
		return true
	})
}
