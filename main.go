// Copyright 2024 The Arc Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tf64"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = .5
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Example is a learning example
type Example struct {
	Input  [][]byte `json:"input"`
	Output [][]byte `json:"output"`
}

// Set is a set of examples
type Set struct {
	Test  []Example `json:"test"`
	Train []Example `json:"train"`
}

func main() {
	rng := rand.New(rand.NewSource(1))

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
	length := len(sets[0].Train)
	input := tf64.NewV(size, length)
	input.X = input.X[:cap(input.X)]
	for e, example := range sets[0].Train {
		for i, v := range example.Input {
			for j, vv := range v {
				input.X[e*length+i*len(v)+j] = float64(vv) / 10.0
			}
		}
	}

	set := tf64.NewSet()
	set.Add("q", size, size)
	set.Add("k", size, size)
	set.Add("v", size, size)

	for i := range set.Weights {
		w := set.Weights[i]
		size := w.S[0] * w.S[1]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:size]
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < size; i++ {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float64, len(w.X))
		}
	}

	q := tf64.Mul(set.Get("q"), input.Meta())
	k := tf64.Mul(set.Get("k"), input.Meta())
	v := tf64.Mul(set.Get("v"), input.Meta())
	attention := tf64.T(tf64.Mul(tf64.Softmax(tf64.Mul(q, k)), tf64.T(v)))
	output := tf64.Entropy(tf64.Softmax(attention))
	loss := tf64.Sum(output)
	points := make(plotter.XYs, 0, 8)
	for epoch := 0; epoch < 256; epoch++ {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(epoch+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}
		set.Zero()

		cost := tf64.Gradient(loss).X[0]

		points = append(points, plotter.XY{X: float64(epoch), Y: float64(cost)})
		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for l, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][l] + (1-B1)*g
				v := B2*w.States[StateV][l] + (1-B2)*g*g
				w.States[StateM][l] = m
				w.States[StateV][l] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		fmt.Println(cost)
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}

	output(func(a *tf64.V) bool {
		fmt.Println(a.S)
		return true
	})

	min, max := math.MaxFloat64, -math.MaxFloat64
	for _, v := range set.ByName["q"].X {
		if v > max {
			max = v
		}
		if v < min {
			min = v
		}
	}
	fmt.Println("q", min, max)
	min, max = math.MaxFloat64, -math.MaxFloat64
	for _, v := range set.ByName["k"].X {
		if v > max {
			max = v
		}
		if v < min {
			min = v
		}
	}
	fmt.Println("k", min, max)
	min, max = math.MaxFloat64, -math.MaxFloat64
	for _, v := range set.ByName["v"].X {
		if v > max {
			max = v
		}
		if v < min {
			min = v
		}
	}
	fmt.Println("v", min, max)
}
