// Copyright 2024 The Arc Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"google.golang.org/api/option"

	"github.com/pointlander/gradient/tf64"
	"github.com/pointlander/matrix"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = .7
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

// Softmax is the softmax function for big numbers
func Softmax(k tf64.Continuation, node int, a *tf64.V, options ...map[string]interface{}) bool {
	c, size, width := tf64.NewV(a.S...), len(a.X), a.S[0]
	S := S
	if len(options) > 0 {
		s, ok := options[0]["S"]
		if ok {
			S = s.(float64)
		}
	}
	max, min := float64(-math.MaxFloat64), math.MaxFloat64
	for _, v := range a.X {
		if v > max {
			max = v
		}
		if v < min {
			min = v
		}
	}
	values := make([]float64, width)
	for i := 0; i < size; i += width {
		s := float64(max) * S
		sum := 0.0
		for j, ax := range a.X[i : i+width] {
			values[j] = math.Exp((float64(ax) - s) / (max - min))
			sum += values[j]
		}
		for _, cx := range values {
			c.X = append(c.X, float64(cx/sum))
		}
	}
	if k(&c) {
		return true
	}
	for i, d := range c.D {
		cx := c.X[i]
		for j := range c.X {
			if j == i {
				a.D[j] += d * cx * (1 - cx)
			} else {
				a.D[j] -= d * cx * c.X[j]
			}
		}
	}
	return false
}

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

// Load loads the data
func Load() []Set {
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
	return sets
}

// NeuralNetwork is the neural network model
func NeuralNetwork() {
	rng := rand.New(rand.NewSource(1))
	sets := Load()
	s := 0

	ix, iy := 0, 0
	ox, oy := 0, 0
	for _, value := range sets[s].Train {
		if len(value.Input) > iy {
			iy = len(value.Input)
		}
		if len(value.Input[0]) > ix {
			ix = len(value.Input[0])
		}
		if len(value.Output) > oy {
			oy = len(value.Output)
		}
		if len(value.Output[0]) > ox {
			ox = len(value.Output[0])
		}
	}
	isize := ix * iy * 10
	osize := ox * oy * 10
	size := isize + osize
	length := len(sets[s].Train) + 1

	set := tf64.NewSet()
	set.Add("q", size, size)
	set.Add("k", size, size)
	set.Add("v", size, size)
	set.Add("w", size, size/2)
	set.Add("b", size/2)
	set.Add("input", size, length)

	for i := range set.Weights {
		w := set.Weights[i]
		size := w.S[0] * w.S[1]
		if strings.HasPrefix(w.N, "input") || strings.HasPrefix(w.N, "b") {
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

	input := set.ByName["input"]
	e := 0
	total := 0.0
	for _, example := range sets[s].Train {
		for i, v := range example.Input {
			for j, vv := range v {
				input.X[e*size+(i*len(v)+j)*10+int(vv)] = 1
				total++
			}
		}
		for i, v := range example.Output {
			for j, vv := range v {
				input.X[e*size+isize+(i*len(v)+j)*10+int(vv)] = 1
				total++
			}
		}
		e++
	}
	cutoff := e*size + isize
	for _, example := range sets[s].Test[:1] {
		for i, v := range example.Input {
			for j, vv := range v {
				input.X[e*size+(i*len(v)+j)*10+int(vv)] = 1
				total++
			}
		}
		for i, v := range example.Output {
			for j := range v {
				input.X[e*size+isize+(i*len(v)+j)*10+int(rng.Intn(10))] = 1
				total++
			}
		}
		e++
	}

	prnt := func() {
		e := 0
		for _, example := range sets[s].Train {
			for i, v := range example.Input {
				for j := range v {
					kk, max := 0, 0.0
					for k := 0; k < 10; k++ {
						value := input.X[e*size+(i*len(v)+j)*10+k]
						if value > max {
							kk, max = k, value
						}
					}
					fmt.Printf("%.1d ", kk)
				}
				fmt.Println()
			}
			fmt.Println()
			for i, v := range example.Output {
				for j := range v {
					kk, max := 0, 0.0
					for k := 0; k < 10; k++ {
						value := input.X[e*size+isize+(i*len(v)+j)*10+k]
						if value > max {
							kk, max = k, value
						}
					}
					fmt.Printf("%.1d ", kk)
				}
				fmt.Println()
			}
			fmt.Println()
			e++
		}
		for _, example := range sets[s].Test[:1] {
			for i, v := range example.Input {
				for j := range v {
					kk, max := 0, 0.0
					for k := 0; k < 10; k++ {
						value := input.X[e*size+(i*len(v)+j)*10+k]
						if value > max {
							kk, max = k, value
						}
					}
					fmt.Printf("%.1d ", kk)
				}
				fmt.Println()
			}
			fmt.Println()
			total, correct := 0, 0
			for i, v := range example.Output {
				for j, vv := range v {
					kk, max := 0, 0.0
					for k := 0; k < 10; k++ {
						value := input.X[e*size+isize+(i*len(v)+j)*10+k]
						if value > max {
							kk, max = k, value
						}
					}
					if byte(kk) == vv {
						correct++
					}
					total++
					fmt.Printf("%.1d ", kk)
				}
				fmt.Println()
			}
			fmt.Println()
			e++
			fmt.Println(correct, float64(correct)/float64(total))
		}
	}
	prnt()
	fmt.Println("---------------------------------------")

	total = math.Sqrt(2.0 / total)
	for i, value := range input.X {
		input.X[i] = value * total
	}

	softmax := tf64.U(Softmax)
	in := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w"), input.Meta()), set.Get("b")))
	q := tf64.Mul(set.Get("q"), in)
	k := tf64.Mul(set.Get("k"), in)
	//v := tf64.Mul(set.Get("v"), in)
	//attention := tf64.T(tf64.Mul(softmax(tf64.Mul(q, k)), tf64.T(v)))
	//output := tf64.Entropy(softmax(attention))
	attention := softmax(tf64.Mul(q, k))
	output := tf64.Entropy(attention)
	loss := tf64.Sum(output)
	/*begin := length - 1
	end := length
	loss := tf64.Sum(tf64.Slice(output, map[string]interface{}{"begin": &begin, "end": &end}))*/
	points := make(plotter.XYs, 0, 8)
	for epoch := 0; epoch < 128; epoch++ {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(epoch/256+1))
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
			der := p.D
			if p.N == "input" {
				der = p.D[cutoff:]
			}
			for _, d := range der {
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
			der := w.D
			if w.N == "input" {
				der = w.D[cutoff:]
			}
			for l, d := range der {
				if w.N == "input" {
					l += cutoff
				}
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
	prnt()
}

func MoveToFrontCoder(block []byte) []byte {
	nodes := [256]byte{}
	var first byte

	for node, _ := range nodes {
		nodes[node] = uint8(node) + 1
	}

	current, index := make([]byte, len(block)), 0
	for _, v := range block {
		var node, next byte
		var symbol byte
		for next = first; next != v; node, next = next, nodes[next] {
			symbol++
		}

		current[index], index = symbol, index+1
		if symbol != 0 {
			first, nodes[node], nodes[next] = next, nodes[next], first
		}
	}
	return current
}

func MoveToFrontRunLengthCoder(block []byte) []byte {

	current, index, length := make([]byte, len(block)), 0, uint64(0)
	outputSymbol := func(symbol byte) {
		current[index], index = symbol, index+1
	}
	outputLength := func() {
		if length > 0 {
			length--
			outputSymbol(uint8(length & 1))
			for length > 1 {
				length = (length - 2) >> 1
				outputSymbol(uint8(length & 1))
			}
			length = 0
		}
	}

	var nodes [256]byte
	var first byte
	for node, _ := range nodes {
		nodes[node] = byte(node) + 1
	}

	for _, v := range block {
		var node, next byte
		var symbol byte
		for next = first; next != v; node, next = next, nodes[next] {
			symbol++
		}

		if symbol == 0 {
			length++
			continue
		}

		first, nodes[node], nodes[next] = next, nodes[next], first

		outputLength()
		outputSymbol(symbol + 1)
	}

	return current[:index]
}

func Entropy(buffer []byte) float64 {
	histogram := [256]float64{}
	sum := 0.0
	for _, v := range buffer {
		histogram[v]++
		sum++
	}
	entropy := 0.0
	for _, v := range histogram {
		if v == 0 {
			continue
		}
		entropy += (v / sum) * (math.Log2(v) - math.Log2(sum))
	}
	return -entropy
}

// KolmogorovComplexity is the kolmogorov complexity model
func KolmogorovComplexity() {
	rng := matrix.Rand(1)
	sets := Load()
	s := 0

	buffer := []byte{}
	for _, line := range sets[s].Train[0].Output {
		buffer = append(buffer, line...)
	}
	cp := make([]byte, len(buffer))
	copy(cp, buffer)
	pressed := BijectiveBurrowsWheelerCoder(buffer)
	fmt.Println(Entropy(cp)*float64(len(cp)), cp)
	fmt.Println(Entropy(pressed)*float64(len(pressed)), pressed)
	output := MoveToFrontRunLengthCoder(pressed)
	fmt.Println(Entropy(output)*float64(len(output)), output)

	buffer = []byte{}
	for i := range sets[s].Train[0].Output[0] {
		for _, line := range sets[s].Train[0].Output {
			buffer = append(buffer, line[i])
		}
	}
	cp = make([]byte, len(buffer))
	copy(cp, buffer)
	pressed = BijectiveBurrowsWheelerCoder(buffer)
	fmt.Println(Entropy(cp)*float64(len(cp)), cp)
	fmt.Println(Entropy(pressed)*float64(len(pressed)), pressed)
	output = MoveToFrontRunLengthCoder(pressed)
	fmt.Println(Entropy(output)*float64(len(output)), output)

	ix, iy := 0, 0
	ox, oy := 0, 0
	for _, value := range sets[s].Train {
		if len(value.Input) > iy {
			iy = len(value.Input)
		}
		if len(value.Input[0]) > ix {
			ix = len(value.Input[0])
		}
		if len(value.Output) > oy {
			oy = len(value.Output)
		}
		if len(value.Output[0]) > ox {
			ox = len(value.Output[0])
		}
	}

	optimizer := matrix.NewOptimizer(&rng, 8, .1, 4, func(samples []matrix.Sample, x ...matrix.Matrix) {
		for ss, sample := range samples {
			x1 := sample.Vars[0][0].Sample()
			y1 := sample.Vars[0][1].Sample()
			z1 := sample.Vars[0][2].Sample()
			w1 := x1.Add(y1.H(z1))

			a := []byte{}
			for _, v := range sets[s].Train {
				for _, vv := range v.Input {
					a = append(a, vv...)
				}
				for _, vv := range v.Output {
					a = append(a, vv...)
				}
			}
			for _, v := range sets[s].Test[:1] {
				for _, vv := range v.Input {
					a = append(a, vv...)
				}
			}

			for y := 0; y < oy; y++ {
				for x := 0; x < ox*10; x += 10 {
					sum := float32(0.0)
					for z := 0; z < 10; z++ {
						value := w1.Data[y*10*ox+x+z]
						if value < 0 {
							value = -value
						}
						sum += value
					}
					max, index := float32(0.0), 0
					for z := 0; z < 10; z++ {
						value := w1.Data[y*10*ox+x+z]
						if value < 0 {
							value = -value
						}
						value /= sum
						if value > max {
							max, index = value, z
						}
					}
					a = append(a, byte(index))
				}
			}
			a = BijectiveBurrowsWheelerCoder(a)
			output = MoveToFrontRunLengthCoder(a)
			samples[ss].Cost = Entropy(output) * float64(len(output))
		}
	}, matrix.NewCoord(ox*10, oy))
	var sample matrix.Sample
	for i := 0; i < 33; i++ {
		sample = optimizer.Iterate()
		fmt.Println(i, sample.Cost)
	}

	x1 := sample.Vars[0][0].Sample()
	y1 := sample.Vars[0][1].Sample()
	z1 := sample.Vars[0][2].Sample()
	w1 := x1.Add(y1.H(z1))
	correct, total := 0.0, 0.0
	for y := 0; y < oy; y++ {
		for x := 0; x < ox*10; x += 10 {
			sum := float32(0.0)
			for z := 0; z < 10; z++ {
				value := w1.Data[y*10*ox+x+z]
				if value < 0 {
					value = -value
				}
				sum += value
			}
			max, index := float32(0.0), 0
			for z := 0; z < 10; z++ {
				value := w1.Data[y*10*ox+x+z]
				if value < 0 {
					value = -value
				}
				value /= sum
				if value > max {
					max, index = value, z
				}
			}
			fmt.Printf("%.1d ", index)
			total++
			if byte(index) == sets[s].Test[0].Output[y][x/10] {
				correct++
			}
		}
		fmt.Println()
	}
	fmt.Println(correct, correct/total)
}

// K computes the K complexity
func K(example Example) (a, b int) {
	buffer := []byte{}
	for _, line := range example.Input {
		buffer = append(buffer, line...)
	}
	a = len(MoveToFrontRunLengthCoder(BijectiveBurrowsWheelerCoder(buffer)))

	buffer = []byte{}
	for i := range example.Input[0] {
		for _, line := range example.Input {
			buffer = append(buffer, line[i])
		}
	}
	b = len(MoveToFrontRunLengthCoder(BijectiveBurrowsWheelerCoder(buffer)))

	return a, b
}

// LLM mode generates a programing specification for input into an llm
func LLM() {
	key := os.Getenv("KEY")
	fmt.Println(key)

	ctx := context.Background()
	// Access your API key as an environment variable (see "Set up your API key" above)
	client, err := genai.NewClient(ctx, option.WithAPIKey(key))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	output := &bytes.Buffer{}
	fmt.Fprintf(output, "You are a programmer tasked with creating a python program that can handle different inputs and produce corresponding outputs. Here are some examples:\n")
	sets := Load()
	i := 0
	for s, set := range sets {
		for _, v := range set.Train {
			//iy := len(v.Input)
			//ix := len(v.Input[0])
			fmt.Fprintln(output)
			//fmt.Fprintf(output, "**Input %d:** %dw %dh ", i+1, ix, iy)
			a, b := K(v)
			fa := float64(a) / float64(len(v.Input)*len(v.Input[0]))
			fb := float64(b) / float64(len(v.Input)*len(v.Input[0]))
			fmt.Fprintf(output, "**Input %d:** compression_factor_a=%f compression_factor_b=%f ", i+1, fa, fb)
			for j, vv := range v.Input {
				for _, s := range vv {
					fmt.Fprintf(output, "%.1d", s)
				}
				if j < len(v.Input)-1 {
					fmt.Fprintf(output, "|")
				}
			}
			fmt.Fprintln(output)

			oy := len(v.Output)
			ox := len(v.Output[0])
			fmt.Fprintf(output, "**Output %d:** type=%d %dw %dh ", i+1, s, ox, oy)
			for j, vv := range v.Output {
				for _, s := range vv {
					fmt.Fprintf(output, "%.1d", s)
				}
				if j < len(v.Output)-1 {
					fmt.Fprintf(output, "|")
				}
			}
			fmt.Fprintln(output)
			i++
		}
	}
	fmt.Fprintf(output, "The program should be written in a clear and efficient manner.\n")

	out, err := os.Create("llm.txt")
	if err != nil {
		panic(err)
	}
	defer out.Close()
	out.Write(output.Bytes())

	fmt.Println("generated prompt")

	// The Gemini 1.5 models are versatile and work with both text-only and multimodal prompts
	model := client.GenerativeModel("gemini-1.5-flash")
	resp, err := model.GenerateContent(ctx, genai.Text(output.String()))
	if err != nil {
		log.Fatal(err)
	}
	for i, candidate := range resp.Candidates {
		out, err := os.Create(fmt.Sprintf("candidate%d.py", i))
		for _, part := range candidate.Content.Parts {
			fmt.Fprintf(out, "%s", part)
		}
		err = out.Close()
		if err != nil {
			panic(err)
		}
	}
}

var (
	// FlagNeuralNetwork is a neural network model
	FlagNeuralNetwork = flag.Bool("nn", false, "neural network mode")
	// FlagKolmogorovComplexity is the kolmogorov complexity based model
	FlagKolmogorovComplexity = flag.Bool("k", false, "kolmogorov complexity model")
	// FlagLLM llm mode
	FlagLLM = flag.Bool("llm", false, "llm mode")
)

func main() {
	flag.Parse()

	if *FlagNeuralNetwork {
		NeuralNetwork()
		return
	} else if *FlagKolmogorovComplexity {
		KolmogorovComplexity()
		return
	} else if *FlagLLM {
		LLM()
		return
	}
}
