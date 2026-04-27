package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
)

// readFloats reads one float per line.
func readFloats(filename string) ([]float64, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var nums []float64
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		v, err := strconv.ParseFloat(line, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid float in %s: %v", filename, err)
		}
		nums = append(nums, v)
	}
	return nums, sc.Err()
}

// cliffsDelta computes the Cliff's Delta effect size between two samples.
func cliffsDelta(lst1, lst2 []float64) float64 {
	m, n := len(lst1), len(lst2)
	if m == 0 || n == 0 {
		return 0
	}
	var dominance int64
	for _, x := range lst1 {
		for _, y := range lst2 {
			if x > y {
				dominance++
			} else if x < y {
				dominance--
			}
		}
	}
	return float64(dominance) / float64(m*n)
}

func main() {
	file1 := flag.String("file1", "", "Path to first input file")
	file2 := flag.String("file2", "", "Path to second input file")
	onlyNumerical := flag.Bool("n", false, "Output only the numerical values for Cliff's delta and win probability")
	onlyWinProb := flag.Bool("w", false, "Output only the win probability")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s -file1 <path> -file2 <path> [options]\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if *file1 == "" || *file2 == "" {
		flag.Usage()
		os.Exit(1)
	}

	lst1, err := readFloats(*file1)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading %s: %v\n", *file1, err)
		os.Exit(1)
	}
	lst2, err := readFloats(*file2)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading %s: %v\n", *file2, err)
		os.Exit(1)
	}

	delta := cliffsDelta(lst1, lst2)
	winProb := ((delta + 1) / 2) * 100

	switch {
	case *onlyNumerical && *onlyWinProb:
		fmt.Printf("%f\n", winProb)
	case *onlyNumerical:
		fmt.Printf("%f\n%f\n", delta, winProb)
	default:
		fmt.Printf("Cliff's Delta: %f\n", delta)
		fmt.Printf("Win probability: %f\n", winProb)
	}
}
