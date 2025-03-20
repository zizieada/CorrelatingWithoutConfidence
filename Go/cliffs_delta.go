package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
)

// readFloats reads float64 numbers from a file, one per line
func readFloats(filename string) ([]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var numbers []float64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		num, err := strconv.ParseFloat(line, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid float in file %s: %v", filename, err)
		}
		numbers = append(numbers, num)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return numbers, nil
}

// cliffsDelta calculates Cliff's Delta effect size between two slices of float64 values
func cliffsDelta(lst1, lst2 []float64) float64 {
	// Get lengths of input slices
	m, n := len(lst1), len(lst2)

	// Return 0 if either slice is empty
	if m == 0 || n == 0 {
		return 0
	}

	// Calculate dominance
	var dominance float64 = 0

	// Compare each element from lst1 with each element from lst2
	for _, x := range lst1 {
		for _, y := range lst2 {
			if x > y {
				dominance += 1
			} else if x < y {
				dominance -= 1
			}
		}
	}

	// Calculate final delta
	delta := dominance / float64(m*n)

	return delta
}

func main() {
	// Define command-line flags for file paths
	file1 := flag.String("file1", "", "Path to first input file")
	file2 := flag.String("file2", "", "Path to second input file")

	onlyNumerical := flag.Bool("n", false, "If to output just the numerical values for Cliff's delta and the win probability")
	onlyWinProb := flag.Bool("w", false, "If to output just the win probability")

	flag.Parse()

	// Custom help message
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n", os.Args[0])
		flag.PrintDefaults()
	}

	// Check if both file paths are provided
	if *file1 == "" || *file2 == "" {
		flag.Usage()
		log.Fatal("Usage: cliffs_delta -file1 <path> -file2 <path>")
		os.Exit(1)
	}

	// Read numbers from both files
	lst1, err := readFloats(*file1)
	if err != nil {
		log.Fatalf("Error reading %s: %v", *file1, err)
	}
	lst2, err := readFloats(*file2)
	if err != nil {
		log.Fatalf("Error reading %s: %v", *file2, err)
	}

	// Calculate Cliff's Delta
	delta := cliffsDelta(lst1, lst2)

	win_probability := ((delta + 1) / 2) * 100

	if *onlyNumerical {

		if *onlyWinProb {

			fmt.Printf("%f\n", win_probability)

		} else {

			fmt.Printf("%f\n", delta)
			fmt.Printf("%f\n", win_probability)

		}

	} else {

		fmt.Printf("Cliff's Delta: %f\n", delta)
		fmt.Printf("Win probability: %f\n", win_probability)

	}
}
