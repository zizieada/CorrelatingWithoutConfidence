package main

import (
	"bufio"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

// SceneData holds the information for each scene/stimulus
type SceneData struct {
	Name       string
	Mean       float64
	Std        float64
	RawScores  []float64
	HasRawData bool
}

var cpuprofile = flag.String("cpuprofile", "", "write CPU profile to file")
var memprofile = flag.String("memprofile", "", "write memory profile to file")

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	objectiveFile := flag.String("objective", "objective_scores.csv", "Path to the objective scores")
	flag.StringVar(objectiveFile, "o", "objective_scores.csv", "Path to the objective scores (short)")

	subjectiveFile := flag.String("subjective", "subjective_scores.csv", "Path to the subjective scores")
	flag.StringVar(subjectiveFile, "s", "subjective_scores.csv", "Path to the subjective scores (short)")

	nBootstrap := flag.Int("num_bootstraps", 14999, "Number of bootstrap samples")
	flag.IntVar(nBootstrap, "nb", 14999, "Number of bootstrap samples (short)")

	bootstrapScenes := flag.Bool("boostrap_scenes", false, "If to compute scene bootstrapping")
	flag.BoolVar(bootstrapScenes, "b", false, "If to compute scene bootstrapping (short)")

	nVotes := flag.Int("votes", 15, "Number of votes per stimuli")
	flag.IntVar(nVotes, "nv", 15, "Number of votes per stimuli (short)")

	lowerBound := flag.Float64("lower_bound", 0, "Lower bound of the rating scale")
	flag.Float64Var(lowerBound, "l", 0, "Lower bound of the rating scale (short)")

	upperBound := flag.Float64("upper_bound", 100, "Upper bound of the rating scale")
	flag.Float64Var(upperBound, "u", 100, "Upper bound of the rating scale (short)")

	pearsonCorr := flag.Bool("pearson", false, "If to compute the Pearson correlation")
	flag.BoolVar(pearsonCorr, "r", false, "If to compute the Pearson correlation (short)")

	spearmanCorr := flag.Bool("spearman", false, "If to compute the Spearman correlation")
	flag.BoolVar(spearmanCorr, "rho", false, "If to compute the Spearman correlation (short)")

	kendallCorr := flag.Bool("kendall", false, "If to compute the Kendall correlation")
	flag.BoolVar(kendallCorr, "tau", false, "If to compute the Kendall correlation (short)")

	saveTxt := flag.Bool("txt", false, "If to save the output correlation distributions as .txt files")
	saveCSV := flag.Bool("csv", false, "If to save the output correlation distributions as a combined .csv file")

	fileNamePrefix := flag.String("name", "", "What prefix would you like to add to the output .txt files (if any)")

	outputPath := flag.String("output", "", "Output path to a directory")

	// Custom help message
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n", os.Args[0])
		flag.PrintDefaults()
	}

	// Parse the command-line flags
	flag.Parse()

	// Display help if no arguments are provided
	if len(os.Args) == 1 {
		flag.Usage()
		os.Exit(1)
	}

	if *lowerBound >= *upperBound {
		fmt.Println("Error: The lower bound of the rating scale must be smaller than the upper bound.")
		os.Exit(1) // Exit with a non-zero status to indicate an error
	}

	// Construct the correlation coefficient list based on user input
	var corrCoeffs []string

	if *pearsonCorr {
		corrCoeffs = append(corrCoeffs, "pearson")
	}
	if *spearmanCorr {
		corrCoeffs = append(corrCoeffs, "spearman")
	}
	if *kendallCorr {
		corrCoeffs = append(corrCoeffs, "kendall")
	}

	// If none were selected, default to Pearson and print a warning
	if len(corrCoeffs) == 0 {
		fmt.Println("No correlation coefficient selected. Defaulting to Pearson correlation.")
		corrCoeffs = append(corrCoeffs, "pearson")
	}

	// CPU Profiling
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			fmt.Println("could not create CPU profile:", err)
			return
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Println("could not start CPU profile:", err)
			return
		}
		defer pprof.StopCPUProfile()
	}

	numColumns, headers, err := GetCSVHeader(*objectiveFile)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	if numColumns < 2 {

		fmt.Println("Error: The csv with objective scores seems to have less than 2 columns!.")

	}

	subjectiveScores := readSubjectiveScores(*subjectiveFile)

	var subjectiveDistribution map[string][]float64

	if rawDataAvailable(subjectiveScores) {

		subjectiveDistribution = extractRawScores(subjectiveScores)

	} else {

		subjectiveDistribution = processScoresWithPool(subjectiveScores, *nVotes, *lowerBound, *upperBound)

	}

	bootstrappedSubjective := parallelBootstrap(subjectiveDistribution, 1499, 0)
	var fullPath string
	fileMapPearson := make(map[string]string)
	fileMapSpearman := make(map[string]string)
	fileMapKendall := make(map[string]string)

	for id, metric := range headers[1:] {

		objectiveScores := readObjectiveScores(*objectiveFile, id+1)

		distributions := computeCorrelationDistributions(bootstrappedSubjective, objectiveScores, *nBootstrap, *bootstrapScenes, corrCoeffs)

		prefix := ""

		if *fileNamePrefix != "" {

			prefix = *fileNamePrefix + "_" + metric + "_"

		}

		fmt.Printf("Results for %s\n", metric)
		// Print a summary of the computed distributions.
		for coeff, values := range distributions {

			mean, lower, upper := bootstrapConfidenceInterval(values)
			fmt.Printf("%s: %.4f\n95%% CI: [%.4f, %.4f]\n", coeff, mean, lower, upper)
			fmt.Printf("\n")

			if *outputPath != "" {

				err := ensureDirExists(*outputPath)

				if err != nil {
					fmt.Println("Failed to ensure directory exists:", err)
					return
				}

				fullPath = filepath.Join(*outputPath, prefix+coeff+".txt")

			} else {

				fullPath = prefix + coeff + ".txt"

			}

			if coeff == "pearson" {
				fileMapPearson[metric] = fullPath
			}

			if coeff == "spearman" {
				fileMapSpearman[metric] = fullPath
			}

			if coeff == "kendall" {
				fileMapKendall[metric] = fullPath
			}

			if *saveTxt {

				err := saveToText(fullPath, values)
				if err != nil {
					fmt.Println("Could not save "+fullPath, err)
				}
			}

		}
		fmt.Printf("\n")

	}

	if *saveCSV {

		prefix := ""

		if *fileNamePrefix != "" {

			prefix = *fileNamePrefix + "_"

		}

		if len(fileMapPearson) != 0 {
			CombineTxtToCSV(fileMapPearson, *outputPath, prefix+"pearson.csv")
		}

		if len(fileMapSpearman) != 0 {
			CombineTxtToCSV(fileMapSpearman, *outputPath, prefix+"spearman.csv")
		}

		if len(fileMapKendall) != 0 {
			CombineTxtToCSV(fileMapKendall, *outputPath, prefix+"kendall.csv")
		}
	}

	// corrResults := computeMeanCorrelation(bootstrappedSubjective, objectiveScores)
	// fmt.Println("Correlation Results:", corrResults)

	// Memory Profiling
	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			fmt.Println("could not create memory profile:", err)
			return
		}
		defer f.Close()
		pprof.WriteHeapProfile(f)
	}

}

func trackTime(start time.Time, name string) {
	fmt.Printf("%s took %v\n", name, time.Since(start))
}

// GetCSVHeader reads the header of a CSV file and returns the number of columns and their names.
func GetCSVHeader(filePath string) (int, []string, error) {
	// Check if the file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return 0, nil, fmt.Errorf("file does not exist: %s", filePath)
	}

	// Open the file
	file, err := os.Open(filePath)
	if err != nil {
		return 0, nil, fmt.Errorf("unable to open file: %s", err)
	}
	defer file.Close()

	// Create a CSV reader
	reader := csv.NewReader(file)

	// Read the header
	header, err := reader.Read()
	if err != nil {
		return 0, nil, fmt.Errorf("unable to read header: %s", err)
	}

	// Return the number of columns and the header slice
	return len(header), header, nil
}

func rawDataAvailable(subjectiveScores map[string]*SceneData) bool {
	for _, scene := range subjectiveScores {
		if !scene.HasRawData {
			return false // As soon as one is false, return false
		}
	}
	return true // If loop completes, all are true
}

func extractRawScores(subjectiveScores map[string]*SceneData) map[string][]float64 {
	result := make(map[string][]float64)

	for _, scene := range subjectiveScores {
		result[scene.Name] = scene.RawScores
	}

	return result
}

// // Optimized version
// func processScores(subjectiveScores map[string]*SceneData, nVotes int, lowerBound float64, upperBound float64) map[string][]float64 {

// 	defer trackTime(time.Now(), "processScores") // Defer function call

// 	// Pre-allocate the map with expected size
// 	results := make(map[string][]float64, len(subjectiveScores))

// 	// Use a channel to control concurrent operations
// 	concurrencyLimit := runtime.GOMAXPROCS(0) // Or another reasonable number
// 	sem := make(chan struct{}, concurrencyLimit)

// 	var wg sync.WaitGroup
// 	var mu sync.Mutex

// 	// Create a custom mutex-protected map writer
// 	mapWriter := func(key string, samples []float64) {
// 		mu.Lock()
// 		results[key] = samples
// 		mu.Unlock()
// 	}

// 	for key, scene := range subjectiveScores {
// 		wg.Add(1)
// 		sem <- struct{}{} // Acquire semaphore

// 		go func(k string, s SceneData) {
// 			defer wg.Done()
// 			defer func() { <-sem }() // Release semaphore

// 			// Process data without holding the lock
// 			samples := getSamples(s.Mean, s.Std, nVotes, lowerBound, upperBound)

// 			// Minimize time spent holding the lock
// 			mapWriter(k, samples)
// 		}(key, *scene)
// 	}

// 	wg.Wait()
// 	close(sem)

// 	return results
// }

// Alternative approach using worker pool
func processScoresWithPool(subjectiveScores map[string]*SceneData, nVotes int, lowerBound float64, upperBound float64) map[string][]float64 {

	// defer trackTime(time.Now(), "processScoresWithPool") // Defer function call

	type workItem struct {
		key   string
		scene SceneData
	}

	results := make(map[string][]float64, len(subjectiveScores))
	var mu sync.Mutex

	// Create work channel
	work := make(chan workItem, len(subjectiveScores))

	// Number of worker goroutines
	numWorkers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range work {
				samples := getSamples(item.scene.Mean, item.scene.Std, nVotes, lowerBound, upperBound)
				mu.Lock()
				results[item.key] = samples
				mu.Unlock()
			}
		}()
	}

	// Send work to workers
	for key, scene := range subjectiveScores {
		work <- workItem{key, *scene}
	}
	close(work)

	wg.Wait()
	return results
}

func readObjectiveScores(filename string, id int) map[string]float64 {
	file, err := os.Open(filename)
	if err != nil {
		//fmt.Printf("Error: Unable to open file %s: %v\n", filename, err)
		panic(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}

	scores := make(map[string]float64)
	for i, record := range records {
		if i == 0 { // Skip header
			continue
		}
		name := record[0]
		score, err := strconv.ParseFloat(record[id], 64)
		if err != nil {
			continue
		}
		scores[name] = score
	}

	return scores
}

func readSubjectiveScores(filename string) map[string]*SceneData {
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}

	scores := make(map[string]*SceneData)
	headers := records[0]

	// Determine if we have raw scores or mean/std
	hasRawScores := len(headers) == 2 && strings.Contains(strings.ToLower(headers[1]), "scores")

	for i, record := range records {
		if i == 0 { // Skip header
			continue
		}

		name := record[0]
		sceneData := &SceneData{Name: name}

		if hasRawScores {
			// Parse raw scores from comma-separated string
			scoreStr := strings.Trim(record[1], "[]")
			scoreStrs := strings.Split(scoreStr, ",")
			sceneData.RawScores = make([]float64, 0, len(scoreStrs))

			for _, s := range scoreStrs {
				score, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
				if err != nil {
					continue
				}
				sceneData.RawScores = append(sceneData.RawScores, score)
			}
			sceneData.HasRawData = true

			// Calculate mean and std from raw scores
			sceneData.Mean = stat.Mean(sceneData.RawScores, nil)
			sceneData.Std = stat.StdDev(sceneData.RawScores, nil)
		} else {
			// Parse mean and std directly
			mean, err := strconv.ParseFloat(record[1], 64)
			if err != nil {
				continue
			}
			std, err := strconv.ParseFloat(record[2], 64)
			if err != nil {
				continue
			}
			sceneData.Mean = mean
			sceneData.Std = std
			sceneData.HasRawData = false
		}

		scores[name] = sceneData
	}

	return scores
}

// betaDistribution generates samples from a beta distribution with specified mean, standard deviation, and bounds.
func betaDistribution(mean, std, lower, upper float64, nSamples int) ([]float64, error) {
	// Input validation
	if mean <= 0 || std <= 0 {
		samples := make([]float64, nSamples)
		for i := range samples {
			samples[i] = mean
		}
		return samples, nil
	}

	if mean < lower || mean > upper {
		return nil, errors.New("mean must be between lower and upper bounds")
	}

	// Rescale mean and variance to [0, 1] range
	rescaledMean := (mean - lower) / (upper - lower)
	rescaledVar := math.Pow(std/(upper-lower), 2)

	// Check for near-zero variance
	if rescaledVar < 1e-10 {
		samples := make([]float64, nSamples)
		for i := range samples {
			samples[i] = mean
		}
		return samples, nil
	}

	// Compute alpha and beta parameters
	temp := (rescaledMean * (1 - rescaledMean) / rescaledVar) - 1
	alpha := rescaledMean * temp
	betaParam := (1 - rescaledMean) * temp

	// Validate alpha and beta parameters
	if alpha <= 0 || betaParam <= 0 {
		samples := make([]float64, nSamples)
		for i := range samples {
			samples[i] = mean - 0.5
		}
		return samples, nil
	}

	// Use "golang.org/x/exp/rand" since gonum requires it
	src := rand.NewSource(uint64(time.Now().UnixNano()))
	rng := rand.New(src)

	// Generate beta-distributed samples
	betaDist := distuv.Beta{Alpha: alpha, Beta: betaParam, Src: rng}
	samples := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		samples[i] = lower + betaDist.Rand()*(upper-lower)
	}

	return samples, nil

}

// Cached standard normal distribution and constant.
var stdNorm = distuv.Normal{Mu: 0, Sigma: 1}
var sqrt2Pi = math.Sqrt(2 * math.Pi)

func standardNormalPDF(x float64) float64 {
	return math.Exp(-0.5*x*x) / sqrt2Pi
}

func standardNormalCDF(x float64) float64 {
	return stdNorm.CDF(x)
}

func adjustTruncatedNormalParams(targetMean, targetStd, lower, upper float64, maxIterations int, epsilon float64) (float64, float64) {
	mu := targetMean
	sigma := targetStd

	// Helper function to compute the truncated normal moments.
	truncatedNormalMoments := func(mu, sigma, lower, upper float64) (float64, float64) {
		if sigma < epsilon {
			return mu, sigma
		}

		a := (lower - mu) / sigma
		b := (upper - mu) / sigma

		// Cache PDF and CDF values.
		pdfA := standardNormalPDF(a)
		pdfB := standardNormalPDF(b)
		cdfA := standardNormalCDF(a)
		cdfB := standardNormalCDF(b)

		alpha := pdfA - pdfB
		beta := cdfB - cdfA

		if math.Abs(beta) < epsilon {
			return mu, sigma
		}

		// Compute truncated mean.
		truncatedMean := mu + (alpha/beta)*sigma

		// Use multiplication instead of math.Pow for efficiency.
		varianceTerm := math.Max(0, 1+((a*pdfA-b*pdfB)/beta)-(alpha*alpha)/(beta*beta))
		truncatedVar := sigma * sigma * varianceTerm

		return truncatedMean, math.Sqrt(truncatedVar)
	}

	// Iteratively adjust parameters.
	for i := 0; i < maxIterations; i++ {
		truncMean, truncStd := truncatedNormalMoments(mu, sigma, lower, upper)
		mu += targetMean - truncMean

		if truncStd < epsilon {
			fmt.Printf("Warning: truncStd is very small (%.6f) at iteration %d. Stopping iterations.\n", truncStd, i)
			break
		}

		sigma *= targetStd / truncStd
	}
	return mu, sigma
}

// Old code
// func truncatedNormalSamples(mean, std, lower, upper float64, nSamples int) []float64 {
// 	rng := rand.New(rand.NewSource(uint64(time.Now().UnixNano())))
// 	norm := distuv.Normal{Mu: mean, Sigma: std, Src: rng}

// 	samples := make([]float64, 0, nSamples)
// 	for len(samples) < nSamples {
// 		sample := norm.Rand()
// 		if sample >= lower && sample <= upper {
// 			samples = append(samples, sample)
// 		}
// 	}
// 	return samples
// }

func truncatedNormalSamplesQuantile(mean, std, lower, upper float64, nSamples int) []float64 {
	// Create a normal distribution with the desired parameters.
	norm := distuv.Normal{Mu: mean, Sigma: std}
	// Compute CDF values at the truncation boundaries.
	cdfLower := norm.CDF(lower)
	cdfUpper := norm.CDF(upper)
	samples := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		// Draw a uniform random number in the [cdfLower, cdfUpper] interval.
		u := rand.Float64()*(cdfUpper-cdfLower) + cdfLower
		// Use the inverse CDF (Quantile) to obtain a sample from the truncated distribution.
		samples[i] = norm.Quantile(u)
	}
	return samples
}

// computeStats calculates the mean and standard deviation of a slice of float64 values.
func computeStats(samples []float64) (float64, float64) {
	n := float64(len(samples))
	if n == 0 {
		return 0, 0
	}

	// Compute mean
	sum := 0.0
	for _, v := range samples {
		sum += v
	}
	mean := sum / n

	// Compute standard deviation
	varianceSum := 0.0
	for _, v := range samples {
		varianceSum += (v - mean) * (v - mean)
	}
	std := math.Sqrt(varianceSum / n)

	return mean, std
}

// getSamples generates samples with a sophisticated fallback mechanism.
// It tries the beta distribution first, then falls back to a truncated normal distribution if necessary.
func getSamples(mean, std float64, nSamples int, lower, upper float64) []float64 {
	if std <= 0 {
		// If std is zero or negative, return an array filled with the mean
		samples := make([]float64, nSamples)
		for i := range samples {
			samples[i] = mean
		}
		return samples
	}

	// Try generating samples from the beta distribution
	samples, _ := betaDistribution(mean, std, lower, upper, nSamples)
	sampleMean, sampleStd := computeStats(samples)

	// If the generated beta samples are too far from the target mean/std, adjust and switch to truncated normal
	if math.Abs(sampleMean-mean) > 0.2 || math.Abs(sampleStd-std) > 0.1 {
		adjustedMean, adjustedStd := adjustTruncatedNormalParams(mean, std, lower, upper, 100, 1e-8)
		truncSamples := truncatedNormalSamplesQuantile(adjustedMean, adjustedStd, lower, upper, nSamples)

		truncMean, truncStd := computeStats(truncSamples)

		// Choose the distribution that better matches the target mean and std
		if (math.Abs(sampleMean-mean) + math.Abs(sampleStd-std)) < (math.Abs(truncMean-mean) + math.Abs(truncStd-std)) {
			return samples
		} else {
			return truncSamples
		}
	}

	// If beta samples are already good enough, return them
	return samples
}

// bootstrapVotes generates bootstrapped mean samples for a slice of votes.
// For each iteration, it resamples the votes (with replacement) and computes the mean.
func bootstrapVotes(votes []float64, nIterations int) []float64 {
	n := len(votes)
	if n == 0 {
		return nil
	}
	results := make([]float64, nIterations)
	for i := 0; i < nIterations; i++ {
		sum := 0.0
		// Resample 'n' votes with replacement.
		for j := 0; j < n; j++ {
			index := rand.Intn(n)
			sum += votes[index]
		}
		results[i] = sum / float64(n)
	}
	return results
}

// bootstrapResult is a helper type for sending results from workers.
type bootstrapResult struct {
	key   string
	means []float64
}

// parallelBootstrap performs bootstrapping in parallel across the keys of sceneVotes.
// It returns a map with the same keys and the corresponding bootstrapped mean samples.
func parallelBootstrap(sceneVotes map[string][]float64, nIterations int, nProcesses int) map[string][]float64 {
	// Use all available cores if nProcesses is not specified.
	if nProcesses <= 0 {
		nProcesses = runtime.NumCPU()
	}

	// Channel for sending keys to process.
	keysChan := make(chan string)
	// Channel for collecting results.
	resultsChan := make(chan bootstrapResult)
	var wg sync.WaitGroup

	// Start worker goroutines.
	for i := 0; i < nProcesses; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for key := range keysChan {
				votes := sceneVotes[key]
				bootstrapMeans := bootstrapVotes(votes, nIterations)
				resultsChan <- bootstrapResult{key: key, means: bootstrapMeans}
			}
		}()
	}

	// Feed the keys into the keysChan.
	go func() {
		for key := range sceneVotes {
			keysChan <- key
		}
		close(keysChan)
	}()

	// Close the results channel once all workers are done.
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Gather results.
	resultMap := make(map[string][]float64)
	for res := range resultsChan {
		resultMap[res.key] = res.means
	}
	return resultMap
}

// pearson computes the Pearson correlation coefficient.
func pearson(x, y []float64) float64 {
	return stat.Correlation(x, y, nil)
}

// rank returns the ranks (averaging ties) of the values in the slice.
func rank(values []float64) []float64 {
	type pair struct {
		value float64
		index int
	}
	n := len(values)
	pairs := make([]pair, n)
	for i, v := range values {
		pairs[i] = pair{v, i}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].value < pairs[j].value })

	ranks := make([]float64, n)
	for i := 0; i < n; {
		j := i + 1
		// Handle ties by averaging their rank.
		for j < n && pairs[j].value == pairs[i].value {
			j++
		}
		avgRank := float64(i+j+1) / 2.0 // ranks are 1-indexed
		for k := i; k < j; k++ {
			ranks[pairs[k].index] = avgRank
		}
		i = j
	}
	return ranks
}

// spearman computes the Spearman correlation coefficient.
func spearman(x, y []float64) float64 {
	rx := rank(x)
	ry := rank(y)
	return stat.Correlation(rx, ry, nil)
}

// kendall computes Kendall's tau correlation coefficient (naively, O(n²)).
func kendall(x, y []float64) float64 {
	n := len(x)
	var concordant, discordant int
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dx := x[i] - x[j]
			dy := y[i] - y[j]
			if dx*dy > 0 {
				concordant++
			} else if dx*dy < 0 {
				discordant++
			}
		}
	}
	total := n * (n - 1) / 2
	if total == 0 {
		return 0
	}
	return float64(concordant-discordant) / float64(total)
}

// mean computes the average of a slice.
// func mean(arr []float64) float64 {
// 	sum := 0.0
// 	for _, v := range arr {
// 		sum += v
// 	}
// 	return sum / float64(len(arr))
// }

func processWrapperCombined(subjective map[string][]float64, objective map[string]float64,
	bootstrapScenes bool, corrCoeffs []string, keys []string) []float64 {

	var sampledKeys []string
	n := len(keys)
	if bootstrapScenes {
		// Sample scenes with replacement.
		sampledKeys = make([]string, n)
		for i := 0; i < n; i++ {
			sampledKeys[i] = keys[rand.Intn(n)]
		}
	} else {
		sampledKeys = keys
	}

	// For each scene, compute the mean of its subjective ratings.
	bootSubjective := make([]float64, len(sampledKeys))
	objRatings := make([]float64, len(sampledKeys))
	for i, key := range sampledKeys {
		ratings := subjective[key]

		// Pick a random index and retrieve the value.
		index := rand.Intn(len(ratings))
		randomValue := ratings[index]
		// bootSubjective[i] = mean(ratings)

		bootSubjective[i] = randomValue // THIS PART IS IMPORTANT!
		objRatings[i] = objective[key]
	}

	results := make([]float64, 0, len(corrCoeffs))
	for _, coeff := range corrCoeffs {
		lc := strings.ToLower(coeff)
		if lc == "pearson" || coeff == "r" {
			results = append(results, math.Abs(pearson(bootSubjective, objRatings)))
		}
		if lc == "spearman" || coeff == "rho" {
			results = append(results, math.Abs(spearman(bootSubjective, objRatings)))
		}
		if lc == "kendall" || coeff == "tau" {
			results = append(results, math.Abs(kendall(bootSubjective, objRatings)))
		}
	}
	return results
}

// computeCorrelationDistributions performs nBootstrap iterations in parallel.
// It returns a map where each key is the name of a correlation coefficient and the value
// is a slice of its bootstrapped (absolute) values.
func computeCorrelationDistributions(subjective map[string][]float64, objective map[string]float64,
	nBootstrap int, bootstrapScenes bool, corrCoeffs []string) map[string][]float64 {

	// Extract common keys.
	var keys []string
	for k := range subjective {
		if _, ok := objective[k]; ok {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)

	// Prepare the results storage.
	distributions := make(map[string][]float64)
	for _, coeff := range corrCoeffs {
		distributions[coeff] = make([]float64, nBootstrap)
	}

	bootstrapResults := make([][]float64, nBootstrap)
	numWorkers := runtime.NumCPU()
	jobs := make(chan int, nBootstrap)
	var wg sync.WaitGroup

	worker := func() {
		defer wg.Done()
		for i := range jobs {
			bootstrapResults[i] = processWrapperCombined(subjective, objective, bootstrapScenes, corrCoeffs, keys)
		}
	}

	// Launch workers.
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go worker()
	}

	for i := 0; i < nBootstrap; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	// Aggregate results.
	for i := 0; i < nBootstrap; i++ {
		res := bootstrapResults[i]
		for j, coeff := range corrCoeffs {
			distributions[coeff][i] = res[j]
		}
	}
	return distributions
}

// computeMeanCorrelation calculates correlation between mean subjective ratings and objective scores.
func computeMeanCorrelation(subjective map[string][]float64, objective map[string]float64) map[string]float64 {
	var means []float64
	var objValues []float64

	for key, scores := range subjective {
		if obj, exists := objective[key]; exists {
			// Compute mean of subjective scores
			var sum float64
			for _, score := range scores {
				sum += score
			}
			mean := sum / float64(len(scores))

			means = append(means, mean)
			objValues = append(objValues, obj)
		}
	}

	// Compute correlations
	results := map[string]float64{
		"pearson":  pearson(means, objValues),
		"spearman": spearman(means, objValues),
		"kendall":  kendall(means, objValues),
	}

	return results
}

// bootstrapConfidenceInterval computes the mean, lower (2.5th percentile),
// and upper (97.5th percentile) from a slice of values.
func bootstrapConfidenceInterval(data []float64) (mean, lower, upper float64) {
	n := len(data)
	if n == 0 {
		return 0, 0, 0
	}

	// Compute mean.
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	mean = sum / float64(n)

	// Copy and sort the data.
	sortedData := make([]float64, n)
	copy(sortedData, data)
	sort.Float64s(sortedData)

	// Compute indices for the 2.5% and 97.5% percentiles.
	lowerIndex := int(0.025 * float64(n))
	upperIndex := int(0.975 * float64(n))
	// Ensure indices are within bounds.
	if lowerIndex < 0 {
		lowerIndex = 0
	}
	if upperIndex >= n {
		upperIndex = n - 1
	}

	lower = sortedData[lowerIndex]
	upper = sortedData[upperIndex]
	return
}

func saveToText(filename string, data []float64) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write each value on a new line
	for _, value := range data {
		_, err := fmt.Fprintf(file, "%f\n", value)
		if err != nil {
			return err
		}
	}
	return nil
}

func ensureDirExists(path string) error {
	_, err := os.Stat(path)
	if err == nil {
		// Path exists, nothing to do
		return nil
	}
	if os.IsNotExist(err) {
		// Path does not exist, create it
		err = os.MkdirAll(path, os.ModeDir|0755) // Creates the directory and any necessary parent directories
		if err != nil {
			fmt.Println("Error creating directory:", err)
			return err
		}
		return nil
	}
	// Some other error occurred
	fmt.Println("Error checking path:", err)
	return err
}

// CombineTxtToCSV combines all .txt files in the specified directory into a single CSV file
// with each column named after the source text file
func CombineTxtToCSV(fileMap map[string]string, outputPath string, csvName string) error {

	// Read data from each file
	fileData := make(map[string][]string)
	maxLines := 0

	for metric, fullPath := range fileMap {

		// Read file content
		content, err := os.Open(fullPath)
		if err != nil {
			return fmt.Errorf("Error opening file %s: %v", fullPath, err)
		}
		defer content.Close()

		// Read lines
		var lines []string
		scanner := bufio.NewScanner(content)
		for scanner.Scan() {
			lines = append(lines, scanner.Text())
		}

		if err := scanner.Err(); err != nil {
			return fmt.Errorf("error reading file %s: %v", fullPath, err)
		}

		fileData[metric] = lines

		// Track the maximum number of lines
		if len(lines) > maxLines {
			maxLines = len(lines)
		}
	}

	// Prepare CSV data with headers
	headers := make([]string, 0, len(fileData))
	for header := range fileData {
		headers = append(headers, header)
	}

	// Create output CSV file in the same directory
	outputFile := filepath.Join(outputPath, csvName)
	file, err := os.Create(outputFile)
	if err != nil {
		return fmt.Errorf("error creating output file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write headers
	if err := writer.Write(headers); err != nil {
		return fmt.Errorf("Error writing headers: %v", err)
	}

	// Write data rows
	for i := 0; i < maxLines; i++ {
		row := make([]string, len(headers))
		for j, header := range headers {
			if i < len(fileData[header]) {
				row[j] = fileData[header][i]
			} else {
				row[j] = "" // Fill with empty string if this column has fewer rows
			}
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("Error writing row: %v", err)
		}
	}

	return nil
}
