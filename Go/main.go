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
	"regexp"
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

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

var (
	cpuprofile = flag.String("cpuprofile", "", "write CPU profile to file")
	memprofile = flag.String("memprofile", "", "write memory profile to file")
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	objectiveFile := flag.String("objective", "objective_scores.csv", "Path to the objective scores")
	flag.StringVar(objectiveFile, "o", "objective_scores.csv", "Path to the objective scores (short)")

	subjectiveFile := flag.String("subjective", "subjective_scores.csv", "Path to the subjective scores")
	flag.StringVar(subjectiveFile, "s", "subjective_scores.csv", "Path to the subjective scores (short)")

	nBootstrap := flag.Int("num_bootstraps", 14999, "Number of outer bootstrap samples")
	flag.IntVar(nBootstrap, "nb", 14999, "Number of outer bootstrap samples (short)")

	nInner := flag.Int("inner_bootstraps", 1499, "Number of inner bootstraps over votes")

	bootstrapScenes := flag.Bool("bootstrap_scenes", false, "Also bootstrap scenes (sample with replacement across stimuli)")
	flag.BoolVar(bootstrapScenes, "boostrap_scenes", false, "Legacy misspelled alias of -bootstrap_scenes")
	flag.BoolVar(bootstrapScenes, "b", false, "Also bootstrap scenes (short)")

	nVotes := flag.Int("votes", 15, "Number of votes per stimulus to synthesize when only mean/std is given")
	flag.IntVar(nVotes, "nv", 15, "Number of votes per stimulus (short)")

	lowerBound := flag.Float64("lower_bound", 0, "Lower bound of the rating scale")
	flag.Float64Var(lowerBound, "l", 0, "Lower bound of the rating scale (short)")

	upperBound := flag.Float64("upper_bound", 100, "Upper bound of the rating scale")
	flag.Float64Var(upperBound, "u", 100, "Upper bound of the rating scale (short)")

	pearsonCorr := flag.Bool("pearson", false, "Compute Pearson correlation")
	flag.BoolVar(pearsonCorr, "r", false, "Compute Pearson correlation (short)")

	spearmanCorr := flag.Bool("spearman", false, "Compute Spearman correlation")
	flag.BoolVar(spearmanCorr, "rho", false, "Compute Spearman correlation (short)")

	kendallCorr := flag.Bool("kendall", false, "Compute Kendall correlation")
	flag.BoolVar(kendallCorr, "tau", false, "Compute Kendall correlation (short)")

	saveTxt := flag.Bool("txt", false, "Save output correlation distributions as .txt files")
	saveCSV := flag.Bool("csv", false, "Save output correlation distributions as a combined .csv file")

	fileNamePrefix := flag.String("name", "", "Optional prefix for output files")
	outputPath := flag.String("output", "", "Output directory")

	seed := flag.Int64("seed", 0, "RNG seed (0 = time-based, nonzero = reproducible)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if len(os.Args) == 1 {
		flag.Usage()
		os.Exit(1)
	}
	if *lowerBound >= *upperBound {
		fmt.Fprintln(os.Stderr, "Error: lower bound must be smaller than upper bound.")
		os.Exit(1)
	}

	// Select coefficients.
	coeffIDs := selectedCoeffs(*pearsonCorr, *spearmanCorr, *kendallCorr)
	if len(coeffIDs) == 0 {
		fmt.Println("No correlation coefficient selected. Defaulting to Pearson.")
		coeffIDs = []coeffID{pearsonID}
	}

	// Profiling.
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			fatalf("could not create CPU profile: %v", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			fatalf("could not start CPU profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	// Resolve seed.
	baseSeed := uint64(*seed)
	if baseSeed == 0 {
		baseSeed = uint64(time.Now().UnixNano())
	}

	// --- Read subjective: auto-detect layout, materialize vote map ---
	subjectiveVotes, err := parseSubjectiveCSV(*subjectiveFile, *nVotes, *lowerBound, *upperBound, baseSeed)
	if err != nil {
		fatalf("subjective: %v", err)
	}

	// --- Read objective once; all metric columns at once ---
	objHeaders, objData, err := parseObjectiveCSV(*objectiveFile)
	if err != nil {
		fatalf("objective: %v", err)
	}
	if len(objHeaders) < 2 {
		fatalf("objective CSV has fewer than 2 columns")
	}

	// Pre-compute bootstrapped means for every scene once.
	bootstrappedSubjective := parallelBootstrap(subjectiveVotes, *nInner, baseSeed+1)

	fileMaps := map[coeffID]map[string]string{
		pearsonID: {}, spearmanID: {}, kendallID: {},
	}

	// Iterate metric columns (skip the first, which is the "name" column).
	for colIdx, metric := range objHeaders[1:] {
		objective := sliceMetricColumn(objData, colIdx)
		if len(objective) == 0 {
			fmt.Printf("Skipping %s (no valid values)\n", metric)
			continue
		}

		distributions := computeCorrelationDistributions(
			bootstrappedSubjective, objective,
			*nBootstrap, *bootstrapScenes, coeffIDs,
			baseSeed+uint64(colIdx)+2,
		)

		fmt.Printf("Results for %s\n", metric)
		for _, id := range coeffIDs {
			values := distributions[id]
			mean, lower, upper := bootstrapConfidenceInterval(values)
			fmt.Printf("%s: %.4f\n95%% CI: [%.4f, %.4f]\n\n", id.name(), mean, lower, upper)

			fullPath, err := buildOutputPath(*outputPath, *fileNamePrefix, metric, id.name())
			if err != nil {
				fatalf("%v", err)
			}
			fileMaps[id][metric] = fullPath

			if *saveTxt {
				if err := saveToText(fullPath, values); err != nil {
					fmt.Fprintln(os.Stderr, "could not save "+fullPath+":", err)
				}
			}
		}
	}

	if *saveCSV {
		prefix := ""
		if *fileNamePrefix != "" {
			prefix = *fileNamePrefix + "_"
		}
		for _, id := range []coeffID{pearsonID, spearmanID, kendallID} {
			if len(fileMaps[id]) == 0 {
				continue
			}
			if err := CombineTxtToCSV(fileMaps[id], *outputPath, prefix+id.name()+".csv"); err != nil {
				fmt.Fprintln(os.Stderr, "combine csv:", err)
			}
		}
	}

	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "could not create memory profile:", err)
			return
		}
		defer f.Close()
		_ = pprof.WriteHeapProfile(f)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "Error: "+format+"\n", args...)
	os.Exit(1)
}

// ---------------------------------------------------------------------------
// Coefficient enum (cheap dispatch, hoisted out of inner loop)
// ---------------------------------------------------------------------------

type coeffID int

const (
	pearsonID coeffID = iota
	spearmanID
	kendallID
)

func (c coeffID) name() string {
	switch c {
	case pearsonID:
		return "pearson"
	case spearmanID:
		return "spearman"
	case kendallID:
		return "kendall"
	}
	return "unknown"
}

func selectedCoeffs(p, s, k bool) []coeffID {
	ids := make([]coeffID, 0, 3)
	if p {
		ids = append(ids, pearsonID)
	}
	if s {
		ids = append(ids, spearmanID)
	}
	if k {
		ids = append(ids, kendallID)
	}
	return ids
}

// ---------------------------------------------------------------------------
// CSV parsing
// ---------------------------------------------------------------------------

// parseSubjectiveCSV auto-detects three layouts:
//  1. Mean/std: 3 columns with second header matching /mean|mos/i and third /std/i
//     → vote list synthesized via getSamples(mean, std, nVotes, bounds).
//  2. Bracketed raw: exactly 2 columns, second header containing "score"
//     AND first data cell starting with '[' → parse "[v1, v2, ...]".
//  3. Multi-column votes (default): each row's cols[1:] are individual votes;
//     empty / "NaN" / "nan" / unparseable cells are skipped.
func parseSubjectiveCSV(path string, nVotes int, lower, upper float64, seed uint64) (map[string][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1 // allow ragged rows
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) < 2 {
		return nil, errors.New("subjective CSV has no data rows")
	}

	headers := records[0]
	rows := records[1:]

	switch layout := detectSubjectiveLayout(headers, rows); layout {
	case layoutMeanStd:
		return expandMeanStd(rows, nVotes, lower, upper, seed)
	case layoutBracketedRaw:
		return parseBracketedRaw(rows)
	default:
		return parseMultiColumnVotes(rows)
	}
}

type subjectiveLayout int

const (
	layoutMultiColumn subjectiveLayout = iota
	layoutMeanStd
	layoutBracketedRaw
)

var (
	meanHeaderRE = regexp.MustCompile(`(?i)^(mean|mos)$`)
	stdHeaderRE  = regexp.MustCompile(`(?i)^(std|stddev|sd)$`)
)

func detectSubjectiveLayout(headers []string, rows [][]string) subjectiveLayout {
	if len(headers) == 3 &&
		meanHeaderRE.MatchString(strings.TrimSpace(headers[1])) &&
		stdHeaderRE.MatchString(strings.TrimSpace(headers[2])) {
		return layoutMeanStd
	}
	if len(headers) == 2 &&
		strings.Contains(strings.ToLower(headers[1]), "score") &&
		len(rows) > 0 && len(rows[0]) > 1 &&
		strings.HasPrefix(strings.TrimSpace(rows[0][1]), "[") {
		return layoutBracketedRaw
	}
	return layoutMultiColumn
}

func expandMeanStd(rows [][]string, nVotes int, lower, upper float64, seed uint64) (map[string][]float64, error) {
	out := make(map[string][]float64, len(rows))
	rng := rand.New(rand.NewSource(seed))
	for _, rec := range rows {
		if len(rec) < 3 {
			continue
		}
		name := rec[0]
		mean, err1 := strconv.ParseFloat(strings.TrimSpace(rec[1]), 64)
		std, err2 := strconv.ParseFloat(strings.TrimSpace(rec[2]), 64)
		if err1 != nil || err2 != nil {
			continue
		}
		out[name] = getSamples(rng, mean, std, nVotes, lower, upper)
	}
	return out, nil
}

func parseBracketedRaw(rows [][]string) (map[string][]float64, error) {
	out := make(map[string][]float64, len(rows))
	for _, rec := range rows {
		if len(rec) < 2 {
			continue
		}
		name := rec[0]
		inner := strings.Trim(strings.TrimSpace(rec[1]), "[]")
		parts := strings.Split(inner, ",")
		votes := make([]float64, 0, len(parts))
		for _, p := range parts {
			v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
			if err != nil {
				continue
			}
			votes = append(votes, v)
		}
		if len(votes) > 0 {
			out[name] = votes
		}
	}
	return out, nil
}

func parseMultiColumnVotes(rows [][]string) (map[string][]float64, error) {
	out := make(map[string][]float64, len(rows))
	skipped := 0
	for _, rec := range rows {
		if len(rec) < 2 {
			continue
		}
		name := rec[0]
		votes := make([]float64, 0, len(rec)-1)
		for _, cell := range rec[1:] {
			s := strings.TrimSpace(cell)
			if s == "" {
				continue
			}
			if strings.EqualFold(s, "nan") || strings.EqualFold(s, "null") || strings.EqualFold(s, "na") {
				continue
			}
			v, err := strconv.ParseFloat(s, 64)
			if err != nil || math.IsNaN(v) {
				continue
			}
			votes = append(votes, v)
		}
		if len(votes) == 0 {
			skipped++
			continue
		}
		out[name] = votes
	}
	if skipped > 0 {
		fmt.Fprintf(os.Stderr, "Warning: %d scene(s) had no valid votes and were skipped.\n", skipped)
	}
	return out, nil
}

// parseObjectiveCSV reads the file once, returning headers (including name)
// and a map of name → [metric_1, metric_2, ...] values aligned to headers[1:].
// Rows with any unparseable metric cell keep NaN for that slot; NaN-only scenes
// will be filtered per-metric in sliceMetricColumn.
func parseObjectiveCSV(path string) ([]string, map[string][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	records, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(records) < 2 {
		return nil, nil, errors.New("objective CSV has no data rows")
	}
	headers := records[0]
	nMetrics := len(headers) - 1

	out := make(map[string][]float64, len(records)-1)
	for _, rec := range records[1:] {
		if len(rec) == 0 {
			continue
		}
		name := rec[0]
		vec := make([]float64, nMetrics)
		for j := 0; j < nMetrics; j++ {
			if j+1 >= len(rec) {
				vec[j] = math.NaN()
				continue
			}
			v, err := strconv.ParseFloat(strings.TrimSpace(rec[j+1]), 64)
			if err != nil {
				vec[j] = math.NaN()
				continue
			}
			vec[j] = v
		}
		out[name] = vec
	}
	return headers, out, nil
}

// sliceMetricColumn projects one metric column from the objective table,
// dropping scenes whose value is NaN for that metric.
func sliceMetricColumn(data map[string][]float64, col int) map[string]float64 {
	out := make(map[string]float64, len(data))
	for name, vec := range data {
		if col >= len(vec) {
			continue
		}
		v := vec[col]
		if math.IsNaN(v) {
			continue
		}
		out[name] = v
	}
	return out
}

// ---------------------------------------------------------------------------
// Distribution sampling (beta with truncated-normal fallback)
// ---------------------------------------------------------------------------

func betaDistribution(rng *rand.Rand, mean, std, lower, upper float64, nSamples int) ([]float64, error) {
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

	rescaledMean := (mean - lower) / (upper - lower)
	rescaledVar := math.Pow(std/(upper-lower), 2)

	if rescaledVar < 1e-10 {
		samples := make([]float64, nSamples)
		for i := range samples {
			samples[i] = mean
		}
		return samples, nil
	}

	temp := (rescaledMean * (1 - rescaledMean) / rescaledVar) - 1
	alpha := rescaledMean * temp
	betaParam := (1 - rescaledMean) * temp

	if alpha <= 0 || betaParam <= 0 {
		samples := make([]float64, nSamples)
		for i := range samples {
			samples[i] = mean - 0.5
		}
		return samples, nil
	}

	betaDist := distuv.Beta{Alpha: alpha, Beta: betaParam, Src: rng}
	samples := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		samples[i] = lower + betaDist.Rand()*(upper-lower)
	}
	return samples, nil
}

var (
	stdNorm = distuv.Normal{Mu: 0, Sigma: 1}
	sqrt2Pi = math.Sqrt(2 * math.Pi)
)

func standardNormalPDF(x float64) float64 { return math.Exp(-0.5*x*x) / sqrt2Pi }
func standardNormalCDF(x float64) float64 { return stdNorm.CDF(x) }

func adjustTruncatedNormalParams(targetMean, targetStd, lower, upper float64, maxIterations int, epsilon float64) (float64, float64) {
	mu, sigma := targetMean, targetStd

	truncatedNormalMoments := func(mu, sigma, lower, upper float64) (float64, float64) {
		if sigma < epsilon {
			return mu, sigma
		}
		a := (lower - mu) / sigma
		b := (upper - mu) / sigma
		pdfA, pdfB := standardNormalPDF(a), standardNormalPDF(b)
		cdfA, cdfB := standardNormalCDF(a), standardNormalCDF(b)
		alpha := pdfA - pdfB
		beta := cdfB - cdfA
		if math.Abs(beta) < epsilon {
			return mu, sigma
		}
		truncatedMean := mu + (alpha/beta)*sigma
		varianceTerm := math.Max(0, 1+((a*pdfA-b*pdfB)/beta)-(alpha*alpha)/(beta*beta))
		truncatedVar := sigma * sigma * varianceTerm
		return truncatedMean, math.Sqrt(truncatedVar)
	}

	for i := 0; i < maxIterations; i++ {
		truncMean, truncStd := truncatedNormalMoments(mu, sigma, lower, upper)
		mu += targetMean - truncMean
		if truncStd < epsilon {
			fmt.Fprintf(os.Stderr, "Warning: truncStd very small (%.6f) at iter %d; stopping.\n", truncStd, i)
			break
		}
		sigma *= targetStd / truncStd
	}
	return mu, sigma
}

func truncatedNormalSamplesQuantile(rng *rand.Rand, mean, std, lower, upper float64, nSamples int) []float64 {
	norm := distuv.Normal{Mu: mean, Sigma: std}
	cdfLower := norm.CDF(lower)
	cdfUpper := norm.CDF(upper)
	samples := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		u := rng.Float64()*(cdfUpper-cdfLower) + cdfLower
		samples[i] = norm.Quantile(u)
	}
	return samples
}

func computeStats(samples []float64) (float64, float64) {
	n := float64(len(samples))
	if n == 0 {
		return 0, 0
	}
	sum := 0.0
	for _, v := range samples {
		sum += v
	}
	mean := sum / n
	varSum := 0.0
	for _, v := range samples {
		d := v - mean
		varSum += d * d
	}
	return mean, math.Sqrt(varSum / n)
}

// getSamples draws nSamples with beta distribution; if the fit is poor, adjusts
// parameters and tries a truncated normal; returns whichever matches the target
// mean/std more closely.
func getSamples(rng *rand.Rand, mean, std float64, nSamples int, lower, upper float64) []float64 {
	if std <= 0 {
		samples := make([]float64, nSamples)
		for i := range samples {
			samples[i] = mean
		}
		return samples
	}

	samples, _ := betaDistribution(rng, mean, std, lower, upper, nSamples)
	sampleMean, sampleStd := computeStats(samples)

	if math.Abs(sampleMean-mean) > 0.2 || math.Abs(sampleStd-std) > 0.1 {
		adjMean, adjStd := adjustTruncatedNormalParams(mean, std, lower, upper, 100, 1e-8)
		tn := truncatedNormalSamplesQuantile(rng, adjMean, adjStd, lower, upper, nSamples)
		tnMean, tnStd := computeStats(tn)
		betaErr := math.Abs(sampleMean-mean) + math.Abs(sampleStd-std)
		tnErr := math.Abs(tnMean-mean) + math.Abs(tnStd-std)
		if tnErr < betaErr {
			return tn
		}
	}
	return samples
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

func bootstrapVotes(rng *rand.Rand, votes []float64, nIterations int) []float64 {
	n := len(votes)
	if n == 0 {
		return nil
	}
	results := make([]float64, nIterations)
	invN := 1.0 / float64(n)
	for i := 0; i < nIterations; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			sum += votes[rng.Intn(n)]
		}
		results[i] = sum * invN
	}
	return results
}

// parallelBootstrap computes nIterations bootstrapped means per scene in parallel.
func parallelBootstrap(sceneVotes map[string][]float64, nIterations int, seed uint64) map[string][]float64 {
	keys := make([]string, 0, len(sceneVotes))
	for k := range sceneVotes {
		keys = append(keys, k)
	}
	sort.Strings(keys) // deterministic work order for reproducibility

	out := make(map[string][]float64, len(keys))
	var mu sync.Mutex

	numWorkers := runtime.NumCPU()
	if numWorkers > len(keys) {
		numWorkers = len(keys)
	}
	if numWorkers < 1 {
		numWorkers = 1
	}

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed ^ uint64(workerID+1)))
			for i := workerID; i < len(keys); i += numWorkers {
				key := keys[i]
				means := bootstrapVotes(rng, sceneVotes[key], nIterations)
				mu.Lock()
				out[key] = means
				mu.Unlock()
			}
		}(w)
	}
	wg.Wait()
	return out
}

// ---------------------------------------------------------------------------
// Correlation computation on bootstrap samples
// ---------------------------------------------------------------------------

// computeCorrelationDistributions performs nBootstrap iterations in parallel.
// Returns one slice of length nBootstrap per selected coefficient.
func computeCorrelationDistributions(
	subjective map[string][]float64,
	objective map[string]float64,
	nBootstrap int,
	bootstrapScenes bool,
	coeffs []coeffID,
	seed uint64,
) map[coeffID][]float64 {

	// Intersect keys between subjective and objective.
	keys := make([]string, 0, len(objective))
	for k := range subjective {
		if _, ok := objective[k]; ok {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)

	distributions := make(map[coeffID][]float64, len(coeffs))
	for _, c := range coeffs {
		distributions[c] = make([]float64, nBootstrap)
	}

	numWorkers := runtime.NumCPU()
	if numWorkers > nBootstrap {
		numWorkers = nBootstrap
	}
	if numWorkers < 1 {
		numWorkers = 1
	}

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	// Stride partition so iteration i is always handled by worker (i % numWorkers).
	// This keeps per-iteration RNG streams deterministic given a fixed seed.
	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed ^ uint64(workerID+1)))
			bootSubj := make([]float64, len(keys))
			objRate := make([]float64, len(keys))
			sampledKeys := make([]string, len(keys))
			for i := workerID; i < nBootstrap; i += numWorkers {
				runBootstrapIteration(rng, subjective, objective, keys, bootstrapScenes,
					coeffs, distributions, i, bootSubj, objRate, sampledKeys)
			}
		}(w)
	}
	wg.Wait()

	return distributions
}

// runBootstrapIteration executes a single outer bootstrap iteration, writing
// results directly into distributions[coeff][i] (workers own non-overlapping i).
func runBootstrapIteration(
	rng *rand.Rand,
	subjective map[string][]float64,
	objective map[string]float64,
	keys []string,
	bootstrapScenes bool,
	coeffs []coeffID,
	distributions map[coeffID][]float64,
	i int,
	bootSubj, objRate []float64,
	sampledKeys []string,
) {
	n := len(keys)

	var active []string
	if bootstrapScenes {
		for j := 0; j < n; j++ {
			sampledKeys[j] = keys[rng.Intn(n)]
		}
		active = sampledKeys
	} else {
		active = keys
	}

	for j, key := range active {
		ratings := subjective[key]
		bootSubj[j] = ratings[rng.Intn(len(ratings))] // single random bootstrapped mean per scene
		objRate[j] = objective[key]
	}

	for _, c := range coeffs {
		var v float64
		switch c {
		case pearsonID:
			v = pearson(bootSubj, objRate)
		case spearmanID:
			v = spearman(bootSubj, objRate)
		case kendallID:
			v = kendall(bootSubj, objRate)
		}
		distributions[c][i] = math.Abs(v)
	}
}

// ---------------------------------------------------------------------------
// Correlation coefficients
// ---------------------------------------------------------------------------

func pearson(x, y []float64) float64 { return stat.Correlation(x, y, nil) }

// rank assigns average ranks, handling ties.
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
		for j < n && pairs[j].value == pairs[i].value {
			j++
		}
		avgRank := float64(i+j+1) / 2.0
		for k := i; k < j; k++ {
			ranks[pairs[k].index] = avgRank
		}
		i = j
	}
	return ranks
}

func spearman(x, y []float64) float64 { return stat.Correlation(rank(x), rank(y), nil) }

// Kendall's tau-a (naive O(n²)).
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

// ---------------------------------------------------------------------------
// CI / output
// ---------------------------------------------------------------------------

func bootstrapConfidenceInterval(data []float64) (mean, lower, upper float64) {
	n := len(data)
	if n == 0 {
		return 0, 0, 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	mean = sum / float64(n)

	sorted := make([]float64, n)
	copy(sorted, data)
	sort.Float64s(sorted)

	li := int(0.025 * float64(n))
	ui := int(0.975 * float64(n))
	if li < 0 {
		li = 0
	}
	if ui >= n {
		ui = n - 1
	}
	return mean, sorted[li], sorted[ui]
}

func buildOutputPath(outputDir, prefix, metric, coeffName string) (string, error) {
	name := coeffName + ".txt"
	if prefix != "" {
		name = prefix + "_" + metric + "_" + name
	}
	if outputDir != "" {
		if err := ensureDirExists(outputDir); err != nil {
			return "", fmt.Errorf("ensure dir: %w", err)
		}
		return filepath.Join(outputDir, name), nil
	}
	return name, nil
}

func saveToText(filename string, data []float64) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	bw := bufio.NewWriter(f)
	defer bw.Flush()
	for _, v := range data {
		if _, err := fmt.Fprintf(bw, "%f\n", v); err != nil {
			return err
		}
	}
	return nil
}

func ensureDirExists(path string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}
	return os.MkdirAll(path, 0o755)
}

// CombineTxtToCSV merges per-metric .txt distribution files into one CSV,
// one column per metric. Columns may be of unequal length; missing cells are blank.
func CombineTxtToCSV(fileMap map[string]string, outputPath string, csvName string) error {
	// Stable header order: sorted by metric name.
	headers := make([]string, 0, len(fileMap))
	for h := range fileMap {
		headers = append(headers, h)
	}
	sort.Strings(headers)

	columns := make([][]string, len(headers))
	maxLines := 0
	for i, metric := range headers {
		lines, err := readLines(fileMap[metric])
		if err != nil {
			return fmt.Errorf("read %s: %w", fileMap[metric], err)
		}
		columns[i] = lines
		if len(lines) > maxLines {
			maxLines = len(lines)
		}
	}

	outputFile := filepath.Join(outputPath, csvName)
	f, err := os.Create(outputFile)
	if err != nil {
		return fmt.Errorf("create %s: %w", outputFile, err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	if err := w.Write(headers); err != nil {
		return fmt.Errorf("write header: %w", err)
	}

	row := make([]string, len(headers))
	for i := 0; i < maxLines; i++ {
		for j, col := range columns {
			if i < len(col) {
				row[j] = col[i]
			} else {
				row[j] = ""
			}
		}
		if err := w.Write(row); err != nil {
			return fmt.Errorf("write row: %w", err)
		}
	}
	return nil
}

func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	return lines, sc.Err()
}
