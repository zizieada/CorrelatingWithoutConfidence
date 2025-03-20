# Go Implementation (WIP)

This project is a Go implementation of the proposed algorithm.  It is currently a **work in progress (WIP)** and aims to provide a functional alternative to the provided Python implementation.

**Please note:** This Go version is under active development and may not be feature-complete or as stable as the Python script. Use with caution and expect changes.

## Features (Current)

* **Currently Implemented:**
  * The computation of confidence intervals for correlation coefficients given two input CSV files.
  * The ability to choose if scene bootstrapping is to be included or not.
  * The ability to save the output distributions as .txt files. This can be used to load the data into MATLAB, Python, etc. for further processing.
  * Cliff's delta and the win probability calculation.

## Installation

### Using Pre-built Binaries (Recommended for Quick Start)

Pre-built binaries for Linux, macOS, Windows are available in the `/bin` directory.

1. **Download the appropriate binary** for your operating system from the `/bin` directory.

2. **Make the binary executable** (if necessary, especially on Linux/macOS):
   
   ```bash
   chmod +x /path/to/your/downloaded/binary
   ```

### Build from source

```bash
   go build -o compute_correlation ./main.go
   go build -o cliffs_delta ./cliffs_delta.go
```

## Usage

```bash
./compute_correlation[.exe] -s /path/to/subjective/scores -o /path/to/objective/score
# for additional input arguments run just ./compute_correlation[.exe]
```

The output distributions of correlation coefficient can be saved as csvs and later processed with other tools (such as Python or MATLAB - see also the Python example code for more information about how to load and process the csvs further to create visualizations)

To output csvs, use

```bash
./compute_correlation[.exe] \
-s /path/to/subjective/scores \
-o /path/to/objective/score \
-txt -csv \
-output /path/to/an/output/folder \
-name[optional] dataset_name \
-r -rho -tau #[tags to compute all of the implemented correlation coefficients]
```

The input subjective scores have to have either three columns (name, mean, std) or two columns (name, scores), see the example data.

```bash
./cliffs_delta[.exe] -file1 /path/to/first/distribution -file2 /path/to/second/distribution
# for additional input arguments run just ./cliffs_delta[.exe]
```
