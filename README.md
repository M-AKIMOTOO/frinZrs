<img src="./src/frinZmain/logo1.png" width=45%>  <img src="./src/frinZmain/logo2.png" width=45%>

# frinZrs

Rust version of frinZ.py - A high-performance fringe-fitting tool for VLBI data analysis.  
Original Python version: https://github.com/M-AKIMOTOO/frinZ.py

## Note
fontconfig-devel needs to be installed for this program.

## Overview

frinZrs is a Rust implementation of the frinZ fringe-fitting tool for processing Very Long Baseline Interferometry (VLBI) correlation data. It provides accurate delay and rate measurements with enhanced performance compared to the original Python version.

## Features

- **Fringe fitting analysis** for VLBI correlation data (.cor files)
- **Phase reference calibration** with polynomial fitting
- **Precise search mode** with iterative refinement
- **RFI mitigation** with frequency range exclusion
- **Bandpass calibration** with binary file support
- **Visualization** with delay/rate plots and cumulative SNR plots
- **Multiple output formats** (text, binary, plots)
- **Cross-power spectrum analysis**

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/M-AKIMOTOO/frinZrs.git
cd frinZrs

# Install to ~/.cargo/bin
cargo install --bin frinZ --path .
# OR
cargo install --bin frinZrs --path .
```

### Development Build

```bash
# Run directly
cargo run --bin frinZ --release -- [OPTIONS]
# OR
cargo run --bin frinZ-rs --release -- [OPTIONS]
```

**Note:** Both `frinZ` and `frinZ-rs` are identical programs. On Windows, antivirus software may flag the compiled binary.

## Usage

### Basic Syntax

```bash
frinZ [OPTIONS]
```

### Input Options

#### Single File Analysis
```bash
# Basic fringe fitting
frinZ --input data.cor

# With integration time and loop count
frinZ --input data.cor --length 30 --loop 5

# Skip first 10 seconds
frinZ --input data.cor --skip 10
```

#### Phase Reference Analysis
```bash
# Basic phase referencing (calibrator, target)
frinZ --phase-reference cal.cor target.cor

# With polynomial degree and custom integration times
frinZ --phase-reference cal.cor target.cor 2 60 120 3
# Arguments: calibrator target fit_degree cal_length target_length loop
```

### Analysis Options

#### Frequency Domain Analysis
```bash
# Cross-power spectrum instead of fringe
frinZ --input data.cor --frequency
```

#### Precise Search Mode
```bash
# Enable iterative search with custom iterations
frinZ --input data.cor --search --iter 5

# With search windows
frinZ --input data.cor --search --delay-window -10 10 --rate-window -0.1 0.1
```

#### Manual Corrections
```bash
# Apply delay and rate corrections
frinZ --input data.cor --delay-correct 5.2 --rate-correct -0.03
```

### RFI Mitigation

```bash
# Exclude frequency ranges (MHz)
frinZ --input data.cor --rfi "100,120" "400,500"
```

### Bandpass Calibration

```bash
# Generate bandpass table
frinZ --input cal.cor --bandpass-table

# Apply existing bandpass calibration
frinZ --input data.cor --bandpass /path/to/bandpass_table.bin
```

### Output Options

#### Text Output
```bash
# Save analysis results to text files
frinZ --input data.cor --output

# Show header information
frinZ --input data.cor --header
```

#### Plotting
```bash
# Generate fringe plots
frinZ --input data.cor --plot

# Time series plots
frinZ --input data.cor --add-plot

# Cumulative SNR plots
frinZ --input data.cor --cumulate 10
```

### Advanced Examples

#### Complete Analysis with All Options
```bash
frinZ --input data.cor \
  --length 60 --loop 10 --skip 5 \
  --search --iter 3 \
  --delay-window -20 20 --rate-window -0.05 0.05 \
  --rfi "150,200" \
  --plot --add-plot --output \
  --bandpass cal_bandpass.bin
```

#### Phase Reference with Custom Parameters
```bash
frinZ --phase-reference cal.cor target.cor 1 30 60 5 \
  --search --plot --output
```

## Output Files

frinZrs creates organized output directories:

```
frinZ/
├── fringe_graph/          # Delay/rate plots
│   ├── time_domain/
│   └── freq_domain/
├── fringe_output/         # Text analysis results
├── add_plot/             # Time series plots
├── cumulate/             # Cumulative SNR plots
├── bandpass_table/       # Bandpass calibration files
├── phase_reference/      # Phase reference outputs
└── cor_header/           # Header information
```

### Output File Formats

- **Text files (`.txt`)**: Analysis results with delay, rate, SNR, and statistics
- **Binary files (`.bin`, `.cor`)**: Complex spectra and calibrated data
- **Plot files (`.png`)**: Visualization of fringe patterns and time series

## File Naming Convention

Output files follow the pattern:
```
{station1}_{station2}_{timestamp}_{source}_{band}_len{length}s[_rfi][_bp]
```

Example: `YAMAGU32_YAMAGU34_2025001120000_3C84_x_len60s_rfi`

## Command Reference

### Required Arguments (one of)
- `--input <FILE>`: Single .cor file for analysis
- `--phase-reference <CAL> <TARGET> [OPTIONS]`: Phase referencing mode

### Time Parameters
- `--length <SECONDS>`: Integration time (default: entire file)
- `--skip <SECONDS>`: Skip time from start (default: 0)
- `--loop <COUNT>`: Number of processing loops (default: 1)
- `--cumulate <SECONDS>`: Cumulative integration length

### Search Parameters
- `--search`: Enable precise search mode
- `--iter <COUNT>`: Search iterations (default: 3)
- `--delay-window <MIN MAX>`: Delay search range (samples)
- `--rate-window <MIN MAX>`: Rate search range (Hz)
- `--delay-correct <VALUE>`: Manual delay correction (samples)
- `--rate-correct <VALUE>`: Manual rate correction (Hz)

### Analysis Options
- `--frequency`: Frequency domain analysis
- `--rfi <"MIN,MAX">`: RFI frequency ranges to exclude (MHz)
- `--bandpass <FILE>`: Apply bandpass calibration
- `--bandpass-table`: Generate bandpass table

### Output Options
- `--output`: Save text results
- `--header`: Show header information
- `--plot`: Generate fringe plots
- `--add-plot`: Generate time series plots
- `--cross-output`: Output complex visibility data
- `--dynamic-spectrum`: Generate dynamic spectrum plots

## Performance Notes

frinZrs provides significant performance improvements over the Python version:
- **Faster FFT processing** using rustfft
- **Optimized memory usage** for large datasets
- **Parallel processing** capabilities
- **Accuracy comparable** to original (within 0.1%)

The minor numerical differences (≤0.1%) compared to frinZ.py arise from:
- Different FFT library implementations (rustfft vs scipy.fft)
- Precision differences in binary decoding (Rust: 7-8 digits, Python: 6 digits)
- DC component handling in FFT processing

## License

This program is licensed under the MIT License.

## Author

(c) M.AKIMOTO with Gemini in 2025/08/04

## Related Projects

- [frinZ.py](https://github.com/M-AKIMOTOO/frinZ.py) - Original Python implementation
