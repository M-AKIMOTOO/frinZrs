use clap::{ArgAction, Parser};
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "frinZ",
    version = env!("CARGO_PKG_VERSION"),
    author = "Masanori AKIMOTO  <masanori.akimoto.ac@gmail.com>",
    about = "fringe search for Yamaguchi Interferometer and Japanese VLBI Network",
    after_help = r#"(c) M.AKIMOTO with Gemini in 2025/08/04
github: https://github.com/M-AKIMOTOO/frinZrs
This program is licensed under the MIT License
see https://opensource.org/license/mit"#
)]
pub struct Args {
    /// Path to the input .cor file
    #[arg(long, aliases = ["in", "inp", "inpu"])]
    pub input: Option<PathBuf>,

    /// Paths to the calibrator and target .cor files for phase referencing.
    ///
    /// Arguments:
    /// 1. CALIBRATOR: Path to the calibrator .cor file.
    /// 2. TARGET: Path to the target .cor file.
    /// 3. FIT_DEGREE (optional): Degree of the polynomial for phase fitting. Defaults to 1.
    /// 4. CAL_LENGTH (optional): Integration time in seconds for the calibrator. Defaults to the entire file.
    /// 5. TGT_LENGTH (optional): Integration time in seconds for the target. Defaults to the entire file.
    /// 6. LOOP (optional): Number of times to loop the fringe processing for both files. Defaults to 1.
    #[arg(long, num_args = 2..=6, value_names = ["CALIBRATOR", "TARGET", "FIT_DEGREE", "CAL_LENGTH", "TGT_LENGTH", "LOOP"], aliases = ["ph", "pha", "phas", "phase","phase-r", "phase-re","phase-ref","phase-refe","phase-refer","phase-refere","phase-referen","phase-referenc"])]
    pub phase_reference: Vec<String>,

    /// The integration time in seconds. Defaults to the entire file.
    #[arg(long, aliases = ["le", "len", "leng", "lengt"], default_value_t = 0)]
    pub length: i32,

    /// The skip time in seconds from the start of the recording.
    #[arg(long, aliases = ["sk", "ski"], default_value_t = 0)]
    pub skip: i32,

    /// How many times to loop the fringe processing.
    #[arg(long, aliases = ["lo", "loo"], default_value_t = 1)]
    pub loop_: i32,

    /// RFI frequency ranges to exclude (e.g., "100,120" "400,500"). Can be specified multiple times.
    #[arg(long, num_args = 1.., value_name = "MIN,MAX")]
    pub rfi: Vec<String>,

    /// Generate and save plots of the fringe or spectrum.
    #[arg(long, aliases = ["pl", "plo"])]
    pub plot: bool,

    /// Calculate and display the cross-power spectrum instead of the fringe.
    #[arg(long, aliases = ["fre", "freq", "frequ", "freque", "frequen", "frequenc"])]
    pub frequency: bool,

    /// Output the raw complex visibility data to a binary file.
    #[arg(long, aliases = ["c2b", "cor2b", "cor2bi"])]
    pub cor2bin: bool,

    /// Output the frequency spectrum to a binary file.
    #[arg(long, aliases = ["spec", "spect", "spectr", "spectru"])]
    pub spectrum: bool,

    /// Output the analysis results (delay/frequency) to a .txt file.
    #[arg(long, aliases = ["ou", "out", "outp", "outpu"])]
    pub output: bool,

    /// Delay correction value.
    #[arg(long, aliases = ["delay","delay-corr"], default_value_t = 0.0, allow_negative_numbers = true)]
    pub delay_correct: f32,

    /// Rate correction value.
    #[arg(long, aliases = ["rate","rate-corr"], default_value_t = 0.0, allow_negative_numbers = true)]
    pub rate_correct: f32,

    /// Acceleration correction value.
    #[arg(long, aliases = ["acel","acel-corr"], default_value_t = 0.0, allow_negative_numbers = true)]
    pub acel_correct: f32,

    /// Correct delay and rate based on a scan table file.
    /// The file should contain comma-separated values of:
    /// start_time (YYYYDDDHHMMSS), integration_time (s), delay (samples), rate (Hz)
    #[arg(long, value_name = "FILE")]
    pub scan_correct: Option<PathBuf>,

    /// Delay window for fringe search (min, max).
    #[arg(long, aliases = ["delay-w", "delay-wi", "delay-win", "delay-wind", "delay-windo"], num_args = 2, value_name = "MIN MAX", allow_negative_numbers = true)]
    pub delay_window: Vec<f32>,

    /// Rate window for fringe search (min, max).
    #[arg(long, aliases = ["rate-w", "rate-wi", "rate-win", "rate-wind", "rate-windo"], num_args = 2, value_name = "MIN MAX", allow_negative_numbers = true)]
    pub rate_window: Vec<f32>,

    /// レート分解能のパディング係数（1,2,4,8,16 のみ指定可）。デフォルトは 1。
    /// なお、`--cumulate` が指定された場合は内部的に常に 1 に上書きされます。
    #[arg(long, aliases = ["rate-p", "rate-pa", "rate-pad", "rate-padd", "rate-paddi", "rate-paddin"], default_value_t = 1)]
    pub rate_padding: u32,

    /// Cumulate length in seconds.
    #[arg(long, aliases = ["cu", "cum", "cumu", "cumul", "cumula", "cumulat"], default_value_t = 0)]
    pub cumulate: i32,

    /// Generate additional plots for amplitude, SNR, phase, and noise level over time.
    #[arg(long, aliases = ["add", "add-p", "add-pl", "add-plo"])]
    pub add_plot: bool,

    /// Output header information to console and file.
    #[arg(long)]
    pub header: bool,

    /// FFT チャンネル数を平均化して縮小する目標 FFT 点数。
    /// 元の FFT 点数以上は指定できません。
    #[arg(long, value_name = "POINTS")]
    pub fft_rebin: Option<i32>,

    /// Specifies the search mode. [possible values: peak, deep, rate, acel]
    /// - peak: Precise search for the fringe peak using iterative fitting. (equivalent to the old --search flag).
    /// - deep: A deep, hierarchical search for fringes. Computationally expensive. (equivalent to the old --search-deep flag).
    /// - rate: A search for the fringe rate by performing a linear fit. (equivalent to the old --rate-search flag).
    /// - acel: A search for fringe acceleration by performing a quadratic fit.
    ///
    /// If `--search` is provided without a value, it defaults to `peak`.
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "peak",
        value_name = "MODE",
        value_parser = ["peak", "deep", "rate", "acel"],
        action = ArgAction::Append
    )]
    pub search: Vec<String>,

    /// Number of iterations for the precise search mode (--search=peak).
    #[arg(long, default_value_t = 5)]
    pub iter: u32,

    /// Generate dynamic spectrum plot.
    #[arg(long, aliases = ["ds","dynamic"])]
    pub dynamic_spectrum: bool,

    /// Path to the bandpass calibration binary file made by --bandpass-table argument.
    #[arg(long, aliases = ["bp"])]
    pub bandpass: Option<PathBuf>,

    /// Output the bandpass-corrected complex spectrum to a binary file.
    #[arg(long, aliases = ["bptable"])]
    pub bandpass_table: bool,

    /// Number of CPU cores to use for parallel processing. Only effective with `--search=deep`.
    /// If 0, automatically determines the number of cores.
    /// If specified value is greater than available CPU cores, it defaults to half of available cores.
    #[arg(long, default_value_t = 0)]
    pub cpu: u32,

    /// Flag data by time or sector number (pp).
    /// Modes:
    ///  time <START> <END>... : Skips processing for segments within the YYYYDDDHHMMSS time ranges.
    ///  pp <START> <END>...   : Replaces visibility data with 0+0j for the given sector number ranges.
    #[arg(long, num_args = 1.., value_name = "MODE [ARGS...]", aliases = ["flag"])]
    pub flagging: Vec<String>,

    /// Calculate and plot the Allan deviation of the phase data.
    /// Requires --length and --loop to be set to generate a time series.
    #[arg(long, aliases = ["allan","allan-dev"])]
    pub allan_deviance: bool,

    /// Generate heatmaps of raw visibility data (amplitude and phase).
    /// Requires --input. The program will exit after plotting.
    #[arg(long, aliases = ["ra","raw","raw-v","raw-vi","raw-vis","raw-visi","raw-visib","raw-visibi","raw-visibils","raw-visibili","raw-visibilit"])]
    pub raw_visibility: bool,

    /// Generate UV coverage plot; optional value (0 = planar, 1 = 3D). Defaults to 1.
    #[arg(long, num_args = 0..=1, default_missing_value = "1")]
    pub uv: Option<i32>,

    #[arg(long, aliases = ["frmap"], num_args = 0.., value_name = "KEY[:VALUE]")]
    pub fringe_rate_map: Option<Vec<String>>,

    /// Perform maser analysis.
    /// Accepts key-value pairs such as:
    ///   off:<path>     -- Off-source .cor file (required)
    ///   rest:<MHz>     -- Rest frequency override (defaults to 6668.5192)
    ///   Vlst:<km/s>    -- Override LSR velocity correction
    ///   corrfreq:<x>   -- Multiplier applied to the sampling frequency
    ///   band:<start-end> -- Frequency window offsets in MHz relative to observing frequency
    ///   subt:<start-end> -- Absolute frequency window in MHz (overrides band)
    ///   onoff:<0|1>   -- Use (ON-OFF)/OFF when 0 (default), or (ON-OFF) when 1
    ///   gauss:amp,Vlst,fwhm,[amp,Vlst,fwhm...] -- Apply Gaussian mixture fits on the velocity spectrum
    ///
    /// Positional arguments (legacy): first token = off-source path, second = rest frequency.
    #[arg(long, num_args = 1.., value_name = "KEY:VALUE")]
    pub maser: Vec<String>,

    /// Perform multi-sideband analysis.
    /// Arguments:
    /// 1. C_BAND_DATA: Path to the C-band .cor file.
    /// 2. C_BAND_BP: Path to the C-band bandpass file. Use -1 to disable.
    /// 3. C_BAND_DELAY: Delay for C-band in seconds.
    /// 4. X_BAND_DATA: Path to the X-band .cor file.
    /// 5. X_BAND_BP: Path to the X-band bandpass file. Use -1 to disable.
    /// 6. X_BAND_DELAY: Delay for X-band in seconds.
    #[arg(long, num_args = 6, value_names = ["C_BAND_DATA", "C_BAND_BP", "C_BAND_DELAY", "X_BAND_DATA", "X_BAND_BP", "X_BAND_DELAY"], aliases = ["msb"], allow_negative_numbers = true)]
    pub multi_sideband: Vec<String>,

    /// Plot antenna uptime (Az/El) over UT.
    #[arg(long)]
    pub uptimeplot: bool,

    /// Run Earth-rotation imaging on the input visibility data.
    /// Sub-options (key[:value]):
    ///   size:<pixels>        Image size per axis (default 256)
    ///   cell:<arcsec>        Cell size in arcsec/pixel (auto if omitted)
    ///   clean[:0|1]          Enable CLEAN (default 0 / disabled)
    ///   gain:<value>         CLEAN loop gain (default 0.1)
    ///   threshold:<Jy>       CLEAN stopping threshold Jy (default 0.01)
    ///   iter:<count>         CLEAN max iterations (default 200)
    #[arg(long, num_args = 0.., value_name = "KEY[:VALUE]", requires = "input")]
    pub imaging: Option<Vec<String>>,

    /// Perform a test of the Earth-rotation synthesis imaging module.
    #[arg(long)]
    pub imaging_test: bool,
}

impl Args {
    pub fn primary_search_mode(&self) -> Option<&str> {
        self.search
            .iter()
            .find(|mode| *mode == "peak" || *mode == "deep")
            .map(|s| s.as_str())
    }
}
