use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "frinZ",
    version = "2.0",
    author = "Masanori AKIMOTO",
    about = "A Rust implementation of the frinZ fringe-fitting tool.",
    after_help = "(c) M.AKIMOTO with Gemini in 2025/08/04
This program is licensed under the MIT License
see https://opensource.org/license/mit"
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
    #[arg(long, aliases = ["fr", "fre", "freq", "frequ", "freque", "frequen", "frequenc"])]
    pub frequency: bool,

    /// Output the complex visibility data to a .tsv file.
    #[arg(long)]
    pub cross_output: bool,

    /// Output the analysis results (delay/frequency) to a .txt file.
    #[arg(long, aliases = ["ou", "out", "outp", "outpu"])]
    pub output: bool,

    /// Delay correction value.
    #[arg(long, aliases = ["delay"], default_value_t = 0.0, allow_negative_numbers = true)]
    pub delay_correct: f32,

    /// Rate correction value.
    #[arg(long, aliases = ["rate"], default_value_t = 0.0, allow_negative_numbers = true)]
    pub rate_correct: f32,

    /// Acceleration correction value.
    #[arg(long, aliases = ["acel"], default_value_t = 0.0, allow_negative_numbers = true)]
    pub acel_correct: f32,

    /// Delay window for fringe search (min, max).
    #[arg(long, aliases = ["delay-w", "delay-wi", "delay-win", "delay-wind", "delay-windo"], num_args = 2, value_name = "MIN MAX", allow_negative_numbers = true)]
    pub delay_window: Vec<f32>,

    /// Rate window for fringe search (min, max).
    #[arg(long, aliases = ["rate-w", "rate-wi", "rate-win", "rate-wind", "rate-windo"], num_args = 2, value_name = "MIN MAX", allow_negative_numbers = true)]
    pub rate_window: Vec<f32>,

    // /pub cmap_time: bool,

    /// Cumulate length in seconds.
    #[arg(long, aliases = ["cu", "cum", "cumu", "cumul", "cumula", "cumulat"], default_value_t = 0)]
    pub cumulate: i32,

    /// Generate additional plots for amplitude, SNR, phase, and noise level over time.
    /// These plots are useful for visualizing the time-series behavior of fringe parameters.
    /// Best used in conjunction with --length (to specify integration time per point)
    /// and --loop (to specify the number of points).
    #[arg(long, aliases = ["add", "add-p", "add-pl", "add-plo"])]
    pub add_plot: bool,

    /// Output header information to console and file.
    #[arg(long)]
    pub header: bool,

    /// Enable precise search mode.
    #[arg(long)]
    pub search: bool,

    /// Enable deep hierarchical search mode with fine-grained delay and rate search.
    #[arg(long)]
    pub search_deep: bool,

    /// Perform acceleration search with specified fitting degrees (e.g., --acel-search 2 1 1 2).
    /// 2 for quadratic fit, 1 for linear fit.
    ///
    /// Note: --length (must be > 0) and --loop (recommended > 1) are essential for this analysis.
    #[arg(long, num_args = 0.., value_name = "DEGREE")]
    pub acel_search: Option<Vec<i32>>,

    /// Number of iterations for the precise search mode (--search).
    #[arg(long, default_value_t = 1)]
    pub iter: u32,

    /// Generate dynamic spectrum plot.
    #[arg(long)]
    pub dynamic_spectrum: bool,

    /// Path to the bandpass calibration binary file made by --bandpass-table argument.
    #[arg(long)]
    pub bandpass: Option<PathBuf>,

    /// Output the bandpass-corrected complex spectrum to a binary file.
    #[arg(long)]
    pub bandpass_table: bool,

    /// Number of CPU cores to use for parallel processing. Only effective with `--search-deep`.
    /// If 0, automatically determines the number of cores.
    /// If specified value is greater than available CPU cores, it defaults to half of available cores.
    #[arg(long, default_value_t = 0)]
    pub cpu: u32,
    
    }
