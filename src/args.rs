use clap::Parser;
pub use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "frinZ",
    version = "1.0",
    author = "Masanori AKIMOTO",
    about = "A Rust implementation of the frinZ fringe-fitting tool.",
    after_help = "(c) M.AKIMOTO with Gemini in 2025/08/04
This program is licensed under the MIT License
see https://opensource.org/license/mit"
)]
pub struct Args {
    /// Path to the input .cor file
    #[arg(long, aliases = ["in", "inp", "inpu"])]
    pub input: PathBuf,

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

    /// Add plot.
    #[arg(long, aliases = ["add", "add", "add-p", "add-pl", "add-plo"])]
    pub add_plot: bool,

    /// Output header information to console and file.
    #[arg(long)]
    pub header: bool,

    /// Enable precise search mode.
    #[arg(long)]
    pub search: bool,

    /// Number of iterations for precise search.
    #[arg(long, default_value_t = 3)]
    pub iter: i32,
}
