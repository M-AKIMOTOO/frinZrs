#![allow(unused_imports)]
use std::error::Error;
use std::process::exit;

use chrono::{DateTime, Utc};
use clap::{CommandFactory, Parser};
use num_complex::Complex;

mod acel_search;
mod analysis;
mod args;
mod bandpass;
mod deep_search;
//mod error;
mod fft;
mod fitting;
mod fringe_rate_map;
mod header;
mod imaging;
mod logo;
mod multisideband;
mod output;
mod phase_reference;
mod plot;
mod plot_msb;
mod processing;
mod raw_visibility;
mod read;
mod rfi;
mod single_file;
mod utils;
mod pre_check;

use crate::acel_search::run_acel_search_analysis;
use crate::args::Args;
use crate::fringe_rate_map::run_fringe_rate_map_analysis;
use crate::multisideband::run_multisideband_analysis;
use crate::phase_reference::run_phase_reference_analysis;
use crate::raw_visibility::run_raw_visibility_plot;
use crate::single_file::run_single_file_analysis;
use crate::pre_check::check_memory_usage;

// --- Type Aliases for Clarity ---
pub type C32 = Complex<f32>;

// --- Main Application Logic ---
fn main() -> Result<(), Box<dyn Error>> {
    let env_args: Vec<String> = std::env::args().collect();

    // Show logo if help is requested or no arguments are provided.
    if env_args.len() == 1 || env_args.iter().any(|arg| arg == "-h" || arg == "--help") {
        if let Err(e) = logo::show_logo() {
            // Log the error but continue execution, as the logo is not critical.
            eprintln!("Warning: Failed to display logo: {}", e);
        }
    }

    let mut args = match Args::try_parse() {
        Ok(args) => args,
        Err(e) => {
            if std::env::args().len() <= 1 {
                exit(0);
            } else {
                e.exit();
            }
        }
    };

    if !args.rate_padding.is_power_of_two() {
        eprintln!("Error: --rate-padding must be a power of two.");
        exit(1);
    }

    if args.raw_visibility {
        if args.input.is_none() {
            eprintln!("Error: --raw-visibility requires an --input file.");
            exit(1);
        }
        if let Err(e) = run_raw_visibility_plot(&args) {
            eprintln!("Error during raw visibility plotting: {}", e);
            exit(1);
        }
        return Ok(());
    }

    let flag_ranges: Vec<(DateTime<Utc>, DateTime<Utc>)> = args
        .flag_time
        .chunks_exact(2)
        .filter_map(|chunk| {
            let start = utils::parse_flag_time(&chunk[0]);
            let end = utils::parse_flag_time(&chunk[1]);
            match (start, end) {
                (Some(s), Some(e)) => {
                    if s >= e {
                        eprintln!(
                            "Error: Start time ({}) must be before end time ({}) for --flag-time.",
                            chunk[0], chunk[1]
                        );
                        exit(1);
                    }
                    Some((s, e))
                }
                _ => {
                    eprintln!(
                        "Error: Invalid time format in --flag-time arguments: '{}', '{}'. Expected YYYYDDDHHMMSS.",
                        chunk[0], chunk[1]
                    );
                    exit(1);
                }
            }
        })
        .collect();

    if args.fringe_rate_map {
        if let Some(input_path) = &args.input {
            if !check_memory_usage(&args, input_path)? {
                exit(0);
            }
        }
        if args.input.is_none() {
            eprintln!("Error: --fringe-rate-map requires an --input file.");
            exit(1);
        }
        return run_fringe_rate_map_analysis(&args, &flag_ranges);
    }

    if !args.multi_sideband.is_empty() {
        let c_band_path = std::path::PathBuf::from(&args.multi_sideband[0]);
        if !check_memory_usage(&args, &c_band_path)? {
            exit(0);
        }
        return run_multisideband_analysis(&args);
    }

    // Handle default for acel_search if provided without arguments
    if let Some(ref mut acel_search_vec) = args.acel_search {
        if acel_search_vec.is_empty() {
            *acel_search_vec = vec![2];
        }
    }

    // --- Argument Validation & Dispatch ---
    if args.acel_search.is_some() {
        if let Some(input_path) = &args.input {
            if !check_memory_usage(&args, input_path)? {
                exit(0);
            }
        }
        // Check if acel_search was provided by the user
        if args.input.is_none() {
            eprintln!("Error: --acel-search requires an --input file.");
            exit(1);
        }
        if args.length == 0 {
            eprintln!(
                "Warning: --acel-search is used, but --length is not specified. This is required for the analysis."
            );
            exit(1); // lengthが0の場合、分析に必須なのでexit(1)を追加
        }
        if args.loop_ == 1 {
            eprintln!("Warning: --acel-search is used, but --loop is not specified or is 1. Multiple loops are usually needed for fitting.");
        }
        return run_acel_search_analysis(
            &args,
            args.acel_search.as_ref().unwrap(),
            &flag_ranges,
        );
    }

    if args.input.is_some() && !args.phase_reference.is_empty() {
        eprintln!("Error: --input and --phase-reference cannot be used at the same time.");
        exit(1);
    }

    if !args.phase_reference.is_empty() {
        let cal_path = std::path::PathBuf::from(&args.phase_reference[0]);
        if !check_memory_usage(&args, &cal_path)? {
            exit(0);
        }
        let target_path = std::path::PathBuf::from(&args.phase_reference[1]);
        if !check_memory_usage(&args, &target_path)? {
            exit(0);
        }
        return run_phase_reference_analysis(&args, &flag_ranges);
    }

    if let Some(input_path) = &args.input {
        if !check_memory_usage(&args, input_path)? {
            exit(0);
        }
        return run_single_file_analysis(&args, &flag_ranges);
    }

    // If we reach here, no primary mode was selected.
    eprintln!("Error: Either --input or --phase-reference must be provided.");
    let mut cmd = Args::command();
    cmd.print_help().expect("Failed to print help");
    exit(1);
}
