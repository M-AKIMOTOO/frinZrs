#![allow(unused_imports)]
use std::error::Error;
use std::process::exit;
use std::io::{self, Read, Write, Cursor};
use std::fs;
use byteorder::{WriteBytesExt, LittleEndian};

use chrono::{DateTime, Utc};
use clap::{CommandFactory, Parser};
use num_complex::Complex;
use std::path::Path;

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

    // シンプルな仕様: --cumulate が指定されたら rate_padding は常に 1 にする
    if args.cumulate != 0 {
        args.rate_padding = 1;
    }

    if !args.rate_padding.is_power_of_two() {
        eprintln!("Error: --rate-padding must be a power of two.");
        exit(1);
    }

    if args.cor2bin {
        if args.input.is_none() {
            eprintln!("Error: --cor2bin requires an --input file.");
            exit(1);
        }
        let input_path = args.input.as_ref().unwrap();

        // --- Create Output Directory ---
        let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
        let output_dir = parent_dir.join("frinZ").join("cor2bin");
        if let Err(e) = fs::create_dir_all(&output_dir) {
            eprintln!("Error creating output directory {:?}: {}", output_dir, e);
            exit(1);
        }
        let base_filename = input_path.file_stem().unwrap().to_str().unwrap();

        let mut file = match fs::File::open(input_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error opening input file {:?}: {}", input_path, e);
                exit(1);
            }
        };
        let mut buffer = Vec::new();
        if let Err(e) = file.read_to_end(&mut buffer) {
            eprintln!("Error reading input file {:?}: {}", input_path, e);
            exit(1);
        }
        let mut cursor = Cursor::new(buffer.as_slice());

        let header = match crate::header::parse_header(&mut cursor) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("Error parsing header: {}", e);
                exit(1);
            }
        };

        let mut all_spectra: Vec<Vec<C32>> = Vec::new();
        for l1 in 0..header.number_of_sector {
            let (complex_vec, _, _) = match crate::read::read_visibility_data(
                &mut cursor,
                &header,
                1, // length in sectors
                0, // skip in sectors
                l1, // loop_idx, which acts as sector index here
                false,
                &[], // pp_flag_ranges
            ) {
                Ok(data) => data,
                Err(_) => {
                    eprintln!("Warning: Could not read sector {}, stopping read.", l1);
                    break;
                }
            };
            if complex_vec.is_empty() {
                eprintln!("Warning: Empty sector {} found, stopping read.", l1);
                break;
            }
            all_spectra.push(complex_vec);
        }

        if all_spectra.is_empty() {
            eprintln!("No visibility data found in the file.");
            exit(1);
        }

        let flattened_spectra: Vec<C32> = all_spectra.iter().flatten().cloned().collect();
        let output_file_path = output_dir.join(format!("{}.cor.bin", base_filename));

        let mut output_file = match fs::File::create(&output_file_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error creating output file {:?}: {}", output_file_path, e);
                exit(1);
            }
        };

        if let Err(e) = output_file.write_f32::<LittleEndian>(header.fft_point as f32) {
            eprintln!("Error writing fft_point to file: {}", e);
            exit(1);
        }
        if let Err(e) = output_file.write_f32::<LittleEndian>(header.number_of_sector as f32) {
            eprintln!("Error writing number_of_sector to file: {}", e);
            exit(1);
        }

        for val in &flattened_spectra {
            if let Err(e) = output_file.write_f32::<LittleEndian>(val.re) {
                eprintln!("Error writing real part to file: {}", e);
                exit(1);
            }
            if let Err(e) = output_file.write_f32::<LittleEndian>(val.im) {
                eprintln!("Error writing imaginary part to file: {}", e);
                exit(1);
            }
        }
        println!("Raw complex visibility data written to {:?}.", output_file_path);
        println!("このバイナリファイルは以下のフォーマットで構成されています:");
        println!("- 先頭 4 byte: FFT点数 (f32, little-endian) = {}", header.fft_point);
        println!("- 次の 4 byte: セクター数(pp) (f32, little-endian) = {}", header.number_of_sector);
        println!("- それ以降: 複素スペクトルデータ (f32 real, f32 imag の繰り返し)");
        return Ok(());
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

    let mut time_flag_ranges: Vec<(DateTime<Utc>, DateTime<Utc>)> = Vec::new();
    let mut pp_flag_ranges: Vec<(u32, u32)> = Vec::new();

    if !args.flagging.is_empty() {
        let mode = &args.flagging[0];
        let params = &args.flagging[1..];

        match mode.as_str() {
            "time" => {
                if params.len() % 2 != 0 {
                    eprintln!("Error: --flagging time requires pairs of start and end times.");
                    exit(1);
                }
                time_flag_ranges = params
                    .chunks_exact(2)
                    .filter_map(|chunk| {
                        let start = utils::parse_flag_time(&chunk[0]);
                        let end = utils::parse_flag_time(&chunk[1]);
                        match (start, end) {
                            (Some(s), Some(e)) => {
                                if s >= e {
                                    eprintln!(
                                        "Error: Start time ({}) must be before end time ({}) for --flagging time.",
                                        chunk[0], chunk[1]
                                    );
                                    exit(1);
                                }
                                Some((s, e))
                            }
                            _ => {
                                eprintln!(
                                    "Error: Invalid time format in --flagging time: '{}, {}'. Expected YYYYDDDHHMMSS.",
                                    chunk[0], chunk[1]
                                );
                                exit(1);
                            }
                        }
                    })
                    .collect();
            }
            "pp" => {
                if params.len() % 2 != 0 {
                    eprintln!("Error: --flagging pp requires pairs of start and end sector numbers.");
                    exit(1);
                }
                pp_flag_ranges = params
                    .chunks_exact(2)
                    .filter_map(|chunk| {
                        let start_res = chunk[0].parse::<u32>();
                        let end_res = chunk[1].parse::<u32>();
                        match (start_res, end_res) {
                            (Ok(s), Ok(e)) => {
                                if s > e {
                                    eprintln!(
                                        "Error: Start pp ({}) must not be greater than end pp ({}) for --flagging pp.",
                                        s, e
                                    );
                                    exit(1);
                                }
                                Some((s, e))
                            }
                            _ => {
                                eprintln!(
                                    "Error: Invalid sector number in --flagging pp: '{}, {}'. Expected positive integers.",
                                    chunk[0], chunk[1]
                                );
                                exit(1);
                            }
                        }
                    })
                    .collect();
            }
            _ => {
                eprintln!("Error: Invalid mode for --flagging. Use 'time' or 'pp'.");
                exit(1);
            }
        }
    }

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
        return run_fringe_rate_map_analysis(&args, &time_flag_ranges, &pp_flag_ranges);
    }

    if !args.multi_sideband.is_empty() {
        let c_band_path = std::path::PathBuf::from(&args.multi_sideband[0]);
        if !check_memory_usage(&args, &c_band_path)? {
            exit(0);
        }
        return run_multisideband_analysis(&args);
    }

    // --- Argument Validation & Dispatch ---
    if let Some(search_mode) = &args.search {
        if search_mode == "acel" || search_mode == "rate" {
            if let Some(input_path) = &args.input {
                if !check_memory_usage(&args, input_path)? {
                    exit(0);
                }
            }
            if args.input.is_none() {
                eprintln!("Error: --search={} requires an --input file.", search_mode);
                exit(1);
            }
            if args.length == 0 {
                eprintln!(
                    "Warning: --search={} is used, but --length is not specified. This is required for the analysis.",
                    search_mode
                );
                exit(1);
            }
            if args.loop_ == 1 {
                eprintln!("Warning: --search={} is used, but --loop is not specified or is 1. Multiple loops are usually needed for fitting.", search_mode);
            }
            let degrees = if search_mode == "rate" { vec![1] } else { vec![2] };
            return run_acel_search_analysis(
                &args,
                &degrees,
                &time_flag_ranges,
                &pp_flag_ranges,
            );
        }
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
        return run_phase_reference_analysis(&args, &time_flag_ranges, &pp_flag_ranges);
    }

    if let Some(input_path) = &args.input {
        if !check_memory_usage(&args, input_path)? {
            exit(0);
        }
        return run_single_file_analysis(&args, &time_flag_ranges, &pp_flag_ranges);
    }

    // If we reach here, no primary mode was selected.
    eprintln!("Error: Either --input or --phase-reference must be provided.");
    let mut cmd = Args::command();
    cmd.print_help().expect("Failed to print help");
    exit(1);
}
