#![allow(unused_imports)]
use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write, BufWriter};
use std::path::{Path, PathBuf};
use std::process;

use chrono::{DateTime, Utc};
use clap::{CommandFactory, Parser};
use image::ImageBuffer;
use imageproc::filter;
use num_complex::Complex;
use ndarray::Array;

mod args;
mod analysis;
mod bandpass;
mod deep_search;
mod error;
mod fft;
mod fitting;
mod header;
mod logo;
mod output;
mod plot;
mod read;
mod rfi;
mod utils;

use crate::utils::unwrap_phase_radians;
use args::Args;
use analysis::analyze_results;
use bandpass::{apply_bandpass_correction, read_bandpass_file, write_complex_spectrum_binary};
use fft::{apply_phase_correction, process_fft, process_ifft};
use header::{parse_header, CorHeader};
use output::{
    format_delay_output,
    format_freq_output,
    generate_output_names,
    output_header_info,
    write_phase_corrected_spectrum_binary,
};
use plot::{add_plot, cumulate_plot, delay_plane, frequency_plane, phase_reference_plot};
use read::{read_sector_header, read_visibility_data};
use rfi::parse_rfi_ranges;

// --- Type Aliases for Clarity ---
type C32 = Complex<f32>;

/// Holds the results of processing a single .cor file, needed for subsequent plotting.
struct ProcessResult {
    header: CorHeader,
    label: Vec<String>,
    obs_time: chrono::DateTime<Utc>,
    length_arg: i32,
    cumulate_len: Vec<f32>,
    cumulate_snr: Vec<f32>,
    add_plot_times: Vec<DateTime<Utc>>,
    add_plot_amp: Vec<f32>,
    add_plot_snr: Vec<f32>,
    add_plot_phase: Vec<f32>,
    add_plot_noise: Vec<f32>,
}

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
                std::process::exit(0);
            } else {
                e.exit();
            }
        }
    };

    // Handle default for acel_search if provided without arguments
    if let Some(ref mut acel_search_vec) = args.acel_search {
        if acel_search_vec.is_empty() {
            *acel_search_vec = vec![2];
        }
    }

    

    // --- Argument Validation & Dispatch ---
    if args.acel_search.is_some() { // Check if acel_search was provided by the user
        if args.input.is_none() {
            eprintln!("Error: --acel-search requires an --input file.");
            process::exit(1);
        }
        if args.length == 0 {
            eprintln!(
                "Warning: --acel-search is used, but --length is not specified. This is required for the analysis."
            );
            process::exit(1); // lengthが0の場合、分析に必須なのでexit(1)を追加
        }
        if args.loop_ == 1 {
            eprintln!("Warning: --acel-search is used, but --loop is not specified or is 1. Multiple loops are usually needed for fitting.");
        }
        return run_acel_search_analysis(&args, args.acel_search.as_ref().unwrap());
    }

    if args.input.is_some() && !args.phase_reference.is_empty() {
        eprintln!("Error: --input and --phase-reference cannot be used at the same time.");
        process::exit(1);
    }

    if !args.phase_reference.is_empty() {
        return run_phase_reference_analysis(&args);
    }

    if let Some(_) = &args.input {
        return run_single_file_analysis(&args);
    }

    // If we reach here, no primary mode was selected.
    eprintln!("Error: Either --input or --phase-reference must be provided.");
    let mut cmd = Args::command();
    cmd.print_help().expect("Failed to print help");
    process::exit(1);
}

/// Executes the analysis for a single input file.
fn run_single_file_analysis(args: &Args) -> Result<(), Box<dyn Error>> {
    let input_path = args.input.as_ref().unwrap();
    let result = process_cor_file(input_path, args)?;

    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let frinz_dir = parent_dir.join("frinZ");

    if args.cumulate != 0 {
        let path = frinz_dir.join(format!("cumulate/len{}s", args.cumulate));
        cumulate_plot(
            &result.cumulate_len,
            &result.cumulate_snr,
            &path,
            &result.header,
            &result.label.iter().map(|s| s.as_str()).collect::<Vec<&str>>(),
            &result.obs_time,
            args.cumulate,
        )?;
    }

    if args.add_plot {
        let path = frinz_dir.join("add_plot");
        let base_filename = generate_output_names(
            &result.header,
            &result.obs_time,
            &result.label.iter().map(|s| s.as_str()).collect::<Vec<&str>>(),
            !args.rfi.is_empty(),
            args.frequency,
            args.bandpass.is_some(),
            result.length_arg,
        );
        let add_plot_filename = format!("{}_{}", base_filename, result.header.source_name);
        let add_plot_filepath = path.join(add_plot_filename);

        if !result.add_plot_times.is_empty() {
            let first_time = result.add_plot_times[0];
            let elapsed_times_f32: Vec<f32> = result
                .add_plot_times
                .iter()
                .map(|dt| (*dt - first_time).num_seconds() as f32)
                .collect();

            add_plot(
                add_plot_filepath.to_str().unwrap(),
                &elapsed_times_f32, // Use elapsed time
                &result.add_plot_amp,
                &result.add_plot_snr,
                &result.add_plot_phase,
                &result.add_plot_noise,
                &result.header.source_name,
                result.length_arg,
                &result.obs_time, // Added this line
            )?;

            output::write_add_plot_data_to_file(
                &path,
                &base_filename,
                &elapsed_times_f32,
                &result.add_plot_amp,
                &result.add_plot_snr,
                &result.add_plot_phase,
                &result.add_plot_noise,
            )?;
        }
    }

    Ok(())
}

/// Executes the phase reference analysis for a calibrator and a target file.
fn run_phase_reference_analysis(args: &Args) -> Result<(), Box<dyn Error>> {
    let cal_path = PathBuf::from(&args.phase_reference[0]);
    let target_path = PathBuf::from(&args.phase_reference[1]);

    // --- Parse phase_reference arguments ---
    let fit_degree: i32 = if args.phase_reference.len() > 2 {
        args.phase_reference[2].parse().unwrap_or(1)
    } else {
        1
    };
    let cal_length: i32 = if args.phase_reference.len() > 3 {
        args.phase_reference[3].parse().unwrap_or(0)
    } else {
        args.length // Default to global length or 0
    };
    let target_length: i32 = if args.phase_reference.len() > 4 {
        args.phase_reference[4].parse().unwrap_or(0)
    } else {
        args.length // Default to global length or 0
    };
    let loop_count: i32 = if args.phase_reference.len() > 5 {
        args.phase_reference[5].parse().unwrap_or(1)
    } else {
        args.loop_ // Default to global loop
    };

    // --- Create specific Args for calibrator and target ---
    let mut cal_args = args.clone();
    cal_args.length = cal_length;
    cal_args.loop_ = loop_count;
    cal_args.input = Some(cal_path.clone());

    let mut target_args = args.clone();
    target_args.length = target_length;
    target_args.loop_ = loop_count;
    target_args.input = Some(target_path.clone());

    println!("Running phase reference analysis...");
    println!(
        "Calibrator: {:?} (length: {}s, loop: {})",
        &cal_path,
        if cal_length == 0 {
            "all".to_string()
        } else {
            cal_length.to_string()
        },
        loop_count
    );
    let mut cal_results = process_cor_file(&cal_path, &cal_args)?;

    println!(
        "Target:     {:?} (length: {}s, loop: {}\n",
        &target_path,
        if target_length == 0 {
            "all".to_string()
        } else {
            target_length.to_string()
        },
        loop_count
    );
    let mut target_results = process_cor_file(&target_path, &target_args)?;

    // --- Phase Unwrapping ---
    utils::unwrap_phase(&mut cal_results.add_plot_phase);
    utils::unwrap_phase(&mut target_results.add_plot_phase);

    // Store original calibrator phases before fitting
    let original_cal_phases = cal_results.add_plot_phase.clone();
    // Store original target phases before fitting
    let original_target_phases = target_results.add_plot_phase.clone();

    let mut fitted_cal_phases: Vec<f32> = Vec::new(); // To store the fitted curve for calibrator

    // --- Phase Fitting ---
    if fit_degree < 0 {
        eprintln!("Error: Polynomial degree must be non-negative.");
        return Err("Invalid polynomial degree".into());
    }

    let min_data_points = (fit_degree + 1) as usize;
    if cal_results.add_plot_times.is_empty() {
        eprintln!("Error: Calibrator data is empty, cannot proceed with phase fitting.");
        return Err("Empty calibrator data".into());
    }
    let first_time = cal_results.add_plot_times[0];
    if cal_results.add_plot_times.len() < min_data_points {
        eprintln!(
            "Warning: Not enough data points ({}) for polynomial fitting of degree {} on calibrator. Need at least {} points. Proceeding without phase fit.",
            cal_results.add_plot_times.len(),
            fit_degree,
            min_data_points
        );
    } else {
        let cal_times_f64: Vec<f64> = cal_results
            .add_plot_times
            .iter()
            .map(|t| t.signed_duration_since(first_time).num_milliseconds() as f64 / 1000.0)
            .collect();
        let cal_phases_f64: Vec<f64> = cal_results.add_plot_phase.iter().map(|&p| p as f64).collect();

        match fitting::fit_polynomial_least_squares(&cal_times_f64, &cal_phases_f64, fit_degree as usize) {
            Ok(coeffs) => {
                println!(
                    "Polynomial fit (degree {}) to calibrator phase. Coefficients: {:?}",
                    fit_degree,
                    coeffs
                );

                // Helper function to evaluate polynomial
                let evaluate_polynomial = |x: f64, coeffs: &[f64]| -> f64 {
                    coeffs.iter().enumerate().map(|(i, &c)| c * x.powi(i as i32)).sum()
                };

                // Calculate fitted_cal_phases
                fitted_cal_phases = cal_times_f64
                    .iter()
                    .map(|&t| evaluate_polynomial(t, &coeffs) as f32)
                    .collect();

                // Subtract from calibrator
                for (i, t) in cal_times_f64.iter().enumerate() {
                    let fitted_val = evaluate_polynomial(*t, &coeffs);
                    cal_results.add_plot_phase[i] -= fitted_val as f32;
                }

                // Subtract from target
                if !target_results.add_plot_times.is_empty() {
                    let target_times_f64: Vec<f64> = target_results
                        .add_plot_times
                        .iter()
                        .map(|t| t.signed_duration_since(first_time).num_milliseconds() as f64 / 1000.0)
                        .collect();
                    for (i, t) in target_times_f64.iter().enumerate() {
                        let fitted_val = evaluate_polynomial(*t, &coeffs);
                        target_results.add_plot_phase[i] -= fitted_val as f32;
                    }
                }

                // --- Apply phase correction to target and write to binary file ---
                println!(
                    "\nApplying phase correction to target file and writing to binary output..."
                );

                let mut target_file = File::open(&target_path)?;
                let mut target_buffer = Vec::new();
                target_file.read_to_end(&mut target_buffer)?;

                let mut file_header = vec![0u8; 256];
                let mut cursor = Cursor::new(target_buffer.as_slice());
                cursor.read_exact(&mut file_header)?;

                let mut calibrated_spectra: Vec<Vec<C32>> = Vec::new();
                let mut sector_headers_raw: Vec<Vec<u8>> = Vec::new();

                let num_sectors = target_results.header.number_of_sector;
                for l1 in 0..num_sectors {
                    let (complex_vec, current_obs_time, _effective_integ_time) = read_visibility_data(
                        &mut Cursor::new(target_buffer.as_slice()),
                        &target_results.header,
                        1,
                        l1,
                        0,
                        false,
                    )?;

                    let sector_headers = read_sector_header(
                        &mut Cursor::new(target_buffer.as_slice()),
                        &target_results.header,
                        1,
                        l1,
                        0,
                        false,
                    )?;
                    sector_headers_raw.push(sector_headers[0].clone());

                    let time_since_start_sec =
                        current_obs_time.signed_duration_since(first_time).num_milliseconds() as f64
                            / 1000.0;
                    let phase_correction_deg = evaluate_polynomial(time_since_start_sec, &coeffs);
                    let phase_correction_rad = (phase_correction_deg as f32).to_radians();

                    let phase_rotation = Complex::new(0.0, -phase_correction_rad).exp();
                    let calibrated_spectrum: Vec<C32> =
                        complex_vec.iter().map(|c| *c * phase_rotation).collect();
                    calibrated_spectra.push(calibrated_spectrum);
                }

                let target_basename = target_path.file_stem().unwrap().to_str().unwrap();
                let parts: Vec<&str> = target_basename.split('_').collect();
                if parts.len() >= 3 {
                    let new_basename = parts[..3].join("_");
                    let output_filename_str = format!("{}_phsref.cor", new_basename);
                    let phase_reference_dir = target_path
                        .parent()
                        .unwrap_or_else(|| Path::new(""))
                        .join("frinZ")
                        .join("phase_reference");
                    fs::create_dir_all(&phase_reference_dir)?;
                    let output_path = phase_reference_dir.join(output_filename_str);

                    write_phase_corrected_spectrum_binary(
                        &output_path,
                        &file_header,
                        &sector_headers_raw,
                        &calibrated_spectra,
                    )?;
                    println!("Successfully wrote phase-calibrated data to: {:?}", output_path);
                } else {
                    eprintln!("Warning: Could not generate output filename for calibrated data due to unexpected format of target filename.");
                }
            }
            Err(e) => {
                eprintln!("Warning: Polynomial phase fitting failed: {}", e);
            }
        }
    }

    // --- Plotting ---
    let plot_dir = target_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("frinZ")
        .join("phase_reference");
    fs::create_dir_all(&plot_dir)?;

    let target_basename = target_path.file_stem().unwrap().to_str().unwrap();
    let parts: Vec<&str> = target_basename.split('_').collect();
    let output_basename = if parts.len() >= 3 {
        parts[..3].join("_")
    } else {
        // Fallback for unexpected filename format
        format!(
            "phsref_{}_{}",
            cal_path.file_stem().unwrap().to_str().unwrap(),
            target_basename
        )
    };
    let plot_filename = format!("{}_phsref.png", output_basename);
    let output_filepath = plot_dir.join(plot_filename);

    phase_reference_plot(
        &cal_results.add_plot_times,
        &original_cal_phases,
        &fitted_cal_phases,
        &target_results.add_plot_times,
        &original_target_phases,
        &target_results.add_plot_phase,
        output_filepath.to_str().unwrap(),
    )?;

    println!("Phase reference plot saved to: {:?}\n", output_filepath);

    Ok(())
}

/// Executes the acceleration search analysis.
fn run_acel_search_analysis(args: &Args, acel_search_degrees: &[i32]) -> Result<(), Box<dyn Error>> {
    println!("Starting acceleration search analysis...");

    let input_path = args.input.as_ref().unwrap();

    // --- Create Output Directory ---
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let output_dir = parent_dir.join("frinZ").join("acel_search");
    fs::create_dir_all(&output_dir)?;
    let base_filename = input_path.file_stem().unwrap().to_str().unwrap();

    let mut file = File::open(input_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let mut cursor = Cursor::new(buffer.as_slice());

    let header = parse_header(&mut cursor)?;

    let mut total_acel_correct = args.acel_correct;
    let mut total_rate_correct = args.rate_correct;

    // Get effective_integ_time from the first sector
    cursor.set_position(0);
    let (_, _, effective_integ_time) = read_visibility_data(&mut cursor, &header, 1, 0, 0, false)?;

    

    // Helper function to write data
    let write_fit_data = |path: &Path, times: &[f64], phases: &[f32], coeffs: Option<&[f64]>| -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        writeln!(writer, "# Time [s]   Phase [rad]")?;
        if let Some(c) = coeffs {
            if c.len() == 3 { // Quadratic
                writeln!(writer, "# Fitted: y = {:.6e} * x^2 + {:.6e} * x + {:.6e}", c[2], c[1], c[0])?;
                writeln!(writer, "# Corrected Acel: {:.6e} (from x^2 / PI)", c[2] / std::f64::consts::PI)?;
                writeln!(writer, "# Corrected Rate: {:.6e} (from x / (2 * PI))", c[1] / (2.0 * std::f64::consts::PI))?;
            } else if c.len() == 2 { // Linear
                writeln!(writer, "# Fitted: y = {:.6e} * x + {:.6e}", c[1], c[0])?;
                writeln!(writer, "# Corrected Rate: {:.6e} (from x / (2 * PI))", c[1] / (2.0 * std::f64::consts::PI))?;
            }
        }
        for i in 0..times.len() {
            writeln!(writer, "{:.6} {:.6}", times[i], phases[i])?;
        }
        Ok(())
    };

    // Define the data acquisition and phase extraction helper function
    let get_phases_from_corrected_data = |
        cursor: &mut Cursor<&[u8]>,
        header: &CorHeader,
        args: &Args,
        effective_integ_time: f32,
        obs_time_start: DateTime<Utc>,
        current_total_rate_correct: f32,
        current_total_acel_correct: f32,
    | -> Result<(Vec<f64>, Vec<f32>), Box<dyn Error>> {
        cursor.set_position(256); // Reset cursor to after header for each data pass

        let mut phases_collected: Vec<f32> = Vec::new();
        let mut times_collected: Vec<f64> = Vec::new();

        for loop_idx in 0..args.loop_ {
            let (complex_vec, current_obs_time, _eff_integ_time) = match read_visibility_data(
                cursor, header, args.length, args.skip, loop_idx, false,
            ) {
                Ok(data) => data,
                Err(_) => break, // Stop if we can't read more data
            };

            if complex_vec.is_empty() {
                break; // Stop if no data was read
            }

            let start_time_offset_sec =
                (current_obs_time - obs_time_start).num_seconds() as f32;

            let corrected_complex_vec = {
                let input_data_2d: Vec<Vec<Complex<f64>>> = complex_vec
                    .chunks(header.fft_point as usize / 2)
                    .map(|chunk| {
                        chunk.iter().map(|&c| Complex::new(c.re as f64, c.im as f64)).collect()
                    })
                    .collect();
                let corrected_2d = apply_phase_correction(
                    &input_data_2d,
                    current_total_rate_correct,
                    0.0, // No delay correction in this mode
                    current_total_acel_correct,
                    effective_integ_time,
                    header.sampling_speed as u32,
                    header.fft_point as u32,
                    start_time_offset_sec,
                );
                corrected_2d
                    .into_iter()
                    .flatten()
                    .map(|v| Complex::new(v.re as f32, v.im as f32))
                    .collect::<Vec<C32>>()
            };

            let fft_point_half = (header.fft_point / 2) as usize;
            let actual_length = corrected_complex_vec.len() / fft_point_half;

            if actual_length == 0 {
                continue;
            }

            let (freq_rate_array, padding_length) =
                process_fft(&corrected_complex_vec, actual_length as i32, header.fft_point, &[]);
            let delay_rate_array_comp =
                process_ifft(&freq_rate_array, header.fft_point, padding_length);
            let _delay_rate_array_abs = delay_rate_array_comp.mapv(|x| x.norm());

            let (peak_rate_idx, peak_delay_idx) = {
                let fft_point_f32 = header.fft_point as f32;
                let fft_point_usize = header.fft_point as usize;
                let fft_point_half = fft_point_usize / 2;
                let padding_length_half = padding_length / 2;

                let delay_range = Array::linspace(-(fft_point_f32 / 2.0) + 1.0, fft_point_f32 / 2.0, fft_point_usize);
                let rate_range = crate::utils::rate_cal(padding_length as f32, effective_integ_time);

                if !args.delay_window.is_empty() && !args.rate_window.is_empty() {
                    let delay_win_low = args.delay_window[0];
                    let delay_win_high = args.delay_window[1];
                    let rate_win_low = args.rate_window[0];
                    let rate_win_high = args.rate_window[1];

                    let mut max_val_in_window = 0.0f32;
                    let mut temp_peak_rate_idx = padding_length_half;
                    let mut temp_peak_delay_idx = fft_point_half;

                    for r_idx in 0..rate_range.len() {
                        if rate_range[r_idx] >= rate_win_low && rate_range[r_idx] <= rate_win_high {
                            for d_idx in 0..delay_range.len() {
                                if delay_range[d_idx] >= delay_win_low && delay_range[d_idx] <= delay_win_high {
                                    let current_val = _delay_rate_array_abs[[r_idx, d_idx]];
                                    if current_val > max_val_in_window {
                                        max_val_in_window = current_val;
                                        temp_peak_rate_idx = r_idx;
                                        temp_peak_delay_idx = d_idx;
                                    }
                                }
                            }
                        }
                    }
                    (temp_peak_rate_idx, temp_peak_delay_idx)
                } else {
                    let mut max_val = -1.0f32;
                    let mut max_r_idx = 0;
                    let mut max_d_idx = 0;
                    for r_idx in 0.._delay_rate_array_abs.shape()[0] {
                        for d_idx in 0.._delay_rate_array_abs.shape()[1] {
                            let current_val = _delay_rate_array_abs[[r_idx, d_idx]];
                            if current_val > max_val {
                                max_val = current_val;
                                max_r_idx = r_idx;
                                max_d_idx = d_idx;
                            }
                        }
                    }
                    (max_r_idx, max_d_idx)
                }
            };
            let phase_rad = delay_rate_array_comp[[peak_rate_idx, peak_delay_idx]].arg();

            phases_collected.push(phase_rad);
            times_collected.push(start_time_offset_sec as f64);
            // current_time_offset_sec += actual_length as f64 * effective_integ_time as f64; // This line is no longer needed for times_collected
        }

        unwrap_phase_radians(&mut phases_collected);
        Ok((times_collected, phases_collected))
    };


    // Initialize obs_time_start once before the loop
    cursor.set_position(256); // Reset cursor for first data read
    let (_, first_obs_time, _) = read_visibility_data(&mut cursor, &header, args.length, args.skip, 0, false)?;
    let obs_time_start = first_obs_time;

    for (step_idx, &degree) in acel_search_degrees.iter().enumerate() {
        println!("Step {}: Fitting with degree {}", step_idx + 1, degree);

        // Get data for this step
        let (times_for_fit, phases_for_fit) = get_phases_from_corrected_data(
            &mut cursor,
            &header,
            args,
            effective_integ_time,
            obs_time_start,
            total_rate_correct, // Use current total corrections
            total_acel_correct,
        )?;
        let phases_f64: Vec<f64> = phases_for_fit.iter().map(|&p| p as f64).collect();

        if times_for_fit.len() < (degree + 1) as usize {
            eprintln!(
                "Warning: Not enough data points for degree {} fitting (need at least {}). Skipping this step.",
                degree, degree + 1
            );
            println!(
                "  Updated acel: {:.6e}, Updated rate: {:.6e}",
                total_acel_correct,
                total_rate_correct
            );
            continue;
        }

        match degree {
            2 => { // Quadratic Fit
                // --- Output Data for Quad Fit ---
                let quad_path = output_dir.join(format!("{}_step{}_quadric.txt", base_filename, step_idx + 1));
                if let Ok(coeffs) = fitting::fit_polynomial_least_squares(&times_for_fit, &phases_f64, 2) {
                    println!("  Quad fit: x^2={:.6e}, x={:.6e}, c={:.6e}", coeffs[2], coeffs[1], coeffs[0]);

                    // Apply the corrections (with PI division, as per paper's model)
                    total_acel_correct += (coeffs[2] / std::f64::consts::PI) as f32;
                    total_rate_correct += (coeffs[1] / (2.0 * std::f64::consts::PI)) as f32;

                    write_fit_data(&quad_path, &times_for_fit, &phases_for_fit, Some(&coeffs))?;
                } else {
                    eprintln!("Warning: Quadratic fitting failed. Skipping acel and quad-rate update for this step.");
                    write_fit_data(&quad_path, &times_for_fit, &phases_for_fit, None)?;
                }
            }
            1 => { // Linear Fit
                // --- Output Data for Linear Fit ---
                let linear_path = output_dir.join(format!("{}_step{}_linear.txt", base_filename, step_idx + 1));
                if let Ok((slope, intercept)) = fitting::fit_linear_least_squares(&times_for_fit, &phases_f64) {
                    write_fit_data(&linear_path, &times_for_fit, &phases_for_fit, Some(&vec![intercept, slope]))?;
                    println!("  Linear fit: m={:.6e}", slope);
                    total_rate_correct += (slope / (2.0 * std::f64::consts::PI)) as f32;
                } else {
                    eprintln!("Warning: Linear fitting failed. Skipping linear-rate update for this step.");
                    write_fit_data(&linear_path, &times_for_fit, &phases_for_fit, None)?;
                }
            }
            _ => {
                eprintln!("Error: Unsupported fitting degree {}. Skipping this step.", degree);
            }
        }

        println!("  Updated acel: {:.9e} (Hz/s), Updated rate: {:.9e} (Hz)", total_acel_correct, total_rate_correct);

        println!("  Copypaste acel: {:.18} (Hz/s), Updated rate: {:.15} (Hz)", total_acel_correct, total_rate_correct);
    }

    Ok(())
}

/// Processes a single .cor file and returns the collected data for plotting.
fn process_cor_file(
    input_path: &Path,
    args: &Args,
) -> Result<ProcessResult, Box<dyn Error>> {
    // --- File and Path Setup ---
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let frinz_dir = parent_dir.join("frinZ");
    fs::create_dir_all(&frinz_dir)?;

    let basename = input_path.file_stem().unwrap().to_str().unwrap();
    let label: Vec<String> = basename.split('_').map(String::from).collect();

    // --- Create Output Directories ---
    let mut plot_path: Option<PathBuf> = None;
    if args.plot {
        let path = frinz_dir.join("fringe_graph");
        fs::create_dir_all(&path)?;
        plot_path = Some(path);
    }

    let mut output_path: Option<PathBuf> = None;
    if args.output {
        let path = frinz_dir.join("fringe_output");
        fs::create_dir_all(&path)?;
        output_path = Some(path);
    }

    let mut bandpass_output_path: Option<PathBuf> = None;
    if args.bandpass_table {
        let path = frinz_dir.join("bandpass_table");
        fs::create_dir_all(&path)?;
        bandpass_output_path = Some(path);
    }

    if args.cumulate != 0 {
        let path = frinz_dir.join(format!("cumulate/len{}s", args.cumulate));
        fs::create_dir_all(&path)?;
    }

    if !args.rfi.is_empty() {
        let _ = frinz_dir.join("rfi_history");
    }

    if args.add_plot {
        let path = frinz_dir.join("add_plot");
        fs::create_dir_all(&path)?;
    }

    // --- Read .cor File ---
    let mut file = File::open(input_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let mut cursor = Cursor::new(buffer.as_slice());

    // --- Parse Header ---
    let header = parse_header(&mut cursor)?;
    let bw = header.sampling_speed as f32 / 2.0 / 1_000_000.0;
    let rbw = bw / header.fft_point as f32 * 2.0;

    // --- RFI Handling ---
    let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw)?;

    // --- Bandpass Handling ---
    let bandpass_data = if let Some(bp_path) = &args.bandpass {
        Some(read_bandpass_file(bp_path)?)
    } else {
        None
    };

    // --- Output Header Information ---
    if args.output || args.header {
        let cor_header_path = frinz_dir.join("cor_header");
        fs::create_dir_all(&cor_header_path)?;
        let header_info_str = output_header_info(&header, &cor_header_path, basename)?;
        if args.header {
            println!("{}", header_info_str);
        }
    }

    // --- Loop and Processing Setup ---
    // Get effective_integ_time from the first sector
    cursor.set_position(0); // Reset cursor to beginning of file
    let (_, _, effective_integ_time) = read_visibility_data(
        &mut cursor,
        &header,
        1, // read only 1 sector
        0, // skip 0
        0, // loop_index 0
        false,
    )?;
    cursor.set_position(256); // Reset cursor to after header for main loop

    let pp = header.number_of_sector;
    let mut length = if args.length == 0 { pp } else { args.length }; // This length is in seconds if args.length is in seconds

    // Cap args.length if it's too large
    let total_obs_time_seconds = pp as f32 * effective_integ_time;
    if args.length != 0 && args.length as f32 > total_obs_time_seconds {
        // Cap args.length to total_obs_time_seconds
        // Then convert this capped seconds value back to sectors for the 'length' variable
        length = (total_obs_time_seconds / effective_integ_time).ceil() as i32;
        // If effective_integ_time is 0, this will cause division by zero.
        // effective_integ_time should not be 0 for valid data.
    } else if args.length != 0 {
        // If args.length is specified and not capped, convert it to sectors
        length = (args.length as f32 / effective_integ_time).ceil() as i32;
    }
    let mut loop_count = if (pp - args.skip) / length <= 0 {
        1
    } else if (pp - args.skip) / length <= args.loop_ {
        (pp - args.skip) / length
    } else {
        args.loop_
    };

    if args.cumulate != 0 {
        if args.cumulate >= pp {
            eprintln!(
                "The specified cumulation length, {} s, is more than the observation time, {} s.",
                args.cumulate,
                pp
            );
            process::exit(1);
        }
        length = args.cumulate;
        loop_count = pp / args.cumulate;
    }

    let mut delay_output_str = String::new();
    let mut freq_output_str = String::new();

    let mut cumulate_len: Vec<f32> = Vec::new();
    let mut cumulate_snr: Vec<f32> = Vec::new();

    let mut add_plot_amp: Vec<f32> = Vec::new();
    let mut add_plot_phase: Vec<f32> = Vec::new();
    let mut add_plot_snr: Vec<f32> = Vec::new();
    let mut add_plot_noise: Vec<f32> = Vec::new();
    let mut add_plot_times: Vec<DateTime<Utc>> = Vec::new();
    let mut obs_time: Option<chrono::DateTime<chrono::Utc>> = None;

    // --- Main Processing Loop ---
    for l1 in 0..loop_count {
        let current_length = if args.cumulate != 0 { (l1 + 1) * length } else { length };
        let (complex_vec, current_obs_time, effective_integ_time) = read_visibility_data(
            &mut cursor,
            &header,
            current_length,
            args.skip,
            l1,
            args.cumulate != 0,
        )?;
        if l1 == 0 {
            obs_time = Some(current_obs_time);
        }

        let mut analysis_results;
        let freq_rate_array;
        let delay_rate_2d_data_comp;

        if args.search_deep {
            // --- DEEP SEARCH MODE ---
            let deep_search_result = deep_search::run_deep_search(
                &complex_vec,
                &header,
                current_length,
                effective_integ_time,
                &current_obs_time,
                &obs_time.unwrap(),
                &rfi_ranges,
                &bandpass_data,
                args,
                pp,
                args.cpu, // Pass the new cpu argument
            )?;

            analysis_results = deep_search_result.analysis_results;
            freq_rate_array = deep_search_result.freq_rate_array;
            delay_rate_2d_data_comp = deep_search_result.delay_rate_2d_data;
        } else if args.search {
            // --- SEARCH MODE ---
            let mut total_delay_correct = args.delay_correct;
            let mut total_rate_correct = args.rate_correct;
            let total_acel_correct = args.acel_correct;

            let mut analysis_results_mut = None;
            let mut freq_rate_array_mut = None;
            let mut delay_rate_2d_data_comp_mut = None;

            for i in 0..args.iter {
                let mut current_args = args.clone();
                current_args.delay_correct = total_delay_correct;
                current_args.rate_correct = total_rate_correct;
                current_args.acel_correct = total_acel_correct;
                if i > 0 {
                    // After the first iteration, narrow the search window
                    current_args.delay_window = vec![-2.0, 2.0];
                    current_args.rate_window = vec![-0.05, 0.05];
                }

                let temp_complex_vec = if total_delay_correct == 0.0
                    && total_rate_correct == 0.0
                    && total_acel_correct == 0.0
                {
                    complex_vec.clone()
                } else {
                    let input_data_2d: Vec<Vec<Complex<f64>>> = complex_vec
                        .chunks(header.fft_point as usize / 2)
                        .map(|chunk| {
                            chunk
                                .iter()
                                .map(|&c| Complex::new(c.re as f64, c.im as f64))
                                .collect()
                        })
                        .collect();
                    let start_time_offset_sec =
                        (current_obs_time - obs_time.unwrap()).num_seconds() as f32;
                    let corrected_complex_vec_2d = fft::apply_phase_correction(
                        &input_data_2d,
                        total_rate_correct,
                        total_delay_correct,
                        total_acel_correct,
                        effective_integ_time,
                        header.sampling_speed as u32,
                        header.fft_point as u32,
                        start_time_offset_sec,
                    );
                    corrected_complex_vec_2d
                        .into_iter()
                        .flatten()
                        .map(|v| Complex::new(v.re as f32, v.im as f32))
                        .collect()
                };

                let (mut iter_freq_rate_array, padding_length) = process_fft(
                    &temp_complex_vec,
                    current_length,
                    header.fft_point,
                    &rfi_ranges,
                );
                if let Some(bp_data) = &bandpass_data {
                    apply_bandpass_correction(&mut iter_freq_rate_array, bp_data);
                }
                let iter_delay_rate_2d_data_comp =
                    process_ifft(&iter_freq_rate_array, header.fft_point, padding_length);
                let iter_results = analyze_results(
                    &iter_freq_rate_array,
                    &iter_delay_rate_2d_data_comp,
                    &header,
                    current_length,
                    effective_integ_time,
                    &current_obs_time,
                    padding_length,
                    &current_args,
                );

                total_delay_correct += iter_results.delay_offset;
                total_rate_correct += iter_results.rate_offset;

                // Store the results of the last iteration
                analysis_results_mut = Some(iter_results);
                freq_rate_array_mut = Some(iter_freq_rate_array);
                delay_rate_2d_data_comp_mut = Some(iter_delay_rate_2d_data_comp);
            }

            analysis_results = analysis_results_mut.unwrap();
            analysis_results.length_f32 = current_length as f32 * effective_integ_time;
            analysis_results.residual_delay = total_delay_correct;
            analysis_results.residual_rate = total_rate_correct;
            analysis_results.residual_acel = total_acel_correct;
            analysis_results.corrected_delay = total_delay_correct;
            analysis_results.corrected_rate = total_rate_correct;
            analysis_results.corrected_acel = total_acel_correct;
            freq_rate_array = freq_rate_array_mut.unwrap();
            delay_rate_2d_data_comp = delay_rate_2d_data_comp_mut.unwrap();
        } else {
            // --- NORMAL MODE ---
            let mut corrected_complex_vec: Vec<C32> = complex_vec.clone();
            if args.delay_correct != 0.0 || args.rate_correct != 0.0 || args.acel_correct != 0.0 {
                let input_data_2d: Vec<Vec<Complex<f64>>> = complex_vec
                    .chunks(header.fft_point as usize / 2)
                    .map(|chunk| {
                        chunk
                            .iter()
                            .map(|&c| Complex::new(c.re as f64, c.im as f64))
                            .collect()
                    })
                    .collect();
                let start_time_offset_sec =
                    (current_obs_time - obs_time.unwrap()).num_seconds() as f32;
                let corrected_complex_vec_2d = fft::apply_phase_correction(
                    &input_data_2d,
                    args.rate_correct,
                    args.delay_correct,
                    args.acel_correct,
                    effective_integ_time,
                    header.sampling_speed as u32,
                    header.fft_point as u32,
                    start_time_offset_sec,
                );
                corrected_complex_vec = corrected_complex_vec_2d
                    .into_iter()
                    .flatten()
                    .map(|v| Complex::new(v.re as f32, v.im as f32))
                    .collect();
            }
            let (mut final_freq_rate_array, padding_length) = process_fft(
                &corrected_complex_vec,
                current_length,
                header.fft_point,
                &rfi_ranges,
            );
            if let Some(bp_data) = &bandpass_data {
                apply_bandpass_correction(&mut final_freq_rate_array, bp_data);
            }
            let final_delay_rate_2d_data_comp =
                process_ifft(&final_freq_rate_array, header.fft_point, padding_length);
            analysis_results = analyze_results(
                &final_freq_rate_array,
                &final_delay_rate_2d_data_comp,
                &header,
                current_length,
                effective_integ_time,
                &current_obs_time,
                padding_length,
                &args,
            );
            analysis_results.length_f32 = (current_length as f32 * effective_integ_time).ceil();
            freq_rate_array = final_freq_rate_array;
            delay_rate_2d_data_comp = final_delay_rate_2d_data_comp;
        }

        // --- Output and Plotting ---
        let label_str: Vec<&str> = label.iter().map(|s| s.as_str()).collect();
        let base_filename = generate_output_names(
            &header,
            &current_obs_time,
            &label_str,
            !rfi_ranges.is_empty(),
            args.frequency,
            args.bandpass.is_some(),
            current_length,
        );

        if args.bandpass_table {
            if let Some(path) = &bandpass_output_path {
                let output_file_path = path.join(format!("{}_bandpass_table.bin", base_filename));
                write_complex_spectrum_binary(
                    &output_file_path,
                    &analysis_results.freq_rate_spectrum.to_vec(),
                    header.fft_point,
                )?;
                println!("Bandpass binary file written to {:?}", output_file_path);
                println!("Bandpass binary file format:");
                println!(
                    "  - Subsequent data consists of interleaved real and imaginary parts of complex spectra."
                );
                println!("  - Each real and imaginary part is a 4-byte f32 (LittleEndian).");
                println!("  - File extension should be .bin");
                println!("  - Output to {:?}", output_file_path);
                println!("Net step:");
                println!(
                    "  - Add --bandpass {:?} to the commandline argument in this program.",
                    output_file_path
                );
                return Ok(ProcessResult {
                    header,
                    label,
                    obs_time: obs_time.unwrap(),
                    length_arg: length,
                    cumulate_len,
                    cumulate_snr,
                    add_plot_times,
                    add_plot_amp,
                    add_plot_snr,
                    add_plot_phase,
                    add_plot_noise,
                });
            }
        }

        if !args.frequency {
            let delay_output_line = format_delay_output(&analysis_results, &label_str, args.length);
            if l1 == 0 {
                let header_str = "".to_string()
                    + "#************************************************************************************************************************************************************************************\n"
                    + "#      Epoch        Label    Source     Length    Amp      SNR     Phase     Noise-level      Res-Delay     Res-Rate            YAMAGU32-azel            YAMAGU34-azel             MJD \n"
                    + "#                                        [s]      [%]               [deg]     1-sigma[%]       [sample]       [Hz]      az[deg]  el[deg]  hgt[m]    az[deg]  el[deg]  hgt[m]          \n"
                    + "#************************************************************************************************************************************************************************************";
                print!("{}\n", header_str);
                delay_output_str += &header_str;
            }
            print!("{}\n", delay_output_line);
            delay_output_str += &delay_output_line;

            if args.cumulate != 0 {
                cumulate_len.push(current_length as f32);
                cumulate_snr.push(analysis_results.delay_snr);
            }

            add_plot_phase.push(analysis_results.delay_phase);
            add_plot_times.push(current_obs_time);

            if args.add_plot {
                add_plot_amp.push(analysis_results.delay_max_amp * 100.0);
                add_plot_snr.push(analysis_results.delay_snr);
                add_plot_noise.push(analysis_results.delay_noise * 100.0);
            }

            if l1 == loop_count - 1 && args.output {
                if let Some(path) = &output_path {
                    let output_file_path =
                        path.join(format!("{}.txt", format!("{}_time", base_filename)));
                    fs::write(output_file_path, &delay_output_str)?;
                }
            }
        } else {
            let freq_output_line = format_freq_output(&analysis_results, &label_str, args.length);
            if l1 == 0 {
                let header_str = "".to_string()
                    + "#******************************************************************************************************************************************************************************************\n"
                    + "#      Epoch        Label    Source     Length    Amp      SNR     Phase     Frequency     Noise-level      Res-Rate            YAMAGU32-azel             YAMAGU34-azel             MJD      \n"
                    + "#                                        [s]      [%]              [deg]       [MHz]       1-sigma[%]        [Hz]        az[deg]  el[deg]  hgt[m]   az[deg]  el[deg]  hgt[m]                \n"
                    + "#******************************************************************************************************************************************************************************************";
                print!("{}\n", header_str);
                freq_output_str += &header_str;
            }
            print!("{}\n", freq_output_line);
            freq_output_str += &freq_output_line;

            if l1 == loop_count - 1 && args.output {
                if let Some(path) = &output_path {
                    let output_file_path =
                        path.join(format!("{}.txt", format!("{}_freq", base_filename)));
                    fs::write(output_file_path, &freq_output_str)?;
                }
            }
        }

        if args.plot && args.cumulate == 0 {
            if let Some(path) = &plot_path {
                let length_label =
                    if args.length == 0 { "0".to_string() } else { args.length.to_string() };
                let plot_dir = if !args.frequency {
                    path.join(format!("time_domain/len{}s", length_label))
                } else {
                    path.join(format!("freq_domain/len{}s", length_label))
                };
                fs::create_dir_all(&plot_dir)?;
                let output_filename = if !args.frequency {
                    plot_dir.join(format!("{}_delay_rate_search.png", base_filename))
                } else {
                    plot_dir.join(format!("{}_freq_rate_search.png", base_filename))
                };

                if !args.frequency {
                    let delay_profile: Vec<(f64, f64)> = analysis_results
                        .delay_range
                        .iter()
                        .zip(analysis_results.visibility.iter())
                        .map(|(&x, &y)| (x as f64, y as f64))
                        .collect();
                    let rate_profile: Vec<(f64, f64)> = analysis_results
                        .rate_range
                        .iter()
                        .zip(analysis_results.delay_rate.iter())
                        .map(|(&x, &y)| (x as f64, y as f64))
                        .collect();
                    let rows = delay_rate_2d_data_comp.shape()[0] as u32;
                    let cols = delay_rate_2d_data_comp.shape()[1] as u32;
                    let max_norm = delay_rate_2d_data_comp
                        .iter()
                        .map(|c| c.norm())
                        .fold(0.0f32, |acc, x| acc.max(x));
                    let mut img = ImageBuffer::new(cols, rows);
                    for y in 0..rows {
                        for x in 0..cols {
                            let val = delay_rate_2d_data_comp[[y as usize, x as usize]].norm();
                            let normalized_val =
                                if max_norm > 0.0 { (val / max_norm * 255.0) as u8 } else { 0 };
                            img.put_pixel(x, y, image::Luma([normalized_val]));
                        }
                    }
                    let blurred_img = filter::gaussian_blur_f32(&img, 1.0);
                    let delay_data: Vec<f32> =
                        analysis_results.delay_range.iter().map(|&x| x as f32).collect();
                    let rate_data: Vec<f32> =
                        analysis_results.rate_range.iter().map(|&x| x as f32).collect();
                    let heatmap_func = move |delay: f64, rate: f64| -> f64 {
                        let d_min = delay_data[0] as f64;
                        let d_max = *delay_data.last().unwrap() as f64;
                        let r_min = rate_data[0] as f64;
                        let r_max = *rate_data.last().unwrap() as f64;
                        let x_img = ((delay - d_min) / (d_max - d_min) * (cols - 1) as f64)
                            .max(0.0)
                            .min((cols - 1) as f64);
                        let y_img = ((rate - r_min) / (r_max - r_min) * (rows - 1) as f64)
                            .max(0.0)
                            .min((rows - 1) as f64);
                        let x_floor = x_img.floor() as u32;
                        let y_floor = y_img.floor() as u32;
                        let x_ceil = (x_img.ceil() as u32).min(cols - 1);
                        let y_ceil = (y_img.ceil() as u32).min(rows - 1);
                        let fx = x_img - x_img.floor();
                        let fy = y_img - y_img.floor();
                        let q11 = (blurred_img.get_pixel(x_floor, y_floor)[0] as f64) / 255.0
                            * (max_norm as f64);
                        let q12 = (blurred_img.get_pixel(x_ceil, y_floor)[0] as f64) / 255.0
                            * (max_norm as f64);
                        let q21 = (blurred_img.get_pixel(x_floor, y_ceil)[0] as f64) / 255.0
                            * (max_norm as f64);
                        let q22 = (blurred_img.get_pixel(x_ceil, y_ceil)[0] as f64) / 255.0
                            * (max_norm as f64);
                        let r1 = q11 * (1.0 - fx) + q12 * fx;
                        let r2 = q21 * (1.0 - fx) + q22 * fx;
                        r1 * (1.0 - fy) + r2 * fy
                    };
                    let stat_keys = vec![
                        "Epoch (UTC)",
                        "Station 1 & 2",
                        "Source",
                        "Length [s]",
                        "Frequency [MHz]",
                        "Peak Amp [%]",
                        "Peak Phs [deg]",
                        "SNR (1 σ [%])",
                        "Delay (residual) [sps]",
                        "Delay (corrected) [sps]",
                        "Rate (residual) [mHz]",
                        "Rate (corrected) [mHz]",
                    ];
                    let stat_vals = vec![
                        analysis_results.yyyydddhhmmss1.to_string(),
                        format!("{} & {}", header.station1_name, header.station2_name),
                        analysis_results.source_name.to_string(),
                        format!("{:.3}", analysis_results.length_f32.ceil()),
                        format!("{:.3}", header.observing_frequency as f32 / 1e6),
                        format!("{:.6}", analysis_results.delay_max_amp * 100.0),
                        format!("{:+.5}", analysis_results.delay_phase),
                        format!(
                            "{:.3} ({:.6})",
                            analysis_results.delay_snr,
                            analysis_results.delay_noise * 100.0
                        ),
                        format!("{:+.6}", analysis_results.residual_delay),
                        format!("{:+.6}", analysis_results.corrected_delay),
                        format!("{:+.6}", analysis_results.residual_rate * 1000.0),
                        format!("{:+.6}", analysis_results.corrected_rate * 1000.0),
                    ];
                    delay_plane(
                        &delay_profile,
                        &rate_profile,
                        heatmap_func,
                        &stat_keys.iter().map(|s| s.as_ref()).collect::<Vec<&str>>(),
                        &stat_vals.iter().map(|s| s.as_ref()).collect::<Vec<&str>>(),
                        output_filename.to_str().unwrap(),
                        &analysis_results.rate_range,
                        analysis_results.length_f32,
                        effective_integ_time,
                        &args.delay_window,
                        &args.rate_window,
                        max_norm as f64,
                    )?;
                } else {
                    let freq_amp_profile: Vec<(f64, f64)> = analysis_results
                        .freq_range
                        .iter()
                        .zip(analysis_results.freq_rate_spectrum.iter().map(|c| c.norm()))
                        .map(|(&x, y)| (x as f64, y as f64))
                        .collect();
                    let freq_phase_profile: Vec<(f64, f64)> = analysis_results
                        .freq_range
                        .iter()
                        .zip(
                            analysis_results.freq_rate_spectrum.iter().map(|c| c.arg().to_degrees()),
                        )
                        .map(|(&x, y)| (x as f64, y as f64))
                        .collect();
                    let rate_profile: Vec<(f64, f64)> = analysis_results
                        .rate_range
                        .iter()
                        .zip(analysis_results.freq_rate.iter())
                        .map(|(&x, &y)| (x as f64, y as f64))
                        .collect();
                    let freq_data: Vec<f32> =
                        analysis_results.freq_range.iter().map(|&x| x as f32).collect();
                    let rate_data: Vec<f32> =
                        analysis_results.rate_range.iter().map(|&x| x as f32).collect();
                    let heatmap_func = |freq: f64, rate: f64| -> f64 {
                        let f_min = freq_data[0] as f64;
                        let f_max = *freq_data.last().unwrap() as f64;
                        let r_min = rate_data[0] as f64;
                        let r_max = *rate_data.last().unwrap() as f64;
                        if freq < f_min || freq > f_max || rate < r_min || rate > r_max {
                            return 0.0;
                        }
                        let rows = freq_rate_array.shape()[0];
                        let cols = freq_rate_array.shape()[1];
                        let freq_idx =
                            (((freq - f_min) / (f_max - f_min)) * (rows - 1) as f64).round() as usize;
                        let rate_idx =
                            (((rate - r_min) / (r_max - r_min)) * (cols - 1) as f64).round() as usize;
                        if freq_idx < rows && rate_idx < cols {
                            freq_rate_array[[freq_idx, rate_idx]].norm() as f64
                        } else {
                            0.0
                        }
                    };
                    let stat_keys = vec![
                        "Epoch (UTC)",
                        "Station 1 & 2",
                        "Source",
                        "Length [s]",
                        "Frequency [MHz]",
                        "Peak Amp [%]",
                        "Peak Phs [deg]",
                        "Peak Freq [MHz]",
                        "SNR (1 σ [%])",
                        "Rate (residual) [mHz]",
                    ];
                    let stat_vals = vec![
                        analysis_results.yyyydddhhmmss1.to_string(),
                        format!("{} & {}", header.station1_name, header.station2_name),
                        analysis_results.source_name.to_string(),
                        format!("{:.3}", analysis_results.length_f32.ceil()),
                        format!("{:.3}", header.observing_frequency as f32 / 1e6),
                        format!("{:.6}", analysis_results.freq_max_amp * 100.0),
                        format!("{:+.5}", analysis_results.freq_phase),
                        format!("{:+.6}", analysis_results.freq_max_freq),
                        format!(
                            "{:.3} ({:.6})",
                            analysis_results.freq_snr,
                            analysis_results.freq_noise * 100.0
                        ),
                        format!("{:+.6}", analysis_results.residual_rate * 1000.0),
                    ];
                    let max_norm_freq =
                        freq_rate_array.iter().map(|c| c.norm()).fold(0.0f32, |acc, x| acc.max(x));
                    frequency_plane(
                        &freq_amp_profile,
                        &freq_phase_profile,
                        &rate_profile,
                        heatmap_func,
                        &stat_keys.iter().map(|s| s.as_ref()).collect::<Vec<&str>>(),
                        &stat_vals.iter().map(|s| s.as_ref()).collect::<Vec<&str>>(),
                        output_filename.to_str().unwrap(),
                        bw as f64,
                        max_norm_freq as f64,
                    )?;
                }
            }
        }
    }

    Ok(ProcessResult {
        header,
        label,
        obs_time: obs_time.unwrap(),
        length_arg: length,
        cumulate_len,
        cumulate_snr,
        add_plot_times,
        add_plot_amp,
        add_plot_snr,
        add_plot_phase,
        add_plot_noise,
    })
}
