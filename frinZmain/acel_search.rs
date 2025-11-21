use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::Path;

use chrono::{DateTime, Utc};
use num_complex::Complex;

use crate::args::Args;
use crate::fitting;
use crate::header::{parse_header, CorHeader};
use crate::plot::plot_acel_search_result;
use crate::processing::run_analysis_pipeline;
use crate::read::read_visibility_data;
use crate::rfi::parse_rfi_ranges;
use crate::utils::unwrap_phase;

type C32 = Complex<f32>;

struct VisibilityDataPoint {
    complex_vec: Vec<C32>,
    obs_time: DateTime<Utc>,
    sector_count: i32,
}

fn collect_visibility_data(
    cursor: &mut Cursor<&[u8]>,
    header: &CorHeader,
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<Vec<VisibilityDataPoint>, Box<dyn Error>> {
    let mut collected_data = Vec::new();
    cursor.set_position(256); // Reset cursor to after header

    for loop_idx in 0..args.loop_ {
        let (complex_vec, current_obs_time, _eff_integ_time) = match read_visibility_data(
            cursor,
            header,
            args.length,
            args.skip,
            loop_idx,
            false,
            pp_flag_ranges,
        ) {
            Ok(data) => data,
            Err(_) => break, // Stop if we can't read more data
        };

        if complex_vec.is_empty() {
            break; // Stop if no data was read
        }

        let fft_point_half = (header.fft_point / 2) as usize;
        if fft_point_half == 0 {
            eprintln!("#ERROR: FFT point が 0 です (acel-search)");
            break;
        }
        if complex_vec.len() % fft_point_half != 0 {
            eprintln!(
                "#ERROR: 複素データ長 ({}) が FFT チャンネル数 ({}) の整数倍ではありません (acel-search)。",
                complex_vec.len(),
                fft_point_half
            );
            continue;
        }
        let sector_count = (complex_vec.len() / fft_point_half) as i32;
        if sector_count == 0 {
            continue;
        }

        let is_flagged = time_flag_ranges
            .iter()
            .any(|(start, end)| current_obs_time >= *start && current_obs_time < *end);

        if is_flagged {
            println!(
                "#INFO: Skipping data at {} due to --flagging time range in acel-search.",
                current_obs_time.format("%Y-%m-%d %H:%M:%S")
            );
            continue;
        }

        collected_data.push(VisibilityDataPoint {
            complex_vec,
            obs_time: current_obs_time,
            sector_count,
        });
    }
    Ok(collected_data)
}

fn get_phases_from_collected_data(
    collected_data: &[VisibilityDataPoint],
    header: &CorHeader,
    args: &Args,
    effective_integ_time: f32,
    obs_time_start: DateTime<Utc>,
    current_total_rate_correct: f32,
    current_total_acel_correct: f32,
    rfi_ranges: &[(usize, usize)],
    bandpass_data: &Option<Vec<C32>>,
) -> Result<(Vec<f64>, Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn Error>> {
    let mut phases_collected: Vec<f32> = Vec::new();
    let mut times_collected: Vec<f64> = Vec::new();
    let mut residual_rates_hz: Vec<f32> = Vec::new();
    let mut residual_delays_samples: Vec<f32> = Vec::new();

    for data_point in collected_data {
        let start_time_offset_sec = (data_point.obs_time - obs_time_start).num_seconds() as f32;

        if data_point.sector_count <= 0 {
            continue;
        }
        let current_length = data_point.sector_count;
        let fft_point_half = data_point.complex_vec.len() / current_length as usize;
        if fft_point_half == 0 {
            continue;
        }
        let effective_fft_point = (fft_point_half * 2) as i32;

        let (analysis_results, _, _) = run_analysis_pipeline(
            &data_point.complex_vec,
            header,
            args,
            Some("peak"),
            args.delay_correct,
            current_total_rate_correct,
            current_total_acel_correct,
            current_length,
            current_length,
            effective_integ_time,
            &data_point.obs_time,
            &obs_time_start,
            rfi_ranges,
            bandpass_data,
            effective_fft_point,
        )?;

        let phase_rad = analysis_results.delay_phase.to_radians() as f32;

        phases_collected.push(phase_rad);
        times_collected.push(start_time_offset_sec as f64);
        residual_rates_hz.push(analysis_results.residual_rate);
        residual_delays_samples.push(analysis_results.residual_delay);
    }

    unwrap_phase(&mut phases_collected, true);
    Ok((
        times_collected,
        phases_collected,
        residual_rates_hz,
        residual_delays_samples,
    ))
}

pub fn run_acel_search_analysis(
    args: &Args,
    acel_search_degrees: &[i32],
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(), Box<dyn Error>> {
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

    let bandwidth_mhz = header.sampling_speed as f32 / 2.0 / 1_000_000.0;
    let rbw_mhz = bandwidth_mhz / header.fft_point as f32 * 2.0;
    let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw_mhz)?;
    let bandpass_data: Option<Vec<C32>> = None;

    // Get effective_integ_time from the first sector
    cursor.set_position(0);
    let (_, _, effective_integ_time) =
        read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;

    // Helper function to write data
    let write_fit_data = |path: &Path, coeffs: Option<&[f64]>| -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        if let Some(c) = coeffs {
            if c.len() == 3 {
                // Quadratic
                writeln!(
                    writer,
                    "# Fitted: y = {:.6e} * x^2 + {:.6e} * x + {:.6e}",
                    c[2], c[1], c[0]
                )?;
                writeln!(
                    writer,
                    "# Corrected Acel (Hz/s): {:.6e} (from x^2 / PI)",
                    c[2] / std::f64::consts::PI
                )?;
                writeln!(
                    writer,
                    "# Corrected Rate (Hz): {:.6e} (from x / (2 * PI))",
                    c[1] / (2.0 * std::f64::consts::PI)
                )?;
            } else if c.len() == 2 {
                // Linear
                writeln!(writer, "# Fitted: y = {:.6e} * x + {:.6e}", c[1], c[0])?;
                writeln!(
                    writer,
                    "# Corrected Rate: {:.6e} (from x / (2 * PI))",
                    c[1] / (2.0 * std::f64::consts::PI)
                )?;
            }
        }
        Ok(())
    };

    // Initialize obs_time_start once before the loop
    cursor.set_position(256); // Reset cursor for first data read
    let (_, first_obs_time, _) =
        read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;
    let obs_time_start = first_obs_time;

    // Collect all visibility data once to avoid re-reading
    let collected_data =
        collect_visibility_data(&mut cursor, &header, args, time_flag_ranges, pp_flag_ranges)?;

    for (step_idx, &degree) in acel_search_degrees.iter().enumerate() {
        println!("Step {}: Fitting with degree {}", step_idx + 1, degree);

        // Get phases from the pre-collected data with current corrections
        let (times_for_fit, phases_for_fit, residual_rates_hz, residual_delays_samples) =
            get_phases_from_collected_data(
                &collected_data,
                &header,
                args,
                effective_integ_time,
                obs_time_start,
                total_rate_correct,
                total_acel_correct,
                &rfi_ranges,
                &bandpass_data,
            )?;
        let phases_f64: Vec<f64> = phases_for_fit.iter().map(|&p| p as f64).collect();
        let rates_f64: Vec<f64> = residual_rates_hz.iter().map(|&r| r as f64).collect();
        let mut rate_fit_series: Option<Vec<f64>> = None;
        let mut rate_residual_series: Option<Vec<f64>> = None;
        let rate_based_acel = if rates_f64.len() >= 2 {
            match fitting::fit_linear_least_squares(&times_for_fit, &rates_f64) {
                Ok((slope, intercept)) => {
                    let fitted: Vec<f64> = times_for_fit
                        .iter()
                        .map(|&t| slope * t + intercept)
                        .collect();
                    let residuals_vec: Vec<f64> = rates_f64
                        .iter()
                        .zip(fitted.iter())
                        .map(|(&obs, &fit)| obs - fit)
                        .collect();
                    rate_fit_series = Some(fitted);
                    rate_residual_series = Some(residuals_vec);
                    Some(slope)
                }
                Err(err) => {
                    eprintln!(
                        "Warning: Rate-based linear fit failed in acel-search step {}: {}",
                        step_idx + 1,
                        err
                    );
                    None
                }
            }
        } else {
            None
        };

        let delays_samples_f64: Vec<f64> =
            residual_delays_samples.iter().map(|&d| d as f64).collect();
        let mut delay_fit_samples_series: Option<Vec<f64>> = None;
        let mut delay_residual_samples_series: Option<Vec<f64>> = None;
        let (delay_based_acel, delay_based_rate) = if delays_samples_f64.len() >= 3 {
            let sampling_hz = header.sampling_speed as f64;
            if sampling_hz > 0.0 {
                let delays_seconds: Vec<f64> = delays_samples_f64
                    .iter()
                    .map(|&d| d / sampling_hz)
                    .collect();
                match fitting::fit_polynomial_least_squares(&times_for_fit, &delays_seconds, 2) {
                    Ok(coeffs) => {
                        let fitted_seconds: Vec<f64> = times_for_fit
                            .iter()
                            .map(|&t| coeffs[0] + coeffs[1] * t + coeffs[2] * t * t)
                            .collect();
                        let residual_seconds: Vec<f64> = delays_seconds
                            .iter()
                            .zip(fitted_seconds.iter())
                            .map(|(&obs, &fit)| obs - fit)
                            .collect();
                        delay_fit_samples_series =
                            Some(fitted_seconds.iter().map(|v| v * sampling_hz).collect());
                        delay_residual_samples_series =
                            Some(residual_seconds.iter().map(|v| v * sampling_hz).collect());

                        let acel = 2.0 * coeffs[2] * header.observing_frequency;
                        let rate = coeffs[1] * header.observing_frequency;
                        (Some(acel), Some(rate))
                    }
                    Err(err) => {
                        eprintln!(
                            "Warning: Delay-based quadratic fit failed in acel-search step {}: {}",
                            step_idx + 1,
                            err
                        );
                        (None, None)
                    }
                }
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let mut phase_fit_series: Option<Vec<f64>> = None;
        let mut phase_residual_series: Option<Vec<f64>> = None;

        if times_for_fit.len() < (degree + 1) as usize {
            eprintln!(
                "Warning: Not enough data points for degree {} fitting (need at least {}). Skipping this step.",
                degree, degree + 1
            );
            println!(
                "  Updated acel: {:.6e}, Updated rate: {:.6e}",
                total_acel_correct, total_rate_correct
            );
            continue;
        }

        match degree {
            2 => {
                // Quadratic Fit
                let quad_path = output_dir.join(format!(
                    "{}_step{}_quadric.txt",
                    base_filename,
                    step_idx + 1
                ));
                if let Ok(coeffs) =
                    fitting::fit_polynomial_least_squares(&times_for_fit, &phases_f64, 2)
                {
                    println!(
                        "  Quad fit: x^2={:.6e}, x={:.6e}, c={:.6e}",
                        coeffs[2], coeffs[1], coeffs[0]
                    );
                    let fitted_phases: Vec<f64> = times_for_fit
                        .iter()
                        .map(|&t| coeffs[0] + coeffs[1] * t + coeffs[2] * t * t)
                        .collect();
                    let residual_phases: Vec<f64> = phases_f64
                        .iter()
                        .zip(fitted_phases.iter())
                        .map(|(&obs, &fit)| obs - fit)
                        .collect();
                    phase_fit_series = Some(fitted_phases);
                    phase_residual_series = Some(residual_phases);
                    total_acel_correct += (coeffs[2] / std::f64::consts::PI) as f32;
                    total_rate_correct += (coeffs[1] / (2.0 * std::f64::consts::PI)) as f32;
                    write_fit_data(&quad_path, Some(&coeffs))?;
                } else {
                    eprintln!("Warning: Quadratic fitting failed. Skipping acel and quad-rate update for this step.");
                    write_fit_data(&quad_path, None)?;
                }
            }
            1 => {
                // Linear Fit
                let linear_path =
                    output_dir.join(format!("{}_step{}_linear.txt", base_filename, step_idx + 1));
                if let Ok((slope, intercept)) =
                    fitting::fit_linear_least_squares(&times_for_fit, &phases_f64)
                {
                    let fitted_phases: Vec<f64> = times_for_fit
                        .iter()
                        .map(|&t| slope * t + intercept)
                        .collect();
                    let residual_phases: Vec<f64> = phases_f64
                        .iter()
                        .zip(fitted_phases.iter())
                        .map(|(&obs, &fit)| obs - fit)
                        .collect();
                    phase_fit_series = Some(fitted_phases);
                    phase_residual_series = Some(residual_phases);
                    write_fit_data(&linear_path, Some(&vec![intercept, slope]))?;
                    println!("  Linear fit: m={:.6e}", slope);
                    total_rate_correct += (slope / (2.0 * std::f64::consts::PI)) as f32;
                } else {
                    eprintln!("Warning: Linear fitting failed. Skipping linear-rate update for this step.");
                    write_fit_data(&linear_path, None)?;
                }
            }
            _ => {
                eprintln!(
                    "Error: Unsupported fitting degree {}. Skipping this step.",
                    degree
                );
            }
        }

        println!(
            "  +----------------------+--------------------------+--------------------------+"
        );
        println!(
            "  | Derivation Method    | Acceleration (Hz/s)      | Rate (Hz)                |"
        );
        println!(
            "  +----------------------+--------------------------+--------------------------+"
        );

        // Phase Fit
        println!(
            "  | Phase Fit (Quad)     | {:<+24.9e} | {:<+24.9e} |",
            total_acel_correct, total_rate_correct
        );

        // Rate-derived
        let rate_acel_str = rate_based_acel
            .map(|v| format!("{:<+24.9e}", v))
            .unwrap_or_else(|| format!("{:<24}", "(N/A)"));
        println!(
            "  | Rate-derived         | {} | {:<24} |",
            rate_acel_str, "(N/A)"
        );

        // Delay-derived
        let delay_acel_str = delay_based_acel
            .map(|v| format!("{:<+24.9e}", v))
            .unwrap_or_else(|| format!("{:<24}", "(N/A)"));
        let delay_rate_str = delay_based_rate
            .map(|v| format!("{:<+24.9e}", v))
            .unwrap_or_else(|| format!("{:<24}", "(N/A)"));
        println!(
            "  | Delay-derived        | {} | {} |",
            delay_acel_str, delay_rate_str
        );

        println!(
            "  +----------------------+--------------------------+--------------------------+"
        );

        // Copypaste lines
        println!(
            "  Copypaste (Phase Fit): --acel {:.18} --rate {:.15}",
            total_acel_correct, total_rate_correct
        );
        if let Some(rate_acel) = rate_based_acel {
            println!("  Copypaste (Rate-derived): --acel {:.18}", rate_acel);
        }
        if let (Some(delay_acel), Some(delay_rate)) = (delay_based_acel, delay_based_rate) {
            println!(
                "  Copypaste (Delay-derived): --acel {:.18} --rate {:.15}",
                delay_acel, delay_rate
            );
        }

        if let Some(fitted) = &phase_fit_series {
            let residuals = phase_residual_series.as_ref().map(|v| v.as_slice());
            let phase_plot_path =
                output_dir.join(format!("{}_step{}_phase.png", base_filename, step_idx + 1));
            plot_acel_search_result(
                &phase_plot_path,
                &times_for_fit,
                &phases_f64,
                Some(fitted.as_slice()),
                residuals,
                &format!("Phase Fit (step {})", step_idx + 1),
                "Phase [rad]",
            )?;
        }

        if let Some(fitted) = &rate_fit_series {
            let residuals = rate_residual_series.as_ref().map(|v| v.as_slice());
            let rate_plot_path = output_dir.join(format!(
                "{}_step{}_res_rate.png",
                base_filename,
                step_idx + 1
            ));
            plot_acel_search_result(
                &rate_plot_path,
                &times_for_fit,
                &rates_f64,
                Some(fitted.as_slice()),
                residuals,
                &format!("Residual Rate Fit (step {})", step_idx + 1),
                "Residual Rate [Hz]",
            )?;
        }

        if let Some(fitted) = &delay_fit_samples_series {
            let residuals = delay_residual_samples_series.as_ref().map(|v| v.as_slice());
            let delay_plot_path = output_dir.join(format!(
                "{}_step{}_res_delay.png",
                base_filename,
                step_idx + 1
            ));
            plot_acel_search_result(
                &delay_plot_path,
                &times_for_fit,
                &delays_samples_f64,
                Some(fitted.as_slice()),
                residuals,
                &format!("Residual Delay Fit (step {})", step_idx + 1),
                "Residual Delay [sample]",
            )?;
        }
    }

    Ok(())
}
