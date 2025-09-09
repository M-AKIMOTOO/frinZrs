use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use num_complex::Complex;
use ndarray::Array;

use crate::args::Args;
use crate::fft::{apply_phase_correction, process_fft, process_ifft};
use crate::fitting;
use crate::header::{parse_header, CorHeader};
use crate::plot::plot_acel_search_result;
use crate::read::read_visibility_data;
use crate::utils::{unwrap_phase_radians, safe_arg};

type C32 = Complex<f32>;

struct VisibilityDataPoint {
    complex_vec: Vec<C32>,
    obs_time: DateTime<Utc>,
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
        let (complex_vec, current_obs_time, _eff_integ_time) =
            match read_visibility_data(cursor, header, args.length, args.skip, loop_idx, false, pp_flag_ranges) {
                Ok(data) => data,
                Err(_) => break, // Stop if we can't read more data
            };

        if complex_vec.is_empty() {
            break; // Stop if no data was read
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
) -> Result<(Vec<f64>, Vec<f32>), Box<dyn Error>> {
    let mut phases_collected: Vec<f32> = Vec::new();
    let mut times_collected: Vec<f64> = Vec::new();

    for data_point in collected_data {
        let start_time_offset_sec =
            (data_point.obs_time - obs_time_start).num_seconds() as f32;

        let corrected_complex_vec = {
            let input_data_2d: Vec<Vec<Complex<f64>>> = data_point.complex_vec
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
            process_fft(&corrected_complex_vec, actual_length as i32, header.fft_point, header.sampling_speed, &[], args.rate_padding);
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
        let phase_rad = safe_arg(&delay_rate_array_comp[[peak_rate_idx, peak_delay_idx]]);

        phases_collected.push(phase_rad);
        times_collected.push(start_time_offset_sec as f64);
    }

    unwrap_phase_radians(&mut phases_collected);
    Ok((times_collected, phases_collected))
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

    // Get effective_integ_time from the first sector
    cursor.set_position(0);
    let (_, _, effective_integ_time) = read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;

    // Helper function to write data
    let write_fit_data = |path: &Path, times: &[f64], phases: &[f32], coeffs: Option<&[f64]>| -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        writeln!(writer, "# Time [s]   Unwrapped Phase [rad]")?;
        if let Some(c) = coeffs {
            if c.len() == 3 { // Quadratic
                writeln!(writer, "# Fitted: y = {:.6e} * x^2 + {:.6e} * x + {:.6e}", c[2], c[1], c[0])?;
                writeln!(writer, "# Corrected Acel (Hz/s): {:.6e} (from x^2 / PI)", c[2] / std::f64::consts::PI)?;
                writeln!(writer, "# Corrected Rate (Hz): {:.6e} (from x / (2 * PI))", c[1] / (2.0 * std::f64::consts::PI))?;
            } else if c.len() == 2 { // Linear
                writeln!(writer, "# Fitted: y = {:.6e} * x + {:.6e}", c[1], c[0])?;
                writeln!(writer, "# Corrected Rate: {:.6e} (from x / (2 * PI))", c[1] / (2.0 * std::f64::consts::PI))?;
            }
        }
        for i in 0..times.len() {
            writeln!(writer, "{:.0} {:.6}", times[i], phases[i])?;
        }
        Ok(())
    };

    // Initialize obs_time_start once before the loop
    cursor.set_position(256); // Reset cursor for first data read
    let (_, first_obs_time, _) = read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;
    let obs_time_start = first_obs_time;

    // Collect all visibility data once to avoid re-reading
    let collected_data = collect_visibility_data(&mut cursor, &header, args, time_flag_ranges, pp_flag_ranges)?;

    let mut generated_txt_files: Vec<PathBuf> = Vec::new();

    for (step_idx, &degree) in acel_search_degrees.iter().enumerate() {
        println!("Step {}: Fitting with degree {}", step_idx + 1, degree);

        // Get phases from the pre-collected data with current corrections
        let (times_for_fit, phases_for_fit) = get_phases_from_collected_data(
            &collected_data,
            &header,
            args,
            effective_integ_time,
            obs_time_start,
            total_rate_correct,
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
                let quad_path = output_dir.join(format!("{}_step{}_quadric.txt", base_filename, step_idx + 1));
                if let Ok(coeffs) = fitting::fit_polynomial_least_squares(&times_for_fit, &phases_f64, 2) {
                    println!("  Quad fit: x^2={:.6e}, x={:.6e}, c={:.6e}", coeffs[2], coeffs[1], coeffs[0]);
                    total_acel_correct += (coeffs[2] / std::f64::consts::PI) as f32;
                    total_rate_correct += (coeffs[1] / (2.0 * std::f64::consts::PI)) as f32;
                    write_fit_data(&quad_path, &times_for_fit, &phases_for_fit, Some(&coeffs))?;
                    generated_txt_files.push(quad_path.clone());
                } else {
                    eprintln!("Warning: Quadratic fitting failed. Skipping acel and quad-rate update for this step.");
                    write_fit_data(&quad_path, &times_for_fit, &phases_for_fit, None)?;
                    generated_txt_files.push(quad_path.clone());
                }
            }
            1 => { // Linear Fit
                let linear_path = output_dir.join(format!("{}_step{}_linear.txt", base_filename, step_idx + 1));
                if let Ok((slope, intercept)) = fitting::fit_linear_least_squares(&times_for_fit, &phases_f64) {
                    write_fit_data(&linear_path, &times_for_fit, &phases_for_fit, Some(&vec![intercept, slope]))?;
                    println!("  Linear fit: m={:.6e}", slope);
                    total_rate_correct += (slope / (2.0 * std::f64::consts::PI)) as f32;
                    generated_txt_files.push(linear_path.clone());
                } else {
                    eprintln!("Warning: Linear fitting failed. Skipping linear-rate update for this step.");
                    write_fit_data(&linear_path, &times_for_fit, &phases_for_fit, None)?;
                    generated_txt_files.push(linear_path.clone());
                }
            }
            _ => {
                eprintln!("Error: Unsupported fitting degree {}. Skipping this step.", degree);
            }
        }

        println!("  Updated acel: {:.9e} (Hz/s), Updated rate: {:.9e} (Hz)", total_acel_correct, total_rate_correct);
        println!("  Copypaste: --acel {:.18}  --rate {:.15}", total_acel_correct, total_rate_correct);
    }

    for txt_path in generated_txt_files {
        let mut png_path = txt_path.clone();
        png_path.set_extension("png");
        if let Err(e) = plot_acel_search_result(txt_path.to_str().unwrap(), png_path.to_str().unwrap()) {
            eprintln!("Error plotting acel search result for {:?}: {}", txt_path, e);
        }
    }

    Ok(())
}