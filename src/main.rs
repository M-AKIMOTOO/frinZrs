use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::process;

use clap::{Parser,CommandFactory};
use num_complex::Complex;

mod args;
mod header;
mod rfi;
mod output;
mod read;
mod fft;
mod analysis;
mod plot;
mod utils;
mod fitting;

use args::Args;
use header::parse_header;
use rfi::parse_rfi_ranges;
use output::{output_header_info, generate_output_names, format_delay_output, format_freq_output};
use read::read_visibility_data;
use fft::{process_fft, process_ifft};
use analysis::analyze_results;
use plot::{delay_plane, frequency_plane, add_plot, cumulate_plot};
use image::ImageBuffer;
use imageproc::filter;

// --- Type Aliases for Clarity ---
type C32 = Complex<f32>;

// --- Main Application Logic ---
fn main() -> Result<(), Box<dyn Error>> {
    let args = match Args::try_parse() {
        Ok(args) => args,
        Err(e) => {
            // If no arguments are provided, print help and exit
            if std::env::args().len() <= 1 {
                let mut cmd = Args::command();
                cmd.print_help().expect("Failed to print help");
                std::process::exit(0);
            } else {
                // For other parsing errors, let clap handle it (prints error and usage)
                e.exit();
            }
        }
    };

    // --- File and Path Setup ---
    let input_path = &args.input;
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let frinz_dir = parent_dir.join("frinZ");
    fs::create_dir_all(&frinz_dir)?;

    let basename = input_path.file_stem().unwrap().to_str().unwrap();
    let label: Vec<&str> = basename.split('_').collect();

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

    let mut cumulate_path: Option<PathBuf> = None;
    if args.cumulate != 0 {
        let path = frinz_dir.join(format!("cumulate/len{}s", args.cumulate));
        fs::create_dir_all(&path)?;
        cumulate_path = Some(path);
    }

    if !args.rfi.is_empty() {
        let rfi_history_path = frinz_dir.join("rfi_history");
        fs::create_dir_all(&rfi_history_path)?;
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
    let pp = header.number_of_sector;
    let mut length = if args.length == 0 { pp } else { args.length };
    let mut loop_count = if (pp - args.skip) / length <= 0 {
        1
    } else if (pp - args.skip) / length <= args.loop_ {
        (pp - args.skip) / length
    } else {
        args.loop_
    };

    if args.cumulate != 0 {
        if args.cumulate >= pp {
            eprintln!("The specified cumulation length, {} s, is more than the observation time, {} s.", args.cumulate, pp);
            process::exit(1);
        }
        length = args.cumulate;
        loop_count = pp / args.cumulate;
    }

    let mut delay_output_str = String::new();
    let mut freq_output_str = String::new();

    let mut cumulate_len: Vec<f32> = Vec::new();
    let mut cumulate_snr: Vec<f32> = Vec::new();
        let mut cumulate_noise: Vec<f32> = Vec::new();

    let mut add_plot_path: Option<PathBuf> = None;
    let mut add_plot_amp: Vec<f32> = Vec::new();
    let mut add_plot_phase: Vec<f32> = Vec::new();
    let mut add_plot_snr: Vec<f32> = Vec::new();
    let mut add_plot_noise: Vec<f32> = Vec::new();
    let mut add_plot_length: Vec<f32> = Vec::new();
    let mut add_plot_len: f32 = 0.0;
    let mut obs_time: Option<chrono::DateTime<chrono::Utc>> = None;
    if args.add_plot {
        let path = frinz_dir.join("add_plot");
        fs::create_dir_all(&path)?;
        add_plot_path = Some(path);
    }

    // --- Main Processing Loop ---
    for l1 in 0..loop_count {
        let current_length = if args.cumulate != 0 { (l1 + 1) * length } else { length };
        let (complex_vec, current_obs_time, effective_integ_time) = read_visibility_data(&mut cursor, &header, current_length, args.skip, l1, args.cumulate != 0)?;
        if l1 == 0 { obs_time = Some(current_obs_time); }

        let mut analysis_results;
        let freq_rate_array;
        let delay_rate_2d_data_comp;

        if args.search {
            // --- SEARCH MODE ---
            // 1. Initial coarse analysis on raw data. This search respects the user-defined window if provided.
            let (coarse_freq_rate_array, padding_length) = process_fft(&complex_vec, current_length, header.fft_point, &rfi_ranges);
            let coarse_delay_rate_2d_data_comp = process_ifft(&coarse_freq_rate_array, header.fft_point, padding_length);
            let coarse_results = analyze_results(&coarse_freq_rate_array, &coarse_delay_rate_2d_data_comp, &header, current_length, effective_integ_time, &current_obs_time, padding_length, &args);

            // 2. Initialize total corrections with the coarse peak values found.
            let mut total_delay_correct = coarse_results.residual_delay;
            let mut total_rate_correct = coarse_results.residual_rate;

            // 3. Iteration loop to refine corrections.
            // For the iterative part, we search in a small, fixed window around the zero-point,
            // because the coarse correction should have moved the peak of interest there.
            // This prevents the search from latching onto other, stronger peaks during refinement.
            let mut iter_args = args.clone();
            iter_args.delay_window = vec![-16.0, 16.0]; // Use a small, fixed window for refinement
            iter_args.rate_window = vec![-0.05, 0.05];   // Use a small, fixed window for refinement

            for _ in 0..args.iter {
                // We pass the *total* correction calculated so far to the analysis functions.
                let mut current_args = iter_args.clone();
                current_args.delay_correct = total_delay_correct;
                current_args.rate_correct = total_rate_correct;

                // Apply the total correction to the original complex data.
                let input_data_2d: Vec<Vec<Complex<f64>>> = complex_vec
                    .chunks(header.fft_point as usize / 2)
                    .map(|chunk| chunk.iter().map(|&c| Complex::new(c.re as f64, c.im as f64)).collect())
                    .collect();

                let temp_complex_vec_2d = fft::apply_phase_correction(
                    &input_data_2d,
                    total_rate_correct,
                    total_delay_correct,
                    effective_integ_time,
                    header.sampling_speed as u32,
                    header.fft_point as u32,
                );
                let temp_complex_vec: Vec<C32> = temp_complex_vec_2d.into_iter().flatten().map(|v| Complex::new(v.re as f32, v.im as f32)).collect();

                // Analyze the corrected data to find the *offset* from the new zero-point.
                let (iter_freq_rate_array, padding_length) = process_fft(&temp_complex_vec, current_length, header.fft_point, &rfi_ranges);
                let iter_delay_rate_2d_data_comp = process_ifft(&iter_freq_rate_array, header.fft_point, padding_length);
                let iter_results = analyze_results(&iter_freq_rate_array, &iter_delay_rate_2d_data_comp, &header, current_length, effective_integ_time, &current_obs_time, padding_length, &current_args);
                
                // Add the found offset to the total correction for the next iteration.
                total_delay_correct += iter_results.delay_offset;
                total_rate_correct += iter_results.rate_offset;
            }

            // 4. Final analysis is complete. The total_delay_correct and total_rate_correct are the refined values.
            // We need to run the analysis one last time to get the final statistics (SNR, etc.) for the corrected data.
            let mut final_args = args.clone();
            final_args.delay_correct = total_delay_correct;
            final_args.rate_correct = total_rate_correct;
            // For the final analysis, we don't need a window, as the peak is at zero.
            final_args.delay_window = Vec::new();
            final_args.rate_window = Vec::new();

            let input_data_2d: Vec<Vec<Complex<f64>>> = complex_vec
                .chunks(header.fft_point as usize / 2)
                .map(|chunk| chunk.iter().map(|&c| Complex::new(c.re as f64, c.im as f64)).collect())
                .collect();

            let final_complex_vec_2d = fft::apply_phase_correction(
                &input_data_2d,
                total_rate_correct,
                total_delay_correct,
                effective_integ_time,
                header.sampling_speed as u32,
                header.fft_point as u32,
            );
            let final_complex_vec: Vec<C32> = final_complex_vec_2d.into_iter().flatten().map(|v| Complex::new(v.re as f32, v.im as f32)).collect();
            let (final_freq_rate_array, padding_length) = process_fft(&final_complex_vec, current_length, header.fft_point, &rfi_ranges);
            let final_delay_rate_2d_data_comp = process_ifft(&final_freq_rate_array, header.fft_point, padding_length);
            
            // The analysis_results will have stats for the corrected data (peak at zero).
            analysis_results = analyze_results(&final_freq_rate_array, &final_delay_rate_2d_data_comp, &header, current_length, effective_integ_time, &current_obs_time, padding_length, &final_args);
            
            // But we must report the *total* corrected delay and rate that we found.
            analysis_results.length_f32 = current_length as f32 * effective_integ_time;
            analysis_results.residual_delay = total_delay_correct;
            analysis_results.residual_rate = total_rate_correct;
            freq_rate_array = final_freq_rate_array;
            delay_rate_2d_data_comp = final_delay_rate_2d_data_comp;
        } else {
            // --- NORMAL MODE ---
            let mut corrected_complex_vec: Vec<C32> = complex_vec.clone();
            if args.delay_correct != 0.0 || args.rate_correct != 0.0 {
                let input_data_2d: Vec<Vec<Complex<f64>>> = complex_vec
                    .chunks(header.fft_point as usize / 2)
                    .map(|chunk| chunk.iter().map(|&c| Complex::new(c.re as f64, c.im as f64)).collect())
                    .collect();

                let corrected_complex_vec_2d = fft::apply_phase_correction(
                    &input_data_2d,
                    args.rate_correct,
                    args.delay_correct,
                    effective_integ_time,
                    header.sampling_speed as u32,
                    header.fft_point as u32,
                );
                corrected_complex_vec = corrected_complex_vec_2d.into_iter().flatten().map(|v| Complex::new(v.re as f32, v.im as f32)).collect();
            }
            let (final_freq_rate_array, padding_length) = process_fft(&corrected_complex_vec, current_length, header.fft_point, &rfi_ranges);
            let final_delay_rate_2d_data_comp = process_ifft(&final_freq_rate_array, header.fft_point, padding_length);
            analysis_results = analyze_results(&final_freq_rate_array, &final_delay_rate_2d_data_comp, &header, current_length, effective_integ_time, &current_obs_time, padding_length, &args);
            analysis_results.length_f32 = (current_length as f32 * effective_integ_time).ceil();
            freq_rate_array = final_freq_rate_array;
            delay_rate_2d_data_comp = final_delay_rate_2d_data_comp;
        }

        // --- Output and Plotting ---
        let base_filename = generate_output_names(&header, &obs_time.as_ref().unwrap(), &label, !rfi_ranges.is_empty(), args.frequency, current_length);

        if !args.frequency {
            let delay_output_line = format_delay_output(&analysis_results, &label);
            if l1 == 0 {
                        let header_str = "".to_string() 
                           + "#************************************************************************************************************************************************************************************\n"
                           + "#      Epoch        Label    Source     Length    Amp      SNR     Phase     Noise-level      Res-Delay     Res-Rate            YAMAGU32-azel            YAMAGU34-azel             MJD      \n"
                           + "#                                        [s]      [%]               [deg]     1-sigma[%]       [sample]       [Hz]      az[deg]  el[deg]  hgt[m]    az[deg]  el[deg]  hgt[m]                \n"
                           + "#************************************************************************************************************************************************************************************\n";
                print!("{}", header_str);
                delay_output_str += &header_str;
            }
            print!("{}\n", delay_output_line);
            delay_output_str += &delay_output_line;

            if args.cumulate != 0 {
                cumulate_len.push(current_length as f32);
                cumulate_snr.push(analysis_results.delay_snr);
                cumulate_noise.push(analysis_results.delay_noise);
            }

            if args.add_plot {
                if l1 > 0 {add_plot_len += current_length as f32;}
                add_plot_amp.push(analysis_results.delay_max_amp * 100.0);
                add_plot_phase.push(analysis_results.delay_phase);
                add_plot_snr.push(analysis_results.delay_snr);
                add_plot_noise.push(analysis_results.delay_noise * 100.0);
                add_plot_length.push(add_plot_len);
            }

            if l1 == loop_count - 1 && args.output {
                 if let Some(path) = &output_path {
                    let output_file_path = path.join(format!("{}.txt", format!("{}_time", base_filename)));
                    fs::write(output_file_path, &delay_output_str)?;
                }
            }
        } else {
            let freq_output_line = format_freq_output(&analysis_results, &label);
            if l1 == 0 {
                let header_str = "".to_string() 
                           + "#******************************************************************************************************************************************************************************************\n"
                           + "#      Epoch        Label    Source     Length    Amp      SNR     Phase     Frequency     Noise-level      Res-Rate            YAMAGU32-azel             YAMAGU34-azel             MJD      \n"
                           + "#                                        [s]      [%]              [deg]       [MHz]       1-sigma[%]        [Hz]        az[deg]  el[deg]  hgt[m]   az[deg]  el[deg]  hgt[m]                \n"
                           + "#******************************************************************************************************************************************************************************************\n";
                print!("{}", header_str);
                freq_output_str += &header_str;
            }
            print!("{}\n", freq_output_line);
            freq_output_str += &freq_output_line;

            if l1 == loop_count - 1 && args.output {
                if let Some(path) = &output_path {
                    let output_file_path = path.join(format!("{}.txt", format!("{}_freq", base_filename)));
                    fs::write(output_file_path, &freq_output_str)?;
                }
            }
        }

        if args.plot && args.cumulate == 0 {
            if let Some(path) = &plot_path {
                let length_label = if args.length == 0 { "0".to_string() } else { args.length.to_string() };
                let plot_dir = if !args.frequency {
                    path.join(format!("time_domain/len{}s", length_label))
                } else {
                    path.join(format!("freq_domain/len{}s", length_label))
                };
                fs::create_dir_all(&plot_dir)?;
                let mut output_filename: PathBuf = PathBuf::new();
                if !args.frequency {
                    output_filename = plot_dir.join(format!("{}_delay_rate_search.png", base_filename));
                } else if args.frequency {
                    output_filename = plot_dir.join(format!("{}_freq_rate_search.png", base_filename));
                }
                
                if !args.frequency {
                    let delay_profile: Vec<(f64, f64)> = analysis_results.delay_range.iter()
                        .zip(analysis_results.visibility.iter())
                        .map(|(&x, &y)| (x as f64, y as f64))
                        .collect();

                    let rate_profile: Vec<(f64, f64)> = analysis_results.rate_range.iter()
                        .zip(analysis_results.delay_rate.iter())
                        .map(|(&x, &y)| (x as f64, y as f64))
                        .collect();

                    let rows = delay_rate_2d_data_comp.shape()[0] as u32;
                    let cols = delay_rate_2d_data_comp.shape()[1] as u32;

                    let mut img = ImageBuffer::new(cols, rows);
                    let max_norm = delay_rate_2d_data_comp.iter().map(|c| c.norm()).fold(0.0f32, |acc, x| acc.max(x));

                    for y in 0..rows {
                        for x in 0..cols {
                            let val = delay_rate_2d_data_comp[[y as usize, x as usize]].norm();
                            let normalized_val = if max_norm > 0.0 { (val / max_norm * 255.0) as u8 } else { 0 };
                            let pixel = image::Luma([normalized_val]);
                            img.put_pixel(x, y, pixel);
                        }
                    }
                    let blurred_img = filter::gaussian_blur_f32(&img, 1.0);

                    let delay_data: Vec<f32> = analysis_results.delay_range.iter().map(|&x| x as f32).collect();
                    let rate_data: Vec<f32> = analysis_results.rate_range.iter().map(|&x| x as f32).collect();

                    let heatmap_func = move |delay: f64, rate: f64| -> f64 {
                        let d_min = delay_data[0] as f64;
                        let d_max = *delay_data.last().unwrap() as f64;
                        let r_min = rate_data[0] as f64;
                        let r_max = *rate_data.last().unwrap() as f64;

                        let x_img = (delay - d_min) / (d_max - d_min) * (cols - 1) as f64;
                        let y_img = (rate - r_min) / (r_max - r_min) * (rows - 1) as f64;

                        let x_floor = x_img.floor() as u32;
                        let y_floor = y_img.floor() as u32;
                        let x_ceil = (x_img.ceil() as u32).min(cols - 1);
                        let y_ceil = (y_img.ceil() as u32).min(rows - 1);

                        let fx = x_img - x_img.floor();
                        let fy = y_img - y_img.floor();

                        let q11 = (blurred_img.get_pixel(x_floor, y_floor)[0] as f64) / 255.0 * (max_norm as f64);
                        let q12 = (blurred_img.get_pixel(x_ceil, y_floor)[0] as f64) / 255.0 * (max_norm as f64);
                        let q21 = (blurred_img.get_pixel(x_floor, y_ceil)[0] as f64) / 255.0 * (max_norm as f64);
                        let q22 = (blurred_img.get_pixel(x_ceil, y_ceil)[0] as f64) / 255.0 * (max_norm as f64);

                        let r1 = q11 * (1.0 - fx) + q12 * fx;
                        let r2 = q21 * (1.0 - fx) + q22 * fx;

                        r1 * (1.0 - fy) + r2 * fy
                    };

                    let stat_keys = vec![
                        "Epoch (UTC)", "Station 1 & 2", "Source", "Length [s]", "Frequency [MHz]",
                        "Peak Amp [%]", "Peak Phs [deg]", "SNR (1 σ [%])", "Delay (residual) [sps]",
                        "Delay (corrected) [sps]", "Rate (residual) [mHz]", "Rate (corrected) [mHz]",
                    ];
                    let stat_vals = vec![
                        analysis_results.yyyydddhhmmss1.to_string(),
                        format!("{} & {}", header.station1_name, header.station2_name),
                        analysis_results.source_name.to_string(),
                        format!("{:.3}", analysis_results.length_f32.ceil()),
                        format!("{:.3}", header.observing_frequency as f32 / 1e6),
                        format!("{:.6}", analysis_results.delay_max_amp * 100.0),
                        format!("{:+.5}", analysis_results.delay_phase),
                        format!("{:.3} ({:.6})", analysis_results.delay_snr, analysis_results.delay_noise * 100.0),
                        format!("{:+.6}", analysis_results.residual_delay),
                        format!("{:+.6}", analysis_results.corrected_delay),
                        format!("{:+.6}", analysis_results.residual_rate * 1000.0),
                        format!("{:+.6}", analysis_results.corrected_rate * 1000.0),
                    ];

                    delay_plane(
                        &delay_profile, &rate_profile, heatmap_func,
                        &stat_keys.iter().map(|s| s.as_ref()).collect::<Vec<&str>>(),
                        &stat_vals.iter().map(|s| s.as_ref()).collect::<Vec<&str>>(),
                        output_filename.to_str().unwrap(),
                        &analysis_results.rate_range, analysis_results.length_f32,
                        effective_integ_time, &args.delay_window, &args.rate_window, max_norm as f64,//, args.cmap_time
                    )?;
                } else {
                    let freq_amp_profile: Vec<(f64, f64)> = analysis_results.freq_range.iter()
                        .zip(analysis_results.freq_rate_spectrum.iter().map(|c| c.norm()))
                        .map(|(&x, y)| (x as f64, y as f64))
                        .collect();

                    let freq_phase_profile: Vec<(f64, f64)> = analysis_results.freq_range.iter()
                        .zip(analysis_results.freq_rate_spectrum.iter().map(|c| c.arg().to_degrees()))
                        .map(|(&x, y)| (x as f64, y as f64))
                        .collect();

                    let rate_profile: Vec<(f64, f64)> = analysis_results.rate_range.iter()
                        .zip(analysis_results.freq_rate.iter())
                        .map(|(&x, &y)| (x as f64, y as f64))
                        .collect();

                    let freq_data: Vec<f32> = analysis_results.freq_range.iter().map(|&x| x as f32).collect();
                    let rate_data: Vec<f32> = analysis_results.rate_range.iter().map(|&x| x as f32).collect();

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

                        let freq_idx = (((freq - f_min) / (f_max - f_min)) * (rows - 1) as f64).round() as usize;
                        let rate_idx = (((rate - r_min) / (r_max - r_min)) * (cols - 1) as f64).round() as usize;

                        if freq_idx < rows && rate_idx < cols {
                            freq_rate_array[[freq_idx, rate_idx]].norm() as f64
                        } else {
                            0.0
                        }
                    };

                    let stat_keys = vec![
                        "Epoch (UTC)", "Station 1 & 2", "Source", "Length [s]", "Frequency [MHz]",
                        "Peak Amp [%]", "Peak Phs [deg]", "Peak Freq [MHz]", "SNR (1 σ [%])",
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
                        format!("{:.3} ({:.6})", analysis_results.freq_snr, analysis_results.freq_noise * 100.0),
                        format!("{:+.6}", analysis_results.residual_rate * 1000.0),
                    ];

                    let max_norm_freq = freq_rate_array.iter().map(|c| c.norm()).fold(0.0f32, |acc, x| acc.max(x));

                    frequency_plane(
                        &freq_amp_profile, &freq_phase_profile, &rate_profile, heatmap_func,
                        &stat_keys.iter().map(|s| s.as_ref()).collect::<Vec<&str>>(),
                        &stat_vals.iter().map(|s| s.as_ref()).collect::<Vec<&str>>(),
                        output_filename.to_str().unwrap(),
                        bw as f64, max_norm_freq as f64,
                    )?;
                }
            }
        }
    }

    if args.cumulate != 0 {
        if let Some(path) = cumulate_path {
            cumulate_plot(
                &cumulate_len,
                &cumulate_snr,
                &path,
                &header,
                &label,
                &obs_time.as_ref().unwrap(),
                args.cumulate,
            )?;
        }
    }

    if args.add_plot {
        if let Some(path) = add_plot_path {
            let base_filename = generate_output_names(&header, &obs_time.unwrap(), &label, false, false, args.length);
            let add_plot_filename = format!("{}_{}", base_filename, header.source_name);
            let add_plot_filepath = path.join(add_plot_filename);
            add_plot(
                &add_plot_filepath.to_str().unwrap(),
                &add_plot_length,
                &add_plot_amp,
                &add_plot_snr,
                &add_plot_phase,
                &add_plot_noise,
                &header.source_name,
                length,
            )?;
        }
    }
    
    Ok(())
}