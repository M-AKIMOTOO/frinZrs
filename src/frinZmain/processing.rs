use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write, BufWriter};
use std::path::{Path, PathBuf};
use std::process;

use chrono::{DateTime, Utc};
use image::ImageBuffer;
use imageproc::filter;
use rustfft::FftPlanner;
use num_complex::Complex;
use ndarray::Array;

use crate::args::Args;
use crate::analysis::{analyze_results, AnalysisResults};
use crate::bandpass::{apply_bandpass_correction, read_bandpass_file, write_complex_spectrum_binary};
use crate::deep_search;
use crate::fft::{self, apply_phase_correction, process_fft, process_ifft};
use crate::header::{parse_header, CorHeader};
use crate::output::{format_delay_output, format_freq_output, generate_output_names, output_header_info};
use crate::plot::{delay_plane, frequency_plane, plot_dynamic_spectrum_freq, plot_dynamic_spectrum_lag};
use crate::read::read_visibility_data;
use crate::rfi::parse_rfi_ranges;
use crate::C32;
use crate::utils::safe_arg;

/// Holds the results of processing a single .cor file, needed for subsequent plotting.
pub struct ProcessResult {
    pub header: CorHeader,
    pub label: Vec<String>,
    pub obs_time: chrono::DateTime<Utc>,
    pub length_arg: i32,
    pub cumulate_len: Vec<f32>,
    pub cumulate_snr: Vec<f32>,
    pub add_plot_times: Vec<DateTime<Utc>>,
    pub add_plot_amp: Vec<f32>,
    pub add_plot_snr: Vec<f32>,
    pub add_plot_phase: Vec<f32>,
    pub add_plot_noise: Vec<f32>,
    pub add_plot_res_delay: Vec<f32>,
    pub add_plot_res_rate: Vec<f32>,
}


pub fn process_cor_file(
    input_path: &Path,
    args: &Args,
    flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
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
        let path = frinz_dir.join("bptable");
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

    let mut spectrum_path: Option<PathBuf> = None;
    if args.spectrum {
        let path = frinz_dir.join("crossspectrum");
        fs::create_dir_all(&path)?;
        spectrum_path = Some(path);
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
    let mut add_plot_res_delay: Vec<f32> = Vec::new();
    let mut add_plot_res_rate: Vec<f32> = Vec::new();
    let mut obs_time: Option<chrono::DateTime<chrono::Utc>> = None;

    // --- Main Processing Loop ---
    for l1 in 0..loop_count {
        let current_length = if args.cumulate != 0 { (l1 + 1) * length } else { length };
        let (complex_vec, current_obs_time, effective_integ_time) = match read_visibility_data(
            &mut cursor,
            &header,
            current_length,
            args.skip,
            l1,
            args.cumulate != 0,
        ) {
            Ok(data) => data,
            Err(_) => break, // Stop if we can't read more data
        };

        // Skip processing if the observation time falls within a flagged range
        let is_flagged = flag_ranges
            .iter()
            .any(|(start, end)| current_obs_time >= *start && current_obs_time < *end);

        if is_flagged {
            println!(
                "#INFO: Skipping data at {} due to --flag-time range.",
                current_obs_time.format("%Y-%m-%d %H:%M:%S")
            );
            continue;
        }

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
                    header.sampling_speed,
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
            analysis_results.corrected_delay = args.delay_correct;
            analysis_results.corrected_rate = args.rate_correct;
            analysis_results.corrected_acel = args.acel_correct;
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
                header.sampling_speed,
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
                let output_file_path = path.join(format!("{}_bptable.bin", base_filename));
                write_complex_spectrum_binary(
                    &output_file_path,
                    &analysis_results.freq_rate_spectrum.to_vec(),
                    header.fft_point,
                    0,
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
                    add_plot_res_delay,
                    add_plot_res_rate,
                });
            }
        }

        if args.dynamic_spectrum {
            let dynamic_spectrum_dir = frinz_dir.join("dynamic_spectrum");
            fs::create_dir_all(&dynamic_spectrum_dir)?;

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

            let fft_point_half = (header.fft_point / 2) as usize;
            let time_samples = complex_vec.len() / fft_point_half;
            let spectrum_array = Array::from_shape_vec((time_samples, fft_point_half), complex_vec.clone()).unwrap();
            
            let output_path_freq = dynamic_spectrum_dir.join(format!("{}_dynamic_spectrum_frequency.png", base_filename));
            plot_dynamic_spectrum_freq(
                output_path_freq.to_str().unwrap(),
                &spectrum_array,
                &header,
                &current_obs_time,
                current_length,
                effective_integ_time,
            )?;

            let mut lag_data = Array::zeros((time_samples, header.fft_point as usize));
            let mut planner = FftPlanner::new();
            let ifft = planner.plan_fft_inverse(header.fft_point as usize);
            let fft_point_usize = header.fft_point as usize;

            for (i, row) in spectrum_array.rows().into_iter().enumerate() {
                let mut ifft_input = vec![C32::new(0.0, 0.0); fft_point_usize];
                ifft_input[..fft_point_half].copy_from_slice(&row.to_vec());
                
                ifft.process(&mut ifft_input);

                let mut shifted_out = vec![C32::new(0.0, 0.0); fft_point_usize];
                let (first_half, second_half) = ifft_input.split_at(fft_point_half);
                shifted_out[..fft_point_half].copy_from_slice(second_half);
                shifted_out[fft_point_half..].copy_from_slice(first_half);

                for val in &mut shifted_out {
                    *val /= header.fft_point as f32;
                }

                shifted_out.reverse();

                for (j, val) in shifted_out.iter().enumerate() {
                    lag_data[[i, j]] = val.norm();
                }
            }

            let output_path_lag = dynamic_spectrum_dir.join(format!("{}_dynamic_spectrum_time_lag.png", base_filename));
            plot_dynamic_spectrum_lag(
                output_path_lag.to_str().unwrap(),
                &lag_data,
                &header,
                &current_obs_time,
                current_length,
                effective_integ_time,
            )?;
        }

        if !args.frequency {
            let delay_output_line = format_delay_output(&analysis_results, &label_str, args.length);
            if l1 == 0 {
                let header_str = "".to_string()
                    + "#*******************************************************************************************************************************************************************************************\n"
                    + "#      Epoch        Label    Source     Length    Amp      SNR     Phase     Noise-level      Res-Delay     Res-Rate            YAMAGU32-azel            YAMAGU34-azel             MJD      \n"
                    + "#                                        [s]      [%]               [deg]     1-sigma[%]       [sample]       [Hz]      az[deg]  el[deg]  hgt[m]    az[deg]  el[deg]  hgt[m]                \n"
                    + "#*******************************************************************************************************************************************************************************************";
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
                add_plot_res_delay.push(analysis_results.residual_delay);
                add_plot_res_rate.push(analysis_results.residual_rate);
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
                    + "#      Epoch        Label    Source     Length    Amp      SNR     Phase     Frequency     Noise-level      Res-Rate            YAMAGU32-azel             YAMAGU34-azel             MJD    \n"
                    + "#                                        [s]      [%]              [deg]       [MHz]       1-sigma[%]        [Hz]        az[deg]  el[deg]  hgt[m]   az[deg]  el[deg]  hgt[m]               \n"
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

            if args.add_plot {
                add_plot_times.push(current_obs_time);
                add_plot_amp.push(analysis_results.freq_max_amp * 100.0);
                add_plot_snr.push(analysis_results.freq_snr);
                add_plot_phase.push(analysis_results.freq_phase);
                add_plot_noise.push(analysis_results.freq_noise * 100.0);
                add_plot_res_delay.push(analysis_results.residual_delay);
                add_plot_res_rate.push(analysis_results.residual_rate);
            }

            if args.spectrum {
                if let Some(path) = &spectrum_path {
                    let output_file_path = path.join(format!("{}_cross.spec", base_filename));
                    write_complex_spectrum_binary(
                        &output_file_path,
                        &analysis_results.freq_rate_spectrum.to_vec(),
                        header.fft_point,
                        1,
                    )?;
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
                            analysis_results.freq_rate_spectrum.iter().map(|c| safe_arg(c).to_degrees()),
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
        add_plot_res_delay,
        add_plot_res_rate,
    })
}
