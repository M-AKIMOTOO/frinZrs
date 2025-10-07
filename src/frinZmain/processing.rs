use std::error::Error;
use std::fs::{self, File};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process;

use crate::analysis::AnalysisResults;
use chrono::{DateTime, Utc};
use image::ImageBuffer;
use imageproc::filter;
use ndarray::Array;
use ndarray::Array2;
use num_complex::Complex;

use crate::analysis::analyze_results;
use crate::args::Args;
use crate::bandpass::{
    apply_bandpass_correction, read_bandpass_file, write_complex_spectrum_binary,
};
use crate::deep_search;
use crate::fft::{self, process_fft, process_ifft};
use crate::header::{parse_header, CorHeader};
use crate::output::{
    format_delay_output, format_freq_output, generate_output_names, output_header_info,
};
use crate::plot::{
    delay_plane, frequency_plane, plot_dynamic_spectrum_freq, plot_dynamic_spectrum_lag,
};
use crate::read::read_visibility_data;
use crate::rfi::parse_rfi_ranges;
use crate::utils::safe_arg;
use memmap2::Mmap;

type C32 = Complex<f32>;

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
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
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

    let mut spectrum_output_path: Option<PathBuf> = None;
    if args.spectrum {
        let path = frinz_dir.join("crossspectrum");
        fs::create_dir_all(&path)?;
        spectrum_output_path = Some(path);
    }

    // --- Read .cor File ---
    let file = File::open(input_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut cursor = Cursor::new(&mmap[..]);

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
        let cor_header_path = frinz_dir.join("header");
        fs::create_dir_all(&cor_header_path)?;
        let header_info_str = output_header_info(&header, &cor_header_path, basename)?;
        if args.header {
            println!("{}", header_info_str);
        }
    }

    // --- Loop and Processing Setup ---
    cursor.set_position(0);
    let (_, obs_time, effective_integ_time) =
        read_visibility_data(&mut cursor, &header, 1, 0, 0, false, pp_flag_ranges)?;
    cursor.set_position(256);

    let pp = header.number_of_sector;
    let mut length = if args.length == 0 { pp } else { args.length };

    let total_obs_time_seconds = pp as f32 * effective_integ_time;
    if args.length != 0 && args.length as f32 > total_obs_time_seconds {
        length = (total_obs_time_seconds / effective_integ_time).ceil() as i32;
    } else if args.length != 0 {
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
                args.cumulate, pp
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

    for l1 in 0..loop_count {
        let current_length = if args.cumulate != 0 {
            (l1 + 1) * length
        } else {
            length
        };
        let (complex_vec, current_obs_time, effective_integ_time) = match read_visibility_data(
            &mut cursor,
            &header,
            current_length,
            args.skip,
            l1,
            args.cumulate != 0,
            pp_flag_ranges,
        ) {
            Ok(data) => data,
            Err(_) => break,
        };

        let is_flagged = time_flag_ranges
            .iter()
            .any(|(start, end)| current_obs_time >= *start && current_obs_time < *end);

        if is_flagged {
            println!(
                "#INFO: Skipping data at {} due to --flagging time range.",
                current_obs_time.format("%Y-%m-%d %H:%M:%S")
            );
            continue;
        }

        let (analysis_results, freq_rate_array, delay_rate_2d_data_comp) = match args
            .search
            .as_deref()
        {
            Some("deep") => {
                let deep_search_result = deep_search::run_deep_search(
                    &complex_vec,
                    &header,
                    current_length,
                    effective_integ_time,
                    &current_obs_time,
                    &obs_time,
                    &rfi_ranges,
                    &bandpass_data,
                    args,
                    pp,
                    args.cpu,
                )?;
                (
                    deep_search_result.analysis_results,
                    deep_search_result.freq_rate_array,
                    deep_search_result.delay_rate_2d_data,
                )
            }
            Some("peak") => {
                let mut total_delay_correct = args.delay_correct;
                let mut total_rate_correct = args.rate_correct;
                let mut analysis_results_mut = None;
                let mut freq_rate_array_mut = None;
                let mut delay_rate_2d_data_comp_mut = None;

                for _ in 0..args.iter {
                    let (iter_results, iter_freq_rate_array, iter_delay_rate_2d_data_comp) =
                        run_analysis_pipeline(
                            &complex_vec,
                            &header,
                            args,
                            Some("peak"),
                            total_delay_correct,
                            total_rate_correct,
                            args.acel_correct,
                            current_length,
                            effective_integ_time,
                            &current_obs_time,
                            &obs_time,
                            &rfi_ranges,
                            &bandpass_data,
                        )?;
                    total_delay_correct += iter_results.delay_offset;
                    total_rate_correct += iter_results.rate_offset;
                    analysis_results_mut = Some(iter_results);
                    freq_rate_array_mut = Some(iter_freq_rate_array);
                    delay_rate_2d_data_comp_mut = Some(iter_delay_rate_2d_data_comp);
                }

                let mut final_analysis_results = analysis_results_mut.unwrap();
                final_analysis_results.length_f32 = current_length as f32 * effective_integ_time;
                final_analysis_results.residual_delay = total_delay_correct;
                final_analysis_results.residual_rate = total_rate_correct;
                final_analysis_results.corrected_delay = args.delay_correct;
                final_analysis_results.corrected_rate = args.rate_correct;
                final_analysis_results.corrected_acel = args.acel_correct;
                (
                    final_analysis_results,
                    freq_rate_array_mut.unwrap(),
                    delay_rate_2d_data_comp_mut.unwrap(),
                )
            }
            _ => {
                // No search or other modes not handled here
                let (mut analysis_results, freq_rate_array, delay_rate_2d_data_comp) =
                    run_analysis_pipeline(
                        &complex_vec,
                        &header,
                        args,
                        None,
                        args.delay_correct,
                        args.rate_correct,
                        args.acel_correct,
                        current_length,
                        effective_integ_time,
                        &current_obs_time,
                        &obs_time,
                        &rfi_ranges,
                        &bandpass_data,
                    )?;
                analysis_results.length_f32 = (current_length as f32 * effective_integ_time).ceil();
                (analysis_results, freq_rate_array, delay_rate_2d_data_comp)
            }
        };

        let label_str: Vec<&str> = label.iter().map(|s| s.as_str()).collect();
        let base_filename = generate_output_names(
            &header,
            &obs_time,
            &label_str,
            !rfi_ranges.is_empty(),
            args.frequency,
            args.bandpass.is_some(),
            current_length,
        );

        if args.spectrum {
            if let Some(path) = &spectrum_output_path {
                let output_file_path = path.join(format!("{}_cross.spec", base_filename));
                write_complex_spectrum_binary(
                    &output_file_path,
                    &analysis_results.freq_rate_spectrum.to_vec(),
                    header.fft_point,
                    1,
                )?;
                println!(
                    "Cross-power spectrum file written to {:?}",
                    output_file_path
                );
            }
        }

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
            let spectrum_array =
                Array::from_shape_vec((time_samples, fft_point_half), complex_vec.clone()).unwrap();
            let output_path_freq = dynamic_spectrum_dir
                .join(format!("{}_dynamic_spectrum_frequency.png", base_filename));
            plot_dynamic_spectrum_freq(
                output_path_freq.to_str().unwrap(),
                &spectrum_array,
                &header,
                &current_obs_time,
                current_length,
                effective_integ_time,
            )?;
            let mut lag_data = Array::zeros((time_samples, header.fft_point as usize));
            let fft_point_usize = header.fft_point as usize;
            for (i, row) in spectrum_array.rows().into_iter().enumerate() {
                let shifted_out =
                    fft::perform_ifft_on_vec(row.as_slice().unwrap(), fft_point_usize);
                for (j, val) in shifted_out.iter().enumerate() {
                    lag_data[[i, j]] = val.norm();
                }
            }
            let output_path_lag = dynamic_spectrum_dir
                .join(format!("{}_dynamic_spectrum_time_lag.png", base_filename));
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
                    + "#*****************************************************************************************************************************************************************************************
"
                    + "#      Epoch        Label    Source     Length    Amp      SNR     Phase     Noise-level      Res-Delay     Res-Rate            YAMAGU32-azel            YAMAGU34-azel             MJD    
"
                    + "#                                        [s]      [%]               [deg]     1-sigma[%]       [sample]       [Hz]      az[deg]  el[deg]  hgt[m]    az[deg]  el[deg]  hgt[m]              
"
                    + "#*****************************************************************************************************************************************************************************************";
                print!("{}\n", header_str);
                delay_output_str += &format!("{}\n", header_str);
            }
            print!("{}\n", delay_output_line);
            delay_output_str += &format!("{}\n", delay_output_line);

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
                    + "#*******************************************************************************************************************************************************************************************\n"
                    + "#      Epoch        Label    Source     Length    Amp      SNR     Phase     Frequency     Noise-level      Res-Rate            YAMAGU32-azel             YAMAGU34-azel             MJD     \n"
                    + "#                                        [s]      [%]              [deg]       [MHz]       1-sigma[%]        [Hz]        az[deg]  el[deg]  hgt[m]   az[deg]  el[deg]  hgt[m]                \n"
                    + "#*******************************************************************************************************************************************************************************************";
                print!("{}\n", header_str);
                freq_output_str += &format!("{}\n", header_str);
            }
            print!("{}\n", freq_output_line);
            freq_output_str += &format!("{}\n", freq_output_line);

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
        }

        if args.plot && args.cumulate == 0 {
            if let Some(path) = &plot_path {
                let length_label = if args.length == 0 {
                    "0".to_string()
                } else {
                    args.length.to_string()
                };
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
                            let normalized_val = if max_norm > 0.0 {
                                (val / max_norm * 255.0) as u8
                            } else {
                                0
                            };
                            img.put_pixel(x, y, image::Luma([normalized_val]));
                        }
                    }
                    let blurred_img = filter::gaussian_blur_f32(&img, 1.0);
                    let delay_data: Vec<f32> = analysis_results
                        .delay_range
                        .iter()
                        .map(|&x| x as f32)
                        .collect();
                    let rate_data: Vec<f32> = analysis_results
                        .rate_range
                        .iter()
                        .map(|&x| x as f32)
                        .collect();
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

                    let (length_key, length_val) = if !pp_flag_ranges.is_empty() {
                        let flag_str = pp_flag_ranges
                            .iter()
                            .map(|(s, e)| format!("{}-{}", s, e))
                            .collect::<Vec<String>>()
                            .join(", ");
                        (
                            "Length (flag) [s]".to_string(),
                            format!("{:.3} ({})", analysis_results.length_f32.ceil(), flag_str),
                        )
                    } else {
                        (
                            "Length [s]".to_string(),
                            format!("{:.3}", analysis_results.length_f32.ceil()),
                        )
                    };

                    let stat_keys = vec![
                        "Epoch (UTC)",
                        "Station 1 & 2",
                        "Source",
                        &length_key,
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
                        length_val,
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
                            analysis_results
                                .freq_rate_spectrum
                                .iter()
                                .map(|c| safe_arg(c).to_degrees()),
                        )
                        .map(|(&x, y)| (x as f64, y as f64))
                        .collect();
                    let rate_profile: Vec<(f64, f64)> = analysis_results
                        .rate_range
                        .iter()
                        .zip(analysis_results.freq_rate.iter())
                        .map(|(&x, &y)| (x as f64, y as f64))
                        .collect();
                    let freq_data: Vec<f32> = analysis_results
                        .freq_range
                        .iter()
                        .map(|&x| x as f32)
                        .collect();
                    let rate_data: Vec<f32> = analysis_results
                        .rate_range
                        .iter()
                        .map(|&x| x as f32)
                        .collect();
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
                        let freq_idx = (((freq - f_min) / (f_max - f_min)) * (rows - 1) as f64)
                            .round() as usize;
                        let rate_idx = (((rate - r_min) / (r_max - r_min)) * (cols - 1) as f64)
                            .round() as usize;
                        if freq_idx < rows && rate_idx < cols {
                            freq_rate_array[[freq_idx, rate_idx]].norm() as f64
                        } else {
                            0.0
                        }
                    };

                    let (length_key, length_val) = if !pp_flag_ranges.is_empty() {
                        let flag_str = pp_flag_ranges
                            .iter()
                            .map(|(s, e)| format!("{}-{}", s, e))
                            .collect::<Vec<String>>()
                            .join(", ");
                        (
                            "Length (flag) [s]".to_string(),
                            format!("{:.3} ({})", analysis_results.length_f32.ceil(), flag_str),
                        )
                    } else {
                        (
                            "Length [s]".to_string(),
                            format!("{:.3}", analysis_results.length_f32.ceil()),
                        )
                    };

                    let (freq_key, freq_val) = if !args.rfi.is_empty() {
                        let rfi_str = args
                            .rfi
                            .iter()
                            .map(|s| s.replace(',', "-"))
                            .collect::<Vec<String>>()
                            .join(", ");
                        (
                            "Frequency (RFI) [MHz]".to_string(),
                            format!(
                                "{:.3} ({})",
                                header.observing_frequency as f32 / 1e6,
                                rfi_str
                            ),
                        )
                    } else {
                        (
                            "Frequency [MHz]".to_string(),
                            format!("{:.3}", header.observing_frequency as f32 / 1e6),
                        )
                    };

                    let stat_keys = vec![
                        "Epoch (UTC)",
                        "Station 1 & 2",
                        "Source",
                        &length_key,
                        &freq_key,
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
                        length_val,
                        freq_val,
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
                    let max_norm_freq = freq_rate_array
                        .iter()
                        .map(|c| c.norm())
                        .fold(0.0f32, |acc, x| acc.max(x));
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
        obs_time,
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

fn run_analysis_pipeline(
    complex_vec: &[C32],
    header: &CorHeader,
    base_args: &Args,
    search_mode: Option<&str>,
    delay_correct: f32,
    rate_correct: f32,
    acel_correct: f32,
    current_length: i32,
    effective_integ_time: f32,
    current_obs_time: &DateTime<Utc>,
    obs_time: &DateTime<Utc>,
    rfi_ranges: &[(usize, usize)],
    bandpass_data: &Option<Vec<C32>>,
) -> Result<(AnalysisResults, Array2<C32>, Array2<C32>), Box<dyn Error>> {
    let mut temp_args = base_args.clone();
    temp_args.delay_correct = delay_correct;
    temp_args.rate_correct = rate_correct;
    temp_args.acel_correct = acel_correct;

    let corrected_complex_vec =
        if delay_correct != 0.0 || rate_correct != 0.0 || acel_correct != 0.0 {
            let input_data_2d: Vec<Vec<Complex<f64>>> = complex_vec
                .chunks(header.fft_point as usize / 2)
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|&c| Complex::new(c.re as f64, c.im as f64))
                        .collect()
                })
                .collect();
            let start_time_offset_sec = (*current_obs_time - *obs_time).num_seconds() as f32;
            let corrected_complex_vec_2d = fft::apply_phase_correction(
                &input_data_2d,
                rate_correct,
                delay_correct,
                acel_correct,
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
        } else {
            complex_vec.to_vec()
        };

    let (mut freq_rate_array, padding_length) = process_fft(
        &corrected_complex_vec,
        current_length,
        header.fft_point,
        header.sampling_speed,
        rfi_ranges,
        base_args.rate_padding,
    );

    if let Some(bp_data) = &bandpass_data {
        apply_bandpass_correction(&mut freq_rate_array, bp_data);
    }

    let delay_rate_2d_data_comp = process_ifft(&freq_rate_array, header.fft_point, padding_length);

    let analysis_results = analyze_results(
        &freq_rate_array,
        &delay_rate_2d_data_comp,
        &header,
        current_length,
        effective_integ_time,
        &current_obs_time,
        padding_length,
        &temp_args,
        search_mode,
    );

    Ok((analysis_results, freq_rate_array, delay_rate_2d_data_comp))
}
