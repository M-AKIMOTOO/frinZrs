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
use crate::scan_correct;
use crate::utils::safe_arg;
use memmap2::Mmap;

type C32 = Complex<f32>;

fn rebin_complex_rows(
    data: &[C32],
    rows: usize,
    original_cols: usize,
    target_cols: usize,
) -> Vec<C32> {
    if rows == 0
        || original_cols == 0
        || target_cols == 0
        || original_cols == target_cols
        || target_cols > original_cols
        || original_cols % target_cols != 0
    {
        return data.to_vec();
    }

    let group = original_cols / target_cols;
    let mut rebinned = Vec::with_capacity(rows.checked_mul(target_cols).unwrap_or_default());

    for row_idx in 0..rows {
        let row_start = row_idx * original_cols;
        for target_idx in 0..target_cols {
            let mut sum = C32::new(0.0, 0.0);
            for offset in 0..group {
                sum += data[row_start + target_idx * group + offset];
            }
            rebinned.push(sum / group as f32);
        }
    }

    rebinned
}

fn pad_time_rows_to_power_of_two(data: &mut Vec<C32>, current_rows: i32, row_width: usize) -> i32 {
    if current_rows <= 0 || row_width == 0 {
        return current_rows;
    }
    let target_rows = if current_rows <= 1 {
        1
    } else {
        (current_rows as u32).next_power_of_two() as i32
    };

    if target_rows > current_rows {
        let additional_samples = (target_rows - current_rows) as usize * row_width;
        data.extend(std::iter::repeat(C32::new(0.0, 0.0)).take(additional_samples));
    }

    target_rows
}

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
    pub add_plot_complex: Vec<Complex<f32>>,
}

pub fn process_cor_file(
    input_path: &Path,
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
    suppress_output: bool,
) -> Result<ProcessResult, Box<dyn Error>> {
    // --- File and Path Setup ---
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let frinz_dir = parent_dir.join("frinZ");
    fs::create_dir_all(&frinz_dir)?;

    let basename = input_path.file_stem().unwrap().to_str().unwrap();
    let mut label: Vec<String> = basename.split('_').map(String::from).collect();
    if label.len() > 3 {
        let tail = label[3..].join("_");
        label.truncate(3);
        label.push(tail);
    }

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
    let original_fft_point = header.fft_point;

    let mut effective_fft_point = original_fft_point;
    if let Some(requested_fft_point) = args.fft_rebin {
        if requested_fft_point <= 0 {
            eprintln!("Error: --fft-rebin には正の値を指定してください。");
            process::exit(1);
        }
        if requested_fft_point % 2 != 0 {
            eprintln!("Error: --fft-rebin は偶数である必要があります。");
            process::exit(1);
        }
        if requested_fft_point > original_fft_point {
            eprintln!(
                "Error: --fft-rebin ({}) はヘッダーの FFT 点数 ({}) を超えています。",
                requested_fft_point, original_fft_point
            );
            process::exit(1);
        }

        let original_half = (original_fft_point / 2) as usize;
        let requested_half = (requested_fft_point / 2) as usize;
        if requested_half == 0 || original_half % requested_half != 0 {
            eprintln!(
                "Error: --fft-rebin ({}) は元のチャンネル数 ({}) を整数分割できません。",
                requested_fft_point, original_fft_point
            );
            process::exit(1);
        }

        effective_fft_point = requested_fft_point;
    }

    let bw = header.sampling_speed as f32 / 2.0 / 1_000_000.0;
    let rbw = bw / effective_fft_point as f32 * 2.0;

    // --- RFI Handling ---
    let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw)?;

    // --- Bandpass Handling ---
    let mut bandpass_data = if let Some(bp_path) = &args.bandpass {
        Some(read_bandpass_file(bp_path)?)
    } else {
        None
    };
    if effective_fft_point != original_fft_point {
        let original_half = (original_fft_point / 2) as usize;
        let target_half = (effective_fft_point / 2) as usize;
        if let Some(bp) = bandpass_data.as_mut() {
            if bp.len() == original_half {
                let rebinned = rebin_complex_rows(bp, 1, original_half, target_half);
                *bp = rebinned;
            } else if bp.len() != target_half {
                eprintln!(
                    "#WARN: バンドパスデータのチャンネル数 ({}) が FFT リビン後のチャンネル数 ({}) と一致しません。補正をスキップします。",
                    bp.len(),
                    target_half
                );
                *bp = Vec::new();
            }
        }
    }

    let mut processing_header = header.clone();
    processing_header.fft_point = effective_fft_point;

    let scan_corrections = if let Some(path) = &args.scan_correct {
        Some(scan_correct::parse_scan_correct_file(path)?)
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
    let (_, file_start_time, effective_integ_time) =
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
    let mut add_plot_complex: Vec<Complex<f32>> = Vec::new();

    let mut prev_deep_solution: Option<(f32, f32)> = None;
    let mut first_output_basename: Option<String> = None;

    for l1 in 0..loop_count {
        let requested_length = if args.cumulate != 0 {
            (l1 + 1) * length
        } else {
            length
        };
        let (mut complex_vec, current_obs_time, effective_integ_time) = match read_visibility_data(
            &mut cursor,
            &header,
            requested_length,
            args.skip,
            l1,
            args.cumulate != 0,
            pp_flag_ranges,
        ) {
            Ok(data) => data,
            Err(_) => break,
        };

        let original_fft_half = (header.fft_point / 2) as usize;
        if original_fft_half == 0 {
            eprintln!("#ERROR: FFT point が不正です (0)。");
            break;
        }

        if complex_vec.len() % original_fft_half != 0 {
            eprintln!(
                "#ERROR: 読み込んだデータ長 ({}) が FFT チャンネル数 ({}) の整数倍ではありません。",
                complex_vec.len(),
                original_fft_half
            );
            break;
        }

        let actual_length = (complex_vec.len() / original_fft_half) as i32;
        if actual_length == 0 {
            eprintln!(
                "#INFO: skip/length の指定により読み取れるセクターが残っていないため、処理を終了します。"
            );
            break;
        }

        let physical_length = actual_length;

        if effective_fft_point != header.fft_point {
            let target_fft_half = (effective_fft_point / 2) as usize;
            complex_vec = rebin_complex_rows(
                &complex_vec,
                actual_length as usize,
                original_fft_half,
                target_fft_half,
            );
        }

        let fft_point_half_used = (effective_fft_point / 2) as usize;
        if complex_vec.len() != actual_length as usize * fft_point_half_used {
            eprintln!(
                "#ERROR: FFT リビン処理後のデータ長 ({}) が期待値 ({}) と一致しません。",
                complex_vec.len(),
                actual_length as usize * fft_point_half_used
            );
            break;
        }

        let current_length =
            pad_time_rows_to_power_of_two(&mut complex_vec, actual_length, fft_point_half_used);

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

        let (delay_correct_to_use, rate_correct_to_use) =
            if let Some(corrections) = &scan_corrections {
                if let Some((delay, rate)) =
                    scan_correct::find_correction_for_time(corrections, &current_obs_time)
                {
                    (delay, rate)
                } else {
                    (args.delay_correct, args.rate_correct)
                }
            } else {
                (args.delay_correct, args.rate_correct)
            };

        let primary_search_mode = args.primary_search_mode();

        let (mut analysis_results, freq_rate_array, delay_rate_2d_data_comp) =
            match primary_search_mode {
                Some("deep") => {
                    let mut deep_search_result = deep_search::run_deep_search(
                        &complex_vec,
                        &processing_header,
                        current_length,
                        physical_length,
                        effective_integ_time,
                        &current_obs_time,
                        &file_start_time,
                        &rfi_ranges,
                        &bandpass_data,
                        args,
                        pp,
                        args.cpu,
                        prev_deep_solution,
                    )?;
                    deep_search_result.analysis_results.residual_delay -= args.delay_correct;
                    deep_search_result.analysis_results.residual_rate -= args.rate_correct;
                    deep_search_result.analysis_results.corrected_delay =
                        args.delay_correct + deep_search_result.analysis_results.residual_delay;
                    deep_search_result.analysis_results.corrected_rate =
                        args.rate_correct + deep_search_result.analysis_results.residual_rate;
                    let result_tuple = (
                        deep_search_result.analysis_results,
                        deep_search_result.freq_rate_array,
                        deep_search_result.delay_rate_2d_data,
                    );

                    prev_deep_solution = Some((
                        result_tuple.0.residual_delay + args.delay_correct,
                        result_tuple.0.residual_rate + args.rate_correct,
                    ));

                    result_tuple
                }
                Some("peak") => {
                    let mut total_delay_correct = args.delay_correct;
                    let mut total_rate_correct = args.rate_correct;

                    for _ in 0..args.iter {
                        let (iter_results, _, _) = run_analysis_pipeline(
                            &complex_vec,
                            &processing_header,
                            args,
                            Some("peak"),
                            total_delay_correct,
                            total_rate_correct,
                            args.acel_correct,
                            current_length,
                            physical_length,
                            effective_integ_time,
                            &current_obs_time,
                            &file_start_time,
                            &rfi_ranges,
                            &bandpass_data,
                            effective_fft_point,
                        )?;
                        total_delay_correct += iter_results.delay_offset;
                        total_rate_correct += iter_results.rate_offset;
                    }

                    let (mut final_analysis_results, final_freq_rate_array, final_delay_rate_array) =
                        run_analysis_pipeline(
                            &complex_vec,
                            &processing_header,
                            args,
                            Some("peak"),
                            total_delay_correct,
                            total_rate_correct,
                            args.acel_correct,
                            current_length,
                            physical_length,
                            effective_integ_time,
                            &current_obs_time,
                            &file_start_time,
                            &rfi_ranges,
                            &bandpass_data,
                            effective_fft_point,
                        )?;

                    final_analysis_results.length_f32 =
                        physical_length as f32 * effective_integ_time;
                    final_analysis_results.corrected_delay = total_delay_correct;
                    final_analysis_results.corrected_rate = total_rate_correct;
                    final_analysis_results.corrected_acel = args.acel_correct;
                    final_analysis_results.residual_delay =
                        total_delay_correct - args.delay_correct;
                    final_analysis_results.residual_rate = total_rate_correct - args.rate_correct;
                    (
                        final_analysis_results,
                        final_freq_rate_array,
                        final_delay_rate_array,
                    )
                }
                _ => {
                    // No search or other modes not handled here
                    let (mut analysis_results, freq_rate_array, delay_rate_2d_data_comp) =
                        run_analysis_pipeline(
                            &complex_vec,
                            &processing_header,
                            args,
                            None,
                            delay_correct_to_use,
                            rate_correct_to_use,
                            args.acel_correct,
                            current_length,
                            physical_length,
                            effective_integ_time,
                            &current_obs_time,
                            &file_start_time,
                            &rfi_ranges,
                            &bandpass_data,
                            effective_fft_point,
                        )?;
                    analysis_results.length_f32 =
                        (physical_length as f32 * effective_integ_time).ceil();
                    (analysis_results, freq_rate_array, delay_rate_2d_data_comp)
                }
            };

        analysis_results.length_f32 = physical_length as f32 * effective_integ_time;

        let label_str: Vec<&str> = label.iter().map(|s| s.as_str()).collect();
        let filename_length = if args.length == 0 {
            physical_length
        } else {
            args.length
        };
        let base_filename = generate_output_names(
            &processing_header,
            &current_obs_time,
            &label_str,
            !rfi_ranges.is_empty(),
            args.frequency,
            args.bandpass.is_some(),
            filename_length,
        );
        if first_output_basename.is_none() {
            first_output_basename = Some(base_filename.clone());
        }

        if args.spectrum {
            if let Some(path) = &spectrum_output_path {
                let output_file_path = path.join(format!("{}_cross.spec", base_filename));
                write_complex_spectrum_binary(
                    &output_file_path,
                    &analysis_results.freq_rate_spectrum.to_vec(),
                    effective_fft_point,
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
                    effective_fft_point,
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
                &processing_header,
                &current_obs_time,
                &label_str,
                !rfi_ranges.is_empty(),
                args.frequency,
                args.bandpass.is_some(),
                filename_length,
            );
            let fft_point_half = (effective_fft_point / 2) as usize;
            let available_rows = complex_vec.len() / fft_point_half;
            let requested_rows = physical_length.max(0) as usize;
            let usable_rows = requested_rows.min(available_rows);
            let usable_len = usable_rows * fft_point_half;
            let truncated_vec = complex_vec[..usable_len].to_vec();
            let spectrum_array =
                Array::from_shape_vec((usable_rows, fft_point_half), truncated_vec).unwrap();
            let output_path_freq = dynamic_spectrum_dir
                .join(format!("{}_dynamic_spectrum_frequency.png", base_filename));
            plot_dynamic_spectrum_freq(
                output_path_freq.to_str().unwrap(),
                &spectrum_array,
                &processing_header,
                &current_obs_time,
                current_length,
                effective_integ_time,
            )?;
            let mut lag_data = Array::zeros((usable_rows, effective_fft_point as usize));
            let fft_point_usize = effective_fft_point as usize;
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
                &processing_header,
                &current_obs_time,
                current_length,
                effective_integ_time,
            )?;
        }

        if !args.frequency {
            let delay_output_line = format_delay_output(&analysis_results, &label_str, args.length);
            if l1 == 0 {
                let station1_label = format!("{}-azel", header.station1_name.trim());
                let station2_label = format!("{}-azel", header.station2_name.trim());
                let header_str = format!(
                    concat!(
                        "#*****************************************************************************************************************************************************************************************\n",
                        "#      Epoch        Label    Source     Length    Amp      SNR     Phase     Noise-level      Res-Delay     Res-Rate            {:<20}            {:<16}         MJD\n",
                        "#                                        [s]      [%]               [deg]     1-sigma[%]       [sample]       [Hz]      az[deg]  el[deg]  hgt[m]    az[deg]  el[deg]  hgt[m]\n",
                        "#*****************************************************************************************************************************************************************************************"
                    ),
                    station1_label,
                    station2_label
                );
                if !suppress_output {
                    print!("{}\n", header_str);
                }
                delay_output_str += &format!("{}\n", header_str);
            }
            if !suppress_output {
                print!("{}\n", delay_output_line);
            }
            delay_output_str += &format!("{}\n", delay_output_line);

            if args.cumulate != 0 {
                // Use actual (unpadded) integration time for cumulation plot
                let integ_time = physical_length as f32 * effective_integ_time;
                cumulate_len.push(integ_time);
                cumulate_snr.push(analysis_results.delay_snr);
            }

            add_plot_phase.push(analysis_results.delay_phase);
            add_plot_times.push(current_obs_time);
            let phase_rad = analysis_results.delay_phase.to_radians();
            let complex_sample = Complex::from_polar(analysis_results.delay_max_amp, phase_rad);
            add_plot_complex.push(complex_sample);

            if args.add_plot {
                add_plot_amp.push(analysis_results.delay_max_amp * 100.0);
                add_plot_snr.push(analysis_results.delay_snr);
                add_plot_noise.push(analysis_results.delay_noise * 100.0);
                add_plot_res_delay.push(analysis_results.residual_delay);
                add_plot_res_rate.push(analysis_results.residual_rate);
            }

            if l1 == loop_count - 1 && args.output {
                if let Some(path) = &output_path {
                    let length_label = if args.length == 0 {
                        "0".to_string()
                    } else {
                        args.length.to_string()
                    };
                    let out_dir = path.join(format!("time_domain/len{}s", length_label));
                    fs::create_dir_all(&out_dir)?;
                    let output_basename = first_output_basename.as_ref().unwrap_or(&base_filename);
                    let output_file_path =
                        out_dir.join(format!("{}_delay_rate_search.txt", output_basename));
                    fs::write(output_file_path, &delay_output_str)?;
                }
            }
        } else {
            let freq_output_line = format_freq_output(&analysis_results, &label_str, args.length);
            if l1 == 0 {
                let station1_label = format!("{}-azel", header.station1_name.trim());
                let station2_label = format!("{}-azel", header.station2_name.trim());
                let header_str = format!(
                    concat!(
                        "#*******************************************************************************************************************************************************************************************\n",
                        "#      Epoch        Label    Source     Length    Amp      SNR     Phase     Frequency     Noise-level      Res-Rate            {:<20}             {:<16}        MJD     \n",
                        "#                                        [s]      [%]              [deg]       [MHz]       1-sigma[%]        [Hz]        az[deg]  el[deg]  hgt[m]   az[deg]  el[deg]  hgt[m]                \n",
                        "#*******************************************************************************************************************************************************************************************"
                    ),
                    station1_label,
                    station2_label
                );
                if !suppress_output {
                    print!("{}\n", header_str);
                }
                freq_output_str += &format!("{}\n", header_str);
            }
            if !suppress_output {
                print!("{}\n", freq_output_line);
            }
            freq_output_str += &format!("{}\n", freq_output_line);

            if l1 == loop_count - 1 && args.output {
                if let Some(path) = &output_path {
                    let length_label = if args.length == 0 {
                        "0".to_string()
                    } else {
                        args.length.to_string()
                    };
                    let out_dir = path.join(format!("freq_domain/len{}s", length_label));
                    fs::create_dir_all(&out_dir)?;
                    let output_basename = first_output_basename.as_ref().unwrap_or(&base_filename);
                    let output_file_path =
                        out_dir.join(format!("{}_freq_rate_search.txt", output_basename));
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
        header: processing_header,
        label,
        obs_time: file_start_time,
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
        add_plot_complex,
    })
}

pub(crate) fn run_analysis_pipeline(
    complex_vec: &[C32],
    header: &CorHeader,
    base_args: &Args,
    search_mode: Option<&str>,
    delay_correct: f32,
    rate_correct: f32,
    acel_correct: f32,
    current_length: i32,
    physical_length: i32,
    effective_integ_time: f32,
    current_obs_time: &DateTime<Utc>,
    file_start_time: &DateTime<Utc>,
    rfi_ranges: &[(usize, usize)],
    bandpass_data: &Option<Vec<C32>>,
    effective_fft_point: i32,
) -> Result<(AnalysisResults, Array2<C32>, Array2<C32>), Box<dyn Error>> {
    let mut temp_args = base_args.clone();
    temp_args.delay_correct = delay_correct;
    temp_args.rate_correct = rate_correct;
    temp_args.acel_correct = acel_correct;
    temp_args.search = search_mode
        .map(|mode| vec![mode.to_string()])
        .unwrap_or_default();

    let mut effective_fft_point = effective_fft_point;
    if effective_fft_point <= 0 {
        if current_length <= 0 {
            return Err("セクター長が 0 以下です".into());
        }
        let rows = current_length as usize;
        if rows == 0 || complex_vec.len() % rows != 0 {
            return Err(format!(
                "複素データ長 ({}) がセクター数 ({}) の整数倍ではありません。",
                complex_vec.len(),
                rows
            )
            .into());
        }
        let fft_half = complex_vec.len() / rows;
        effective_fft_point = (fft_half * 2) as i32;
    }

    let fft_point_half = (effective_fft_point / 2) as usize;
    if fft_point_half == 0 {
        return Err("effective FFT point が不正（0）です".into());
    }
    if complex_vec.len() % fft_point_half != 0 {
        return Err(format!(
            "複素データ長 ({}) が FFT チャンネル数 ({}) の整数倍ではありません。",
            complex_vec.len(),
            fft_point_half
        )
        .into());
    }

    if current_length > 0 && complex_vec.len() / fft_point_half != current_length as usize {
        return Err(format!(
            "与えられたセクター数 ({}) とデータから導かれる値 ({}) が一致しません。",
            current_length,
            complex_vec.len() / fft_point_half
        )
        .into());
    }

    let start_time_offset_sec = if search_mode.is_some() {
        0.0
    } else {
        current_obs_time
            .signed_duration_since(*file_start_time)
            .num_seconds() as f32
    };

    let corrected_complex_vec =
        if delay_correct != 0.0 || rate_correct != 0.0 || acel_correct != 0.0 {
            let input_data_2d: Vec<Vec<Complex<f64>>> = complex_vec
                .chunks(fft_point_half)
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|&c| Complex::new(c.re as f64, c.im as f64))
                        .collect()
                })
                .collect();
            let corrected_complex_vec_2d = fft::apply_phase_correction(
                &input_data_2d,
                rate_correct,
                delay_correct,
                acel_correct,
                effective_integ_time,
                header.sampling_speed as u32,
                effective_fft_point as u32,
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
        physical_length,
        effective_fft_point,
        header.sampling_speed,
        rfi_ranges,
        base_args.rate_padding,
    );

    if let Some(bp_data) = &bandpass_data {
        apply_bandpass_correction(&mut freq_rate_array, bp_data);
    }

    let delay_rate_2d_data_comp =
        process_ifft(&freq_rate_array, effective_fft_point, padding_length);

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
