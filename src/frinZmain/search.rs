pub use acel::run_acel_search_analysis;
pub use deep::{run_coherent_search, run_deep_search, DeepSearchParams, DeepSearchResult};

mod acel {
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::{Cursor, Write};
    use std::path::Path;

    use chrono::{DateTime, Utc};
    use num_complex::Complex;

    use crate::args::Args;
    use crate::fitting;
    use crate::header::{parse_header, CorHeader};
    use crate::input_support::{output_stem_from_path, read_input_bytes};
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
        let output_dir = parent_dir.join("frinZ").join("search");
        fs::create_dir_all(&output_dir)?;
        let base_filename = output_stem_from_path(input_path)?;

        let buffer = read_input_bytes(input_path)?;
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
                    match fitting::fit_polynomial_least_squares(&times_for_fit, &delays_seconds, 2)
                    {
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
                    let linear_path = output_dir.join(format!(
                        "{}_step{}_linear.txt",
                        base_filename,
                        step_idx + 1
                    ));
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
                    "{}_step{}_resrate.png",
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
                    "{}_step{}_resdelay.png",
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
}

mod deep {
    use chrono::{DateTime, Utc};
    use ndarray::prelude::*;
    use num_complex::Complex;
    use rayon::prelude::*;
    use std::error::Error;
    use std::f64::consts::PI;

    use crate::analysis::{analyze_results, AnalysisResults};
    use crate::args::Args;
    use crate::bandpass::apply_bandpass_correction;
    use crate::fft::{apply_phase_correction, process_fft, process_ifft};
    use crate::header::CorHeader;

    type C32 = Complex<f32>;

    /// Deep searchで使用する探索パラメータ
    #[derive(Debug, Clone)]
    pub struct DeepSearchParams {
        pub delay_search_range: f32,       // ±0.5 sample
        pub rate_search_range_factor: f32, // ±1/(2*pp) Hz

        pub max_iterations: usize, // 階層の深さ
    }

    impl Default for DeepSearchParams {
        fn default() -> Self {
            Self {
                delay_search_range: 0.5,
                rate_search_range_factor: 0.5,

                max_iterations: 4,
            }
        }
    }

    /// Deep search探索結果
    #[derive(Debug, Clone)]
    pub struct DeepSearchResult {
        pub analysis_results: AnalysisResults,
        pub freq_rate_array: Array2<C32>,
        pub delay_rate_2d_data: Array2<C32>,
    }

    /// Deep searchメイン関数
    pub fn run_deep_search(
        complex_vec: &[C32],
        header: &CorHeader,
        current_length: i32,
        physical_length: i32,
        effective_integ_time: f32,
        current_obs_time: &DateTime<Utc>,
        _obs_time: &DateTime<Utc>,
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
        args: &Args,
        pp: i32,
        cpu_count_arg: u32, // New argument
        previous_solution: Option<(f32, f32)>,
    ) -> Result<DeepSearchResult, Box<dyn Error>> {
        println!("[DEEP SEARCH] Starting deep hierarchical search algorithm");

        // フリンジ補正はファイル開始時刻からの経過時間で行う
        let start_time_offset_sec = 0.0;

        if current_length <= 0 {
            return Err("有効なセクター長が 0 以下です".into());
        }
        let rows = current_length as usize;
        if rows == 0 || complex_vec.is_empty() {
            return Err("有効なデータが存在しません".into());
        }
        if complex_vec.len() % rows != 0 {
            return Err(format!(
                "複素データ長 ({}) がセクター数 ({}) の整数倍ではありません",
                complex_vec.len(),
                rows
            )
            .into());
        }
        let fft_point_half = complex_vec.len() / rows;
        if fft_point_half == 0 {
            return Err("FFT チャンネル数が 0 です".into());
        }
        let effective_fft_point = (fft_point_half * 2) as i32;

        // Step 1: 粗い遅延・レート推定
        let (coarse_delay, coarse_rate) = if let Some((prev_delay, prev_rate)) = previous_solution {
            println!(
                "[DEEP SEARCH] Seeding from previous solution: delay={:.6}, rate={:.6}",
                prev_delay, prev_rate
            );
            (prev_delay, prev_rate)
        } else {
            println!("[DEEP SEARCH] Running coarse grid search for initial estimate");
            get_coarse_estimates(
                complex_vec,
                header,
                current_length,
                physical_length,
                effective_integ_time,
                current_obs_time,
                rfi_ranges,
                bandpass_data,
                args,
                effective_fft_point,
                true,
            )?
        };

        println!(
            "[DEEP SEARCH] Coarse estimates - Delay: {:.6} samples, Rate: {:.6} Hz",
            coarse_delay, coarse_rate
        );

        // Step 2: 階層的探索
        let mut search_params = DeepSearchParams::default();
        search_params.max_iterations = (args.iter.max(1)) as usize;
        let mut current_delay = coarse_delay;
        let mut current_rate = coarse_rate;
        let effective_cpu_count =
            determine_effective_cpu_count(cpu_count_arg, rows, fft_point_half, args.rate_padding);
        println!(
            "[DEEP SEARCH] Using {} worker threads (requested: {}).",
            effective_cpu_count,
            if cpu_count_arg == 0 {
                "auto".to_string()
            } else {
                cpu_count_arg.to_string()
            }
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(effective_cpu_count)
            .build()?;

        for iteration in 0..search_params.max_iterations {
            println!("[DEEP SEARCH] Iteration {} starting", iteration + 1);

            // 現在の階層での探索範囲とステップサイズを計算
            let scale_factor = 10.0_f32.powi(iteration as i32);
            let delay_range = search_params.delay_search_range / scale_factor;
            let rate_range =
                search_params.rate_search_range_factor / (2.0 * pp as f32) / scale_factor;
            let grid_points_per_axis = 11usize;
            let delay_step = if grid_points_per_axis > 1 {
                (2.0 * delay_range) / (grid_points_per_axis as f32 - 1.0)
            } else {
                0.0
            };
            let rate_step = if grid_points_per_axis > 1 {
                (2.0 * rate_range) / (grid_points_per_axis as f32 - 1.0)
            } else {
                0.0
            };

            println!(
                "[DEEP SEARCH]   Delay range: +/- {:.6} samples, step: {:.6}",
                delay_range, delay_step
            );
            println!(
                "[DEEP SEARCH]   Rate range: +/- {:.6} Hz, step: {:.6}",
                rate_range, rate_step
            );

            // 並列グリッド探索
            let (best_delay, best_rate, _best_score) = parallel_grid_search(
                complex_vec,
                header,
                current_length,
                physical_length,
                effective_integ_time,
                current_obs_time,
                rfi_ranges,
                bandpass_data,
                args,
                current_delay,
                current_rate,
                delay_range,
                rate_range,
                grid_points_per_axis,
                &pool,
                start_time_offset_sec,
                effective_fft_point,
                true,
            )?;

            let (iter_analysis_results, _, _) = perform_final_analysis(
                complex_vec,
                header,
                current_length,
                physical_length,
                effective_integ_time,
                current_obs_time,
                rfi_ranges,
                bandpass_data,
                args,
                best_delay,
                best_rate,
                start_time_offset_sec,
                effective_fft_point,
            )?;
            let iter_snr = iter_analysis_results.delay_snr;

            // 結果を更新
            current_delay = best_delay;
            current_rate = best_rate;

            println!(
                "[DEEP SEARCH]   Best result: delay={:.6} samples, rate={:.6} Hz, SNR={:.3}",
                current_delay, current_rate, iter_snr
            );
        }

        let final_delay = current_delay;
        let final_rate = current_rate;

        // Step 3: 最終的な解析を実行
        println!(
            "[DEEP SEARCH] Final result - Delay: {:.6} samples, Rate: {:.6} Hz",
            final_delay, final_rate
        );

        let (final_analysis_results, final_freq_rate_array, final_delay_rate_2d_data) =
            perform_final_analysis(
                complex_vec,
                header,
                current_length,
                physical_length,
                effective_integ_time,
                current_obs_time,
                rfi_ranges,
                bandpass_data,
                args,
                final_delay,
                final_rate,
                start_time_offset_sec,
                effective_fft_point,
            )?;

        Ok(DeepSearchResult {
            analysis_results: final_analysis_results,
            freq_rate_array: final_freq_rate_array,
            delay_rate_2d_data: final_delay_rate_2d_data,
        })
    }

    /// Coherent search:
    /// delay/rate 候補の評価は FFT/IFFT を使わず、コヒーレント和の SNR のみで行う。
    /// 最後に 1 回だけ既存パイプライン（FFT/IFFT）で最終解析を実行する。
    pub fn run_coherent_search(
        complex_vec: &[C32],
        header: &CorHeader,
        current_length: i32,
        physical_length: i32,
        effective_integ_time: f32,
        current_obs_time: &DateTime<Utc>,
        _obs_time: &DateTime<Utc>,
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
        args: &Args,
        _pp: i32,
        cpu_count_arg: u32,
        previous_solution: Option<(f32, f32)>,
    ) -> Result<DeepSearchResult, Box<dyn Error>> {
        // println!("[COHERENT SEARCH] Starting coherent delay-rate search");

        let start_time_offset_sec = 0.0;

        if current_length <= 0 {
            return Err("有効なセクター長が 0 以下です".into());
        }
        let rows = current_length as usize;
        if rows == 0 || complex_vec.is_empty() {
            return Err("有効なデータが存在しません".into());
        }
        if complex_vec.len() % rows != 0 {
            return Err(format!(
                "複素データ長 ({}) がセクター数 ({}) の整数倍ではありません",
                complex_vec.len(),
                rows
            )
            .into());
        }

        let fft_point_half = complex_vec.len() / rows;
        if fft_point_half == 0 {
            return Err("FFT チャンネル数が 0 です".into());
        }
        let effective_fft_point = (fft_point_half * 2) as i32;

        let has_window = args.drange.len() == 2 || args.rrange.len() == 2;
        let (mut current_delay, mut current_rate) = if let Some((prev_delay, prev_rate)) = previous_solution {
            // println!(
            //     "[COHERENT SEARCH] Seeding from previous solution: delay={:.6}, rate={:.6}",
            //     prev_delay, prev_rate
            // );
            (prev_delay, prev_rate)
        } else if has_window {
            let (d0, r0) = infer_coherent_initial_center(args, fft_point_half);
            // println!(
            //     "[COHERENT SEARCH] Initial center from windows: delay={:.6}, rate={:.6}",
            //     d0, r0
            // );
            (d0, r0)
        } else {
            // println!(
            //     "[COHERENT SEARCH] No windows specified. Running one-shot coarse FFT seed search."
            // );
            let (d0, r0) = get_coarse_estimates(
                complex_vec,
                header,
                current_length,
                physical_length,
                effective_integ_time,
                current_obs_time,
                rfi_ranges,
                bandpass_data,
                args,
                effective_fft_point,
                false,
            )?;
            // println!(
            //     "[COHERENT SEARCH] Initial center: delay={:.6}, rate={:.6}",
            //     d0, r0
            // );
            (d0, r0)
        };

        let (initial_delay_range, initial_rate_range) = infer_coherent_initial_ranges(args, _pp);
        // println!(
        //     "[COHERENT SEARCH] Initial range: delay=+/-{:.6}, rate=+/-{:.6}",
        //     initial_delay_range, initial_rate_range
        // );

        let max_iterations = args.iter.max(1) as usize;
        let effective_cpu_count =
            determine_effective_cpu_count(cpu_count_arg, rows, fft_point_half, args.rate_padding);
        // println!(
        //     "[COHERENT SEARCH] Using {} worker threads (requested: {}).",
        //     effective_cpu_count,
        //     if cpu_count_arg == 0 {
        //         "auto".to_string()
        //     } else {
        //         cpu_count_arg.to_string()
        //     }
        // );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(effective_cpu_count)
            .build()?;

        for iteration in 0..max_iterations {
            let scale_factor = 10.0_f32.powi(iteration as i32);
            let delay_range = initial_delay_range / scale_factor;
            let rate_range = initial_rate_range / scale_factor;
            let grid_points_per_axis = 11usize;
            let _delay_step = if grid_points_per_axis > 1 {
                (2.0 * delay_range) / (grid_points_per_axis as f32 - 1.0)
            } else {
                0.0
            };
            let _rate_step = if grid_points_per_axis > 1 {
                (2.0 * rate_range) / (grid_points_per_axis as f32 - 1.0)
            } else {
                0.0
            };

            // println!("[COHERENT SEARCH] Iteration {} starting", iteration + 1);
            // println!(
            //     "[COHERENT SEARCH]   Delay range: +/- {:.6} samples, step: {:.6}",
            //     delay_range, delay_step
            // );
            // println!(
            //     "[COHERENT SEARCH]   Rate range: +/- {:.6} Hz, step: {:.6}",
            //     rate_range, rate_step
            // );

            let (best_delay, best_rate, _best_score) = parallel_grid_search(
                complex_vec,
                header,
                current_length,
                physical_length,
                effective_integ_time,
                current_obs_time,
                rfi_ranges,
                bandpass_data,
                args,
                current_delay,
                current_rate,
                delay_range,
                rate_range,
                grid_points_per_axis,
                &pool,
                start_time_offset_sec,
                effective_fft_point,
                false,
            )?;

            current_delay = best_delay;
            current_rate = best_rate;
            // println!(
            //     "[COHERENT SEARCH]   Best result: delay={:.6}, rate={:.6}, coh={:.3}",
            //     current_delay, current_rate, best_score
            // );
        }

        // println!(
        //     "[COHERENT SEARCH] Final result - Delay: {:.6} samples, Rate: {:.6} Hz",
        //     current_delay, current_rate
        // );

        let (final_analysis_results, final_freq_rate_array, final_delay_rate_2d_data) =
            perform_final_analysis(
                complex_vec,
                header,
                current_length,
                physical_length,
                effective_integ_time,
                current_obs_time,
                rfi_ranges,
                bandpass_data,
                args,
                current_delay,
                current_rate,
                start_time_offset_sec,
                effective_fft_point,
            )?;

        Ok(DeepSearchResult {
            analysis_results: final_analysis_results,
            freq_rate_array: final_freq_rate_array,
            delay_rate_2d_data: final_delay_rate_2d_data,
        })
    }

    fn infer_coherent_initial_center(args: &Args, fft_point_half: usize) -> (f32, f32) {
        let mut delay_center = if args.drange.len() == 2 {
            0.5 * (args.drange[0] + args.drange[1])
        } else {
            args.delay_correct
        };

        let rate_center = if args.rrange.len() == 2 {
            0.5 * (args.rrange[0] + args.rrange[1])
        } else {
            args.rate_correct
        };

        let delay_limit = fft_point_half as f32;
        delay_center = delay_center.clamp(-delay_limit, delay_limit);

        (delay_center, rate_center)
    }

    fn infer_coherent_initial_ranges(args: &Args, pp: i32) -> (f32, f32) {
        let delay_range = if args.drange.len() == 2 {
            0.5 * (args.drange[1] - args.drange[0]).abs()
        } else {
            0.5
        };

        let rate_range = if args.rrange.len() == 2 {
            0.5 * (args.rrange[1] - args.rrange[0]).abs()
        } else {
            0.5 / (2.0 * (pp.max(1) as f32))
        };

        (delay_range.max(0.0), rate_range.max(0.0))
    }

    fn determine_effective_cpu_count(
        cpu_count_arg: u32,
        rows: usize,
        fft_point_half: usize,
        rate_padding: u32,
    ) -> usize {
        let num_available_cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        let requested = if cpu_count_arg == 0 {
            num_available_cpus
        } else {
            (cpu_count_arg as usize).clamp(1, num_available_cpus)
        };

        let per_worker_bytes =
            estimate_deep_worker_memory_bytes(rows, fft_point_half, rate_padding).max(1);
        let mem_limited = sys_info::mem_info()
            .ok()
            .map(|m| {
                // Leave some headroom for OS + other allocations.
                let available_bytes = (m.avail as u128) * 1024;
                let budget_bytes = available_bytes.saturating_mul(70).saturating_div(100);
                let max_workers_by_mem = (budget_bytes / per_worker_bytes as u128) as usize;
                max_workers_by_mem.max(1)
            })
            .unwrap_or(1);

        requested.min(mem_limited).max(1)
    }

    fn estimate_deep_worker_memory_bytes(
        rows: usize,
        fft_point_half: usize,
        rate_padding: u32,
    ) -> usize {
        if rows == 0 || fft_point_half == 0 {
            return 1;
        }

        let n = rows as u128 * fft_point_half as u128;
        let mut padding_length = rows as u128 * rate_padding.max(1) as u128;
        if rows == 1 {
            padding_length = padding_length.saturating_mul(2);
        }

        // Main temporary allocations in one deep-eval worker.
        // - phase correction path: f64 2D in/out + f32 flatten
        // - FFT path: complex_array, freq-rate array
        // - IFFT path: delay-rate array
        let phase_corr = n.saturating_mul(16 + 16 + 8); // input f64 + corrected f64 + flattened f32
        let fft_arrays = n.saturating_mul(8); // Array copy in process_fft
        let freq_rate = (fft_point_half as u128)
            .saturating_mul(padding_length)
            .saturating_mul(8);
        let delay_rate = (2_u128.saturating_mul(fft_point_half as u128))
            .saturating_mul(padding_length)
            .saturating_mul(8);

        // Safety margin for Vec metadata / temporary work buffers.
        let overhead = 64_u128.saturating_mul(1024).saturating_mul(1024);
        let total = phase_corr
            .saturating_add(fft_arrays)
            .saturating_add(freq_rate)
            .saturating_add(delay_rate)
            .saturating_add(overhead);

        total.min(usize::MAX as u128) as usize
    }

    /// 粗い遅延・レート推定 (delay-windowとrate-windowから、または通常の探索から)
    fn get_coarse_estimates(
        complex_vec: &[C32],
        header: &CorHeader,
        current_length: i32,
        physical_length: i32,
        effective_integ_time: f32,
        current_obs_time: &DateTime<Utc>,
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
        args: &Args,
        effective_fft_point: i32,
        emit_logs: bool,
    ) -> Result<(f32, f32), Box<dyn Error>> {
        let coarse_rate_padding = 1_u32;
        if emit_logs && args.rate_padding > coarse_rate_padding {
            println!(
                "[DEEP SEARCH] Coarse stage uses rate-padding={} (final stage keeps {}).",
                coarse_rate_padding, args.rate_padding
            );
        }

        // drange/rrange が指定されている場合は、その範囲で探索
        if !args.drange.is_empty() || !args.rrange.is_empty() {
            if emit_logs {
                println!("[DEEP SEARCH] Using specified delay/rate windows for coarse estimation");
            }

            let (mut freq_rate_array, padding_length) = process_fft(
                complex_vec,
                physical_length,
                effective_fft_point,
                header.sampling_speed,
                rfi_ranges,
                coarse_rate_padding,
            );

            if let Some(bp_data) = bandpass_data {
                apply_bandpass_correction(&mut freq_rate_array, bp_data);
            }

            let delay_rate_2d_data_comp =
                process_ifft(&freq_rate_array, effective_fft_point, padding_length);

            let analysis_results = analyze_results(
                &freq_rate_array,
                &delay_rate_2d_data_comp,
                header,
                current_length,
                effective_integ_time,
                current_obs_time,
                padding_length,
                args,
                args.primary_search_mode(),
            );

            let coarse_delay = analysis_results.residual_delay;
            let coarse_rate = analysis_results.residual_rate;

            Ok((coarse_delay, coarse_rate))
        } else {
            // delay-windowとrate-windowが指定されていない場合は、二次関数フィッティングなしの粗い探索を実行
            if emit_logs {
                println!(
                    "[DEEP SEARCH] No windows specified, running coarse search (no fitting) for initial estimates"
                );
            }

            let mut search_args = args.clone();
            search_args.search = vec!["deep".to_string()]; // Enable global max search, disable fitting

            let (mut freq_rate_array, padding_length) = process_fft(
                complex_vec,
                physical_length,
                effective_fft_point,
                header.sampling_speed,
                rfi_ranges,
                coarse_rate_padding,
            );

            if let Some(bp_data) = bandpass_data {
                apply_bandpass_correction(&mut freq_rate_array, bp_data);
            }

            let delay_rate_2d_data_comp =
                process_ifft(&freq_rate_array, effective_fft_point, padding_length);

            let analysis_results = analyze_results(
                &freq_rate_array,
                &delay_rate_2d_data_comp,
                header,
                current_length,
                effective_integ_time,
                current_obs_time,
                padding_length,
                &search_args,
                search_args.primary_search_mode(),
            );

            // 粗い探索の結果（フィッティングなし）を取得
            // delay_offsetとrate_offsetは0になるので、delay_rangeとrate_rangeから直接ピーク位置を取得
            let coarse_delay = analysis_results.residual_delay;
            let coarse_rate = analysis_results.residual_rate;

            Ok((coarse_delay, coarse_rate))
        }
    }

    /// 並列グリッド探索
    fn parallel_grid_search(
        complex_vec: &[C32],
        _header: &CorHeader,
        _current_length: i32,
        _physical_length: i32,
        effective_integ_time: f32,
        _current_obs_time: &DateTime<Utc>,
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
        args: &Args,
        center_delay: f32,
        center_rate: f32,
        delay_range: f32,
        rate_range: f32,
        grid_points_per_axis: usize,
        pool: &rayon::ThreadPool,
        start_time_offset_sec: f32,
        effective_fft_point: i32,
        emit_grid_log: bool,
    ) -> Result<(f32, f32, f32), Box<dyn Error>> {
        let fft_point_half = (effective_fft_point / 2) as usize;
        if fft_point_half == 0 {
            return Err("FFT チャンネル数が 0 です".into());
        }

        let mut valid_channels = vec![true; fft_point_half];
        valid_channels[0] = false; // Keep behavior consistent with process_fft (starts from ch=1).
        for &(min_idx, max_idx) in rfi_ranges {
            if min_idx >= fft_point_half {
                continue;
            }
            let clamped_max = max_idx.min(fft_point_half - 1);
            for ch in min_idx..=clamped_max {
                valid_channels[ch] = false;
            }
        }

        let valid_channel_indices: Vec<usize> = valid_channels
            .iter()
            .enumerate()
            .filter_map(|(ch, &is_valid)| if is_valid { Some(ch) } else { None })
            .collect();
        if valid_channel_indices.is_empty() {
            return Ok((center_delay, center_rate, 0.0));
        }

        let channel_correction = build_channel_correction(fft_point_half, bandpass_data);

        // 探索グリッドを生成
        let delay_points =
            generate_search_points(center_delay, delay_range, grid_points_per_axis);
        let rate_points = generate_search_points(center_rate, rate_range, grid_points_per_axis);

        if emit_grid_log {
            println!(
                "[SEARCH]   Grid: {} delay x {} rate = {} combinations",
                delay_points.len(),
                rate_points.len(),
                delay_points.len() * rate_points.len()
            );
        }

        let delay_bounds = if args.drange.len() == 2 {
            Some((
                args.drange[0].min(args.drange[1]),
                args.drange[0].max(args.drange[1]),
            ))
        } else {
            None
        };
        let rate_bounds = if args.rrange.len() == 2 {
            Some((
                args.rrange[0].min(args.rrange[1]),
                args.rrange[0].max(args.rrange[1]),
            ))
        } else {
            None
        };

        // 並列探索実行
        let final_result = pool.install(|| {
            delay_points
                .par_iter()
                .filter_map(|&delay| {
                    if let Some((low, high)) = delay_bounds {
                        if delay < low || delay > high {
                            return None;
                        }
                    }

                    let mut best_for_delay: Option<(f32, f32, f32)> = None;
                    for &rate in &rate_points {
                        if let Some((low, high)) = rate_bounds {
                            if rate < low || rate > high {
                                continue;
                            }
                        }

                        let Ok(score) = evaluate_delay_rate_snr(
                            complex_vec,
                            effective_integ_time,
                            args.acel_correct,
                            &valid_channel_indices,
                            &channel_correction,
                            delay,
                            rate,
                            start_time_offset_sec,
                            effective_fft_point,
                        ) else {
                            continue;
                        };

                        match best_for_delay {
                            Some((_, _, best_score)) if score <= best_score => {}
                            _ => best_for_delay = Some((delay, rate, score)),
                        }
                    }

                    best_for_delay
                })
                .reduce_with(|best, candidate| {
                    if candidate.2 > best.2 {
                        candidate
                    } else {
                        best
                    }
                })
                .unwrap_or((center_delay, center_rate, 0.0f32))
        });

        Ok(final_result)
    }

    /// 特定の遅延・レート組み合わせでのSNRを評価
    fn evaluate_delay_rate_snr(
        complex_vec: &[C32],
        effective_integ_time: f32,
        acel_correct: f32,
        valid_channel_indices: &[usize],
        channel_correction: &[Complex<f64>],
        delay: f32,
        rate: f32,
        start_time_offset_sec: f32,
        effective_fft_point: i32,
    ) -> Result<f32, Box<dyn Error>> {
        let fft_point_half = (effective_fft_point / 2) as usize;
        if fft_point_half == 0 || complex_vec.len() % fft_point_half != 0 {
            return Err("deep候補評価: 複素データ形状が不正です".into());
        }
        if channel_correction.len() != fft_point_half {
            return Err("deep候補評価: チャンネル補正長が不正です".into());
        }

        let row_count = complex_vec.len() / fft_point_half;
        if row_count == 0 || valid_channel_indices.is_empty() {
            return Ok(0.0);
        }

        let fft_point_f64 = effective_fft_point as f64;
        let delay_phase_scale = -2.0 * PI * (delay as f64) / fft_point_f64;
        let start_offset = start_time_offset_sec as f64;
        let integ_time = effective_integ_time as f64;
        let rate_hz = rate as f64;
        let acel_hz = acel_correct as f64;

        let delay_factors: Vec<Complex<f64>> = valid_channel_indices
            .iter()
            .map(|&ch| {
                let phase_delay = delay_phase_scale * ch as f64;
                Complex::<f64>::new(0.0, phase_delay).exp()
            })
            .collect();

        let mut coherent_sum = Complex::<f64>::new(0.0, 0.0);
        let mut total_power = 0.0f64;
        let valid_count = row_count * valid_channel_indices.len();
        if valid_count == 0 {
            return Ok(0.0);
        }

        // phase(row) = a*t + b*t^2 を使って位相因子を再帰更新し、exp() 呼び出しを最小化する。
        let a = -2.0 * PI * rate_hz;
        let b = -PI * acel_hz;
        let dt = integ_time;
        let t0 = start_offset;
        let phase0 = a * t0 + b * t0 * t0;
        let delta0 = a * dt + b * (2.0 * t0 * dt + dt * dt);
        let delta_delta = 2.0 * b * dt * dt;

        let mut rate_factor = Complex::<f64>::new(0.0, phase0).exp();
        let mut rate_step = Complex::<f64>::new(0.0, delta0).exp();
        let rate_step_inc = Complex::<f64>::new(0.0, delta_delta).exp();

        for row in 0..row_count {
            let row_base = row * fft_point_half;
            for (idx, &ch) in valid_channel_indices.iter().enumerate() {
                let raw = complex_vec[row_base + ch];
                let value = Complex::<f64>::new(raw.re as f64, raw.im as f64) * channel_correction[ch];
                let corrected = value * rate_factor * delay_factors[idx];
                coherent_sum += corrected;
                total_power += corrected.norm_sqr();
            }

            rate_factor *= rate_step;
            rate_step *= rate_step_inc;
            if (row & 1023) == 1023 {
                normalize_unit_complex(&mut rate_factor);
                normalize_unit_complex(&mut rate_step);
            }
        }

        let n = valid_count as f64;
        let mean = coherent_sum / n;
        let var = (total_power / n - mean.norm_sqr()).max(0.0);
        let sigma = var.sqrt();
        let denom = (n.sqrt() * sigma).max(1e-12);
        let snr = (coherent_sum.norm() / denom) as f32;

        Ok(snr)
    }

    fn build_channel_correction(
        fft_point_half: usize,
        bandpass_data: &Option<Vec<C32>>,
    ) -> Vec<Complex<f64>> {
        let mut correction = vec![Complex::<f64>::new(1.0, 0.0); fft_point_half];

        let Some(bp_data) = bandpass_data.as_ref() else {
            return correction;
        };
        if bp_data.is_empty() {
            return correction;
        }

        let sum: C32 = bp_data.iter().copied().sum();
        let mean = sum / bp_data.len() as f32;
        let mean64 = Complex::<f64>::new(mean.re as f64, mean.im as f64);

        for ch in 0..fft_point_half.min(bp_data.len()) {
            let bp = bp_data[ch];
            let bp_norm = bp.norm() as f64;
            if bp_norm > 1e-12 {
                let bp64 = Complex::<f64>::new(bp.re as f64, bp.im as f64);
                correction[ch] = mean64 / bp64;
            }
        }
        correction
    }

    #[inline]
    fn normalize_unit_complex(value: &mut Complex<f64>) {
        let norm = value.norm();
        if norm > 0.0 {
            *value /= norm;
        }
    }

    /// 最終的な解析を実行
    fn perform_final_analysis(
        complex_vec: &[C32],
        header: &CorHeader,
        current_length: i32,
        physical_length: i32,
        effective_integ_time: f32,
        current_obs_time: &DateTime<Utc>,
        rfi_ranges: &[(usize, usize)],
        bandpass_data: &Option<Vec<C32>>,
        args: &Args,
        final_delay: f32,
        final_rate: f32,
        start_time_offset_sec: f32,
        effective_fft_point: i32,
    ) -> Result<(AnalysisResults, Array2<C32>, Array2<C32>), Box<dyn Error>> {
        // 最適解で最終的な処理を実行
        let corrected_complex_vec = apply_corrections(
            complex_vec,
            final_rate,
            final_delay,
            args.acel_correct,
            effective_integ_time,
            header,
            start_time_offset_sec,
            effective_fft_point,
        )?;

        let (mut final_freq_rate_array, padding_length) = process_fft(
            &corrected_complex_vec,
            physical_length,
            effective_fft_point,
            header.sampling_speed,
            rfi_ranges,
            args.rate_padding,
        );

        if let Some(bp_data) = bandpass_data {
            apply_bandpass_correction(&mut final_freq_rate_array, bp_data);
        }

        let final_delay_rate_2d_data_comp =
            process_ifft(&final_freq_rate_array, effective_fft_point, padding_length);

        let final_args = create_corrected_args(args, final_delay, final_rate);
        let mut analysis_results = analyze_results(
            &final_freq_rate_array,
            &final_delay_rate_2d_data_comp,
            header,
            current_length,
            effective_integ_time,
            current_obs_time,
            padding_length,
            &final_args,
            Some("deep"), // Final analysis for deep search should find the global max
        );

        // Deep searchの結果を反映
        analysis_results.residual_delay = final_delay;
        analysis_results.residual_rate = final_rate;
        analysis_results.length_f32 = physical_length as f32 * effective_integ_time;

        Ok((
            analysis_results,
            final_freq_rate_array,
            final_delay_rate_2d_data_comp,
        ))
    }

    /// 探索点を生成
    fn generate_search_points(center: f32, range: f32, count: usize) -> Vec<f32> {
        if !range.is_finite() || range <= 0.0 || count <= 1 {
            return vec![center];
        }
        let n = count.max(2);
        let start = center - range;
        let step = (2.0 * range) / (n as f32 - 1.0);
        (0..n).map(|i| start + step * i as f32).collect()
    }

    /// 位相補正を適用
    fn apply_corrections(
        complex_vec: &[C32],
        rate: f32,
        delay: f32,
        acel: f32,
        effective_integ_time: f32,
        header: &CorHeader,
        start_time_offset_sec: f32,
        effective_fft_point: i32,
    ) -> Result<Vec<C32>, Box<dyn Error>> {
        if rate == 0.0 && delay == 0.0 {
            return Ok(complex_vec.to_vec());
        }

        let fft_point_half = (effective_fft_point / 2) as usize;
        if fft_point_half == 0 {
            return Err("FFT チャンネル数が 0 です".into());
        }

        let input_data_2d: Vec<Vec<Complex<f64>>> = complex_vec
            .chunks(fft_point_half)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|&c| Complex::new(c.re as f64, c.im as f64))
                    .collect()
            })
            .collect();

        let corrected_complex_vec_2d = apply_phase_correction(
            &input_data_2d,
            rate,
            delay,
            acel,
            effective_integ_time,
            header.sampling_speed as u32,
            effective_fft_point as u32,
            start_time_offset_sec,
        );

        let corrected_vec = corrected_complex_vec_2d
            .into_iter()
            .flatten()
            .map(|v| Complex::new(v.re as f32, v.im as f32))
            .collect();

        Ok(corrected_vec)
    }

    /// 補正された引数を作成
    fn create_corrected_args(args: &Args, delay: f32, rate: f32) -> Args {
        let mut corrected_args = args.clone();
        corrected_args.delay_correct = delay;
        corrected_args.rate_correct = rate;
        // Keep window semantics in absolute coordinates by converting
        // to residual windows for already-corrected data.
        if corrected_args.drange.len() == 2 {
            corrected_args.drange[0] -= delay;
            corrected_args.drange[1] -= delay;
        }
        if corrected_args.rrange.len() == 2 {
            corrected_args.rrange[0] -= rate;
            corrected_args.rrange[1] -= rate;
        }
        corrected_args.search.clear(); // Prevent infinite loops
        corrected_args
    }
}
