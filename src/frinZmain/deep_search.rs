use chrono::{DateTime, Utc};
use ndarray::prelude::*;
use num_complex::Complex;
use rayon::prelude::*;
use std::error::Error;
use std::sync::{Arc, Mutex};

use crate::analysis::{analyze_results, AnalysisResults};
use crate::args::Args;
use crate::bandpass::apply_bandpass_correction;
use crate::fft::{self, apply_phase_correction, process_fft, process_ifft};
use crate::header::CorHeader;

type C32 = Complex<f32>;

const CONTINUITY_SNR_MARGIN: f32 = 0.02;

/// Deep searchで使用する探索パラメータ
#[derive(Debug, Clone)]
pub struct DeepSearchParams {
    pub delay_fine_step: f32,          // 0.1 sample
    pub rate_fine_step_factor: f32,    // 1/(10*pp)
    pub delay_search_range: f32,       // ±0.5 sample
    pub rate_search_range_factor: f32, // ±1/(2*pp) Hz

    pub max_iterations: usize, // 階層の深さ
}

impl Default for DeepSearchParams {
    fn default() -> Self {
        Self {
            delay_fine_step: 0.1,
            rate_fine_step_factor: 0.1,
            delay_search_range: 0.5,
            rate_search_range_factor: 0.5,

            max_iterations: 6,
        }
    }
}

fn median(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) * 0.5)
    } else {
        Some(values[mid])
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

    // フリンジ補正は各セグメントの相対時間で行う
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
            effective_integ_time,
            current_obs_time,
            rfi_ranges,
            bandpass_data,
            args,
            effective_fft_point,
        )?
    };

    println!(
        "[DEEP SEARCH] Coarse estimates - Delay: {:.6} samples, Rate: {:.6} Hz",
        coarse_delay, coarse_rate
    );

    // Step 2: 階層的探索
    let search_params = DeepSearchParams::default();
    let mut current_delay = coarse_delay;
    let mut current_rate = coarse_rate;
    let mut delay_history = vec![coarse_delay];
    let mut rate_history = vec![coarse_rate];

    for iteration in 0..search_params.max_iterations {
        println!("[DEEP SEARCH] Iteration {} starting", iteration + 1);

        // 現在の階層での探索範囲とステップサイズを計算
        let scale_factor = 10.0_f32.powi(iteration as i32);
        let delay_range = search_params.delay_search_range / scale_factor;
        let rate_range = search_params.rate_search_range_factor / (2.0 * pp as f32) / scale_factor;
        let delay_step = search_params.delay_fine_step / scale_factor;
        let rate_step = search_params.rate_fine_step_factor / (10.0 * pp as f32) / scale_factor;

        println!(
            "[DEEP SEARCH]   Delay range: +/- {:.6} samples, step: {:.6}",
            delay_range, delay_step
        );
        println!(
            "[DEEP SEARCH]   Rate range: +/- {:.6} Hz, step: {:.6}",
            rate_range, rate_step
        );

        // Determine effective CPU count
        let num_available_cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let mut effective_cpu_count = if cpu_count_arg == 0 {
            num_available_cpus
        } else if (cpu_count_arg as usize) > num_available_cpus {
            // If user specified more than available, use half of available, but at least 1
            (num_available_cpus / 2).max(1)
        } else {
            // User specified a valid number within available CPUs
            cpu_count_arg as usize
        };

        // Apply the maximum 10 cores constraint
        effective_cpu_count = effective_cpu_count.min(10);

        // 並列グリッド探索
        let (best_delay, best_rate, best_snr) = parallel_grid_search(
            complex_vec,
            header,
            current_length,
            effective_integ_time,
            current_obs_time,
            rfi_ranges,
            bandpass_data,
            args,
            current_delay,
            current_rate,
            delay_range,
            rate_range,
            delay_step,
            rate_step,
            effective_cpu_count,
            start_time_offset_sec,
            effective_fft_point,
        )?;

        // 結果を更新
        current_delay = best_delay;
        current_rate = best_rate;
        delay_history.push(best_delay);
        rate_history.push(best_rate);

        println!(
            "[DEEP SEARCH]   Best result: delay={:.6} samples, rate={:.6} Hz, SNR={:.3}",
            current_delay, current_rate, best_snr
        );
    }

    let mut delay_history_clone = delay_history.clone();
    let mut rate_history_clone = rate_history.clone();
    let mut final_delay = median(&mut delay_history_clone).unwrap_or(current_delay);
    let mut final_rate = median(&mut rate_history_clone).unwrap_or(current_rate);

    if let Some((prev_delay, prev_rate)) = previous_solution {
        let delay_jump = (final_delay - prev_delay).abs();
        let rate_jump = (final_rate - prev_rate).abs();
        let delay_jump_threshold = search_params.delay_fine_step * 1.5;
        let rate_jump_threshold =
            (search_params.rate_fine_step_factor / (10.0 * pp as f32)).abs() * 5.0;

        if delay_jump > delay_jump_threshold || rate_jump > rate_jump_threshold {
            let candidate_snr = evaluate_delay_rate_snr(
                complex_vec,
                header,
                current_length,
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

            let previous_snr = evaluate_delay_rate_snr(
                complex_vec,
                header,
                current_length,
                effective_integ_time,
                current_obs_time,
                rfi_ranges,
                bandpass_data,
                args,
                prev_delay,
                prev_rate,
                start_time_offset_sec,
                effective_fft_point,
            )?;

            let snr_gain = if previous_snr > 0.0 {
                (candidate_snr - previous_snr) / previous_snr
            } else {
                f32::INFINITY
            };

            if snr_gain < CONTINUITY_SNR_MARGIN {
                println!(
                    "[DEEP SEARCH]   Continuity guard keeping previous solution (Δdelay={:.6}, Δrate={:.6}, ΔSNR={:.2}%)",
                    delay_jump,
                    rate_jump,
                    snr_gain * 100.0
                );
                final_delay = prev_delay;
                final_rate = prev_rate;
            } else {
                println!(
                    "[DEEP SEARCH]   Continuity guard accepted new solution (Δdelay={:.6}, Δrate={:.6}, ΔSNR={:.2}%)",
                    delay_jump,
                    rate_jump,
                    snr_gain * 100.0
                );
            }
        }
    }

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

/// 粗い遅延・レート推定 (delay-windowとrate-windowから、または通常の探索から)
fn get_coarse_estimates(
    complex_vec: &[C32],
    header: &CorHeader,
    current_length: i32,
    effective_integ_time: f32,
    current_obs_time: &DateTime<Utc>,
    rfi_ranges: &[(usize, usize)],
    bandpass_data: &Option<Vec<C32>>,
    args: &Args,
    effective_fft_point: i32,
) -> Result<(f32, f32), Box<dyn Error>> {
    // delay-windowとrate-windowが指定されている場合は、その範囲で探索
    if !args.delay_window.is_empty() && !args.rate_window.is_empty() {
        println!("[DEEP SEARCH] Using specified delay and rate windows for coarse estimation");

        let (mut freq_rate_array, padding_length) = process_fft(
            complex_vec,
            current_length,
            effective_fft_point,
            header.sampling_speed,
            rfi_ranges,
            args.rate_padding,
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

        let coarse_delay = analysis_results.delay_offset;
        let coarse_rate = analysis_results.rate_offset;

        Ok((coarse_delay, coarse_rate))
    } else {
        // delay-windowとrate-windowが指定されていない場合は、二次関数フィッティングなしの粗い探索を実行
        println!("[DEEP SEARCH] No windows specified, running coarse search (no fitting) for initial estimates");

        let mut search_args = args.clone();
        search_args.search = vec!["deep".to_string()]; // Enable global max search, disable fitting

        let (mut freq_rate_array, padding_length) = process_fft(
            complex_vec,
            current_length,
            effective_fft_point,
            header.sampling_speed,
            rfi_ranges,
            args.rate_padding,
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
    header: &CorHeader,
    current_length: i32,
    effective_integ_time: f32,
    current_obs_time: &DateTime<Utc>,
    rfi_ranges: &[(usize, usize)],
    bandpass_data: &Option<Vec<C32>>,
    args: &Args,
    center_delay: f32,
    center_rate: f32,
    delay_range: f32,
    rate_range: f32,
    delay_step: f32,
    rate_step: f32,
    effective_cpu_count: usize,
    start_time_offset_sec: f32,
    effective_fft_point: i32,
) -> Result<(f32, f32, f32), Box<dyn Error>> {
    // 探索グリッドを生成
    let delay_points = generate_search_points(center_delay, delay_range, delay_step);
    let rate_points = generate_search_points(center_rate, rate_range, rate_step);

    println!(
        "[DEEP SEARCH]   Grid: {} delay x {} rate = {} combinations",
        delay_points.len(),
        rate_points.len(),
        delay_points.len() * rate_points.len()
    );

    // 全ての組み合わせを生成
    let mut search_combinations = Vec::new();
    for &delay in &delay_points {
        for &rate in &rate_points {
            search_combinations.push((delay, rate));
        }
    }

    // 並列処理の設定
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(effective_cpu_count)
        .build()?;

    let best_result = Arc::new(Mutex::new((center_delay, center_rate, 0.0f32)));

    // 並列探索実行
    pool.install(|| {
        search_combinations.par_iter().for_each(|&(delay, rate)| {
            if let Ok(snr) = evaluate_delay_rate_snr(
                complex_vec,
                header,
                current_length,
                effective_integ_time,
                current_obs_time,
                rfi_ranges,
                bandpass_data,
                args,
                delay,
                rate,
                start_time_offset_sec,
                effective_fft_point,
            ) {
                let mut best = best_result.lock().unwrap();
                if snr > best.2 {
                    *best = (delay, rate, snr);
                }
            }
        });
    });

    let final_result = best_result.lock().unwrap().clone();
    Ok(final_result)
}

/// 特定の遅延・レート組み合わせでのSNRを評価
fn evaluate_delay_rate_snr(
    complex_vec: &[C32],
    header: &CorHeader,
    current_length: i32,
    effective_integ_time: f32,
    current_obs_time: &DateTime<Utc>,
    rfi_ranges: &[(usize, usize)],
    bandpass_data: &Option<Vec<C32>>,
    args: &Args,
    delay: f32,
    rate: f32,
    start_time_offset_sec: f32,
    effective_fft_point: i32,
) -> Result<f32, Box<dyn Error>> {
    // 位相補正を適用
    let corrected_complex_vec = apply_corrections(
        complex_vec,
        rate,
        delay,
        args.acel_correct,
        effective_integ_time,
        header,
        start_time_offset_sec,
        effective_fft_point,
    )?;

    // FFT処理
    let (mut freq_rate_array, padding_length) = process_fft(
        &corrected_complex_vec,
        current_length,
        effective_fft_point,
        header.sampling_speed,
        rfi_ranges,
        args.rate_padding,
    );

    // バンドパス補正
    if let Some(bp_data) = bandpass_data {
        apply_bandpass_correction(&mut freq_rate_array, bp_data);
    }

    // IFFT処理
    let delay_rate_2d_data_comp =
        process_ifft(&freq_rate_array, effective_fft_point, padding_length);

    // 解析実行
    let temp_args = create_corrected_args(args, delay, rate);
    let analysis_results = analyze_results(
        &freq_rate_array,
        &delay_rate_2d_data_comp,
        header,
        current_length,
        effective_integ_time,
        current_obs_time,
        padding_length,
        &temp_args,
        None, // No search fitting during deep search iterations
    );

    Ok(analysis_results.delay_snr)
}

/// 最終的な解析を実行
fn perform_final_analysis(
    complex_vec: &[C32],
    header: &CorHeader,
    current_length: i32,
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
        current_length,
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
    analysis_results.length_f32 = current_length as f32 * effective_integ_time;

    Ok((
        analysis_results,
        final_freq_rate_array,
        final_delay_rate_2d_data_comp,
    ))
}

/// 探索点を生成
fn generate_search_points(center: f32, range: f32, step: f32) -> Vec<f32> {
    let mut points = Vec::new();

    // Use f64 for precision-critical calculations to avoid floating point errors
    // with very small steps, which can cause inconsistent point counts or infinite loops.
    let center64 = center as f64;
    let range64 = range as f64;
    let step64 = step as f64;

    // Guard against infinite loop if step is zero or too small to be represented.
    if step64 == 0.0 {
        if range64 >= 0.0 {
            points.push(center);
        }
        return points;
    }

    let start = center64 - range64;
    let end = center64 + range64;

    let mut current = start;
    // Add a small tolerance to the end condition to handle floating point inaccuracies
    while current <= end + step64 * 0.5 {
        points.push(current as f32);
        current += step64;
    }

    // 最大10点に制限（計算量制御）
    if points.len() > 10 {
        // Ensure step_by is at least 1 to avoid panic
        let step_by = (points.len() / 10).max(1);
        points = points.into_iter().step_by(step_by).collect();
    }

    points
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
    corrected_args.search.clear(); // Prevent infinite loops
    corrected_args
}
