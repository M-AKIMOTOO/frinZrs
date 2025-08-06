use ndarray::prelude::*;
use num_complex::Complex;
use chrono::{DateTime, Utc};

use crate::header::CorHeader;
use crate::args::Args;
use crate::utils::{rate_cal, noise_level, radec2azalt, mjd_cal};
use crate::fitting;

type C32 = Complex<f32>;

#[derive(Debug)]
pub struct AnalysisResults {
    // Common
    pub yyyydddhhmmss1: String,
    pub source_name: String,
    pub length_f32: f32,
    pub ant1_az: f32,
    pub ant1_el: f32,
    pub ant1_hgt: f32,
    pub ant2_az: f32,
    pub ant2_el: f32,
    pub ant2_hgt: f32,
    pub mjd: f64,
    // Delay
    pub delay_range: Array1<f32>,
    pub visibility: Array1<f32>,
    pub delay_rate: Array1<f32>,
    pub delay_max_amp: f32,
    pub delay_phase: f32,
    pub delay_snr: f32,
    pub delay_noise: f32,
    pub residual_delay: f32,
    pub corrected_delay: f32,
    pub delay_offset: f32,
    // Frequency
    pub freq_max_amp: f32,
    pub freq_phase: f32,
    pub freq_freq: f32,
    pub freq_snr: f32,
    pub freq_noise: f32,
    pub freq_rate: Array1<f32>,
    pub freq_rate_spectrum: Array1<C32>,
    pub freq_range: Array1<f32>,
    pub freq_max_freq: f32,
    pub residual_rate: f32,
    pub corrected_rate: f32,
    pub rate_offset: f32,
    // Ranges
    pub rate_range: Vec<f32>,
}

pub fn analyze_results(
    freq_rate_array: &Array2<C32>,
    delay_rate_array: &Array2<C32>,
    header: &CorHeader,
    length: i32,
    effective_integ_time: f32,
    obs_time: &DateTime<Utc>,
    padding_length: usize,
    args: &Args,
) -> AnalysisResults {
    let fft_point_usize = header.fft_point as usize;
    let fft_point_half = fft_point_usize / 2;
    let fft_point_f32 = header.fft_point as f32;
    let length_f32 = length as f32;
    let padding_length_half = padding_length / 2;

    // --- Ranges ---
    let delay_range = Array::linspace(-(fft_point_f32 / 2.0) + 1.0, fft_point_f32 / 2.0, fft_point_usize);

    let freq_range = Array::linspace(0.0f32, (header.sampling_speed as f32 / 2.0) / 1e6, fft_point_half);
    let rate_range = rate_cal(padding_length as f32, effective_integ_time);

    // --- Delay Analysis ---
    let delay_rate_2d_data_array = delay_rate_array.clone().mapv(|x| x.norm());
    let delay_noise = noise_level(delay_rate_array.view(), delay_rate_array.mean().unwrap(), padding_length, fft_point_usize);

    

    let (peak_rate_idx, peak_delay_idx) = if !args.delay_window.is_empty() && !args.rate_window.is_empty() {
        // Case 3: Window options are specified, search within them.
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
                        let current_val = delay_rate_2d_data_array[[r_idx, d_idx]];
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
    } else if args.search {
        // Case 2: --search is specified, no window. Find the global maximum.
        let (mut max_val, mut max_r_idx, mut max_d_idx) = (0.0f32, 0, 0);
        for r_idx in 0..delay_rate_2d_data_array.shape()[0] {
            for d_idx in 0..delay_rate_2d_data_array.shape()[1] {
                let current_val = delay_rate_2d_data_array[[r_idx, d_idx]];
                if current_val > max_val {
                    max_val = current_val;
                    max_r_idx = r_idx;
                    max_d_idx = d_idx;
                }
            }
        }
        (max_r_idx, max_d_idx)
    } else {
        // Case 1: No window and no --search. Use the center point (delay=0, rate=0).
        (padding_length_half, fft_point_half - 1)
    };
    
    let delay_max_amp = delay_rate_2d_data_array[[peak_rate_idx, peak_delay_idx]];
    let delay_phase = delay_rate_array[[peak_rate_idx, peak_delay_idx]].arg().to_degrees();
    let delay_rate_slice = delay_rate_2d_data_array.column(peak_delay_idx).to_owned();

    let mut residual_delay_val: f32 = delay_range[peak_delay_idx];
    let mut delay_offset = 0.0;
    if args.search {
        // Extract points for quadratic fitting
        let mut x_coords: Vec<f64> = Vec::new();
        let mut y_values: Vec<f64> = Vec::new();

        // Define the window size (e.g., 3 points: peak and 1 point on each side)
        let window_size = 3;
        let half_window = (window_size / 2) as isize;

        for i in -half_window..=half_window {
            let current_idx = peak_delay_idx as isize + i;
            if current_idx >= 0 && current_idx < delay_range.len() as isize {
                x_coords.push(delay_range[current_idx as usize] as f64);
                y_values.push(delay_rate_2d_data_array[[peak_rate_idx, current_idx as usize]] as f64);
            }
        }

        if let Ok(fit_result) = fitting::fit_quadratic_least_squares(&x_coords, &y_values) {
            // The peak_x from fitting is the absolute delay value
            // We need to calculate the offset from the original peak_delay_idx
            delay_offset = fit_result.peak_x as f32; // This is the residual offset
            residual_delay_val = args.delay_correct + delay_offset; // This is the total corrected delay
        } else {
            // Handle error, for now, just print and use original peak
            eprintln!("Warning: Quadratic fitting for delay failed. Using original peak.");
            // delay_offset remains 0.0, residual_delay_val remains original
        }
    }

    let delay_snr = delay_max_amp / delay_noise;
    let visibility = delay_rate_2d_data_array.row(peak_rate_idx).to_owned();

    let mut residual_rate_val: f32;
    let mut rate_offset = 0.0;

    // --- Frequency Analysis ---
    let freq_rate_2d_data_array = freq_rate_array.clone().mapv(|x| x.norm());

    let (peak_freq_row_idx, peak_rate_col_idx) = if !args.rate_window.is_empty() {
        // Case 3: Window option is specified.
        let rate_win_low = args.rate_window[0];
        let rate_win_high = args.rate_window[1];

        let mut max_val_in_window = 0.0f32;
        let mut temp_peak_freq_row_idx = 0;
        let mut temp_peak_rate_col_idx = padding_length_half;

        for r_idx in 0..rate_range.len() {
            if rate_range[r_idx] >= rate_win_low && rate_range[r_idx] <= rate_win_high {
                for f_idx in 0..freq_range.len() {
                    let current_val = freq_rate_2d_data_array[[f_idx, r_idx]];
                    if current_val > max_val_in_window {
                        max_val_in_window = current_val;
                        temp_peak_freq_row_idx = f_idx;
                        temp_peak_rate_col_idx = r_idx;
                    }
                }
            }
        }
        (temp_peak_freq_row_idx, temp_peak_rate_col_idx)
    } else if args.search {
        // Case 2: --search is specified, no window. Find the global maximum.
        let (mut max_val, mut max_f_idx, mut max_r_idx) = (0.0f32, 0, 0);
        for f_idx in 0..freq_rate_2d_data_array.shape()[0] {
            for r_idx in 0..freq_rate_2d_data_array.shape()[1] {
                let current_val = freq_rate_2d_data_array[[f_idx, r_idx]];
                if current_val > max_val {
                    max_val = current_val;
                    max_f_idx = f_idx;
                    max_r_idx = r_idx;
                }
            }
        }
        (max_f_idx, max_r_idx)
    } else {
        // Case 1: No window and no --search. Use the center point (rate=0) and find max frequency.
        let cross_power_slice = freq_rate_2d_data_array.column(padding_length_half);
        let (max_f_idx, _) = cross_power_slice.iter().enumerate().fold((0, 0.0f32), |(i_max, v_max), (i, &v)| {
            if v > v_max { (i, v) } else { (i_max, v_max) }
        });
        (max_f_idx, padding_length_half)
    };

    let freq_max_amp = freq_rate_2d_data_array[[peak_freq_row_idx, peak_rate_col_idx]];
    let freq_phase = freq_rate_array[[peak_freq_row_idx, peak_rate_col_idx]].arg().to_degrees();
    let freq_freq = freq_range[peak_freq_row_idx];
    let freq_noise = noise_level(freq_rate_array.view(), freq_rate_array.mean().unwrap(), fft_point_half, padding_length);
    let freq_snr = freq_max_amp / freq_noise;
    let freq_rate = freq_rate_2d_data_array.row(peak_freq_row_idx).to_owned();
    let freq_rate_spectrum = freq_rate_array.column(peak_rate_col_idx).to_owned();

    // Set the final residual_rate_val based on the mode
    if args.frequency {
        residual_rate_val = rate_range[peak_rate_col_idx];
    } else {
        residual_rate_val = rate_range[peak_rate_idx];
    };

    if args.search {
        if args.frequency {
            let mut x_coords: Vec<f64> = Vec::new();
            let mut y_values: Vec<f64> = Vec::new();

            let window_size = 3;
            let half_window = (window_size / 2) as isize;

            for i in -half_window..=half_window {
                let current_idx = peak_rate_col_idx as isize + i;
                if current_idx >= 0 && current_idx < rate_range.len() as isize {
                    x_coords.push(rate_range[current_idx as usize] as f64);
                    y_values.push(freq_rate_2d_data_array[[peak_freq_row_idx, current_idx as usize]] as f64);
                }
            }

            let rate_scale_factor = (10.0 * padding_length as f64) * effective_integ_time as f64;
            let scaled_x_coords: Vec<f64> = x_coords.iter().map(|&x| x * rate_scale_factor).collect();

            if let Ok(fit_result) = fitting::fit_quadratic_least_squares(&scaled_x_coords, &y_values) {
                rate_offset = (fit_result.peak_x / rate_scale_factor) as f32; // This is the residual offset
                residual_rate_val = args.rate_correct + rate_offset; // This is the total corrected rate
            }
        } else {
            let mut x_coords: Vec<f64> = Vec::new();
            let mut y_values: Vec<f64> = Vec::new();

            let window_size = 3;
            let half_window = (window_size / 2) as isize;

            for i in -half_window..=half_window {
                let current_idx = peak_rate_idx as isize + i;
                if current_idx >= 0 && current_idx < rate_range.len() as isize {
                    x_coords.push(rate_range[current_idx as usize] as f64);
                    y_values.push(delay_rate_array[[current_idx as usize, peak_delay_idx]].norm() as f64);
                }
            }

            let rate_scale_factor = (10.0 * padding_length as f64) * effective_integ_time as f64;
            let scaled_x_coords: Vec<f64> = x_coords.iter().map(|&x| x * rate_scale_factor).collect();

            if let Ok(fit_result) = fitting::fit_quadratic_least_squares(&scaled_x_coords, &y_values) {
                rate_offset = (fit_result.peak_x / rate_scale_factor) as f32; // This is the residual offset
                residual_rate_val = args.rate_correct + rate_offset; // This is the total corrected rate
            }
        }
    }                                                        


    // --- Antenna Az/El Calculation ---
    let (ant1_az, ant1_el, ant1_hgt) = radec2azalt(
        [header.station1_position[0] as f32, header.station1_position[1] as f32, header.station1_position[2] as f32],
        *obs_time, header.source_position_ra as f32, header.source_position_dec as f32
    );
    let (ant2_az, ant2_el, ant2_hgt) = radec2azalt(
        [header.station2_position[0] as f32, header.station2_position[1] as f32, header.station2_position[2] as f32],
        *obs_time, header.source_position_ra as f32, header.source_position_dec as f32
    );
    
    AnalysisResults {
        yyyydddhhmmss1: obs_time.format("%Y/%j %H:%M:%S").to_string(),
        source_name: header.source_name.clone(),
        length_f32,
        ant1_az: ant1_az as f32,
        ant1_el: ant1_el as f32,
        ant1_hgt: ant1_hgt as f32,
        ant2_az: ant2_az as f32,
        ant2_el: ant2_el as f32,
        ant2_hgt: ant2_hgt as f32,
        mjd: mjd_cal(*obs_time),
        delay_range,
        visibility,
        delay_rate: delay_rate_slice,
        delay_max_amp,
        delay_phase,
        delay_snr,
        delay_noise,
        residual_delay: residual_delay_val,
        corrected_delay: args.delay_correct,
        delay_offset,
        freq_max_amp,
        freq_phase,
        freq_freq,
        freq_snr,
        freq_noise,
        freq_rate,
        freq_rate_spectrum,
        freq_range,
        freq_max_freq: freq_freq,
        residual_rate: residual_rate_val,
        corrected_rate: args.rate_correct,
        rate_offset,
        rate_range,
    }
}