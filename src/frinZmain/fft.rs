use ndarray::prelude::*;
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

type C32 = Complex<f32>;

pub fn process_fft(
    complex_vec: &[C32],
    length: i32,
    fft_point: i32,
    sampling_speed: i32,
    rfi_ranges: &[(usize, usize)],
    rate_padding: u32,
) -> (Array2<C32>, usize) {
    let bandwidth = sampling_speed as f32 / 2.0 / 1_000_000.0; // [MHz]

    let length_usize = length as usize;
    let fft_point_half = (fft_point / 2) as usize;
    //let padding_length = (length as u32).next_power_of_two() as usize * rate_padding as usize; // FFTのビン（周波数の区切り）と実際の信号周波数のズレによって生じる「スカロッピング損失」という現象
    let mut padding_length = length as usize * rate_padding as usize; 
    if length == 1 {
        padding_length *= 2.0 as usize
    }
    let padding_length_half = padding_length / 2;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(padding_length);

    let complex_array = Array::from_shape_vec((length_usize, fft_point_half), complex_vec.to_vec()).unwrap();
    let mut freq_rate_array = Array2::<C32>::zeros((fft_point_half, padding_length));

    for i in 1..fft_point_half { // DC成分（FFTシフト前のインデックス0）を0+0jに設定
        let mut fft_exe = vec![C32::new(0.0, 0.0); padding_length];
        let is_rfi_channel = rfi_ranges.iter().any(|(min, max)| i >= *min && i <= *max);

        if !is_rfi_channel {
            for (j, val) in complex_array.column(i).iter().enumerate() {
                fft_exe[j] = *val;
            }
        }

        fft.process(&mut fft_exe);

        let mut shifted_out = vec![C32::new(0.0, 0.0); padding_length];

        // FFT shift
        let (first_half, second_half) = fft_exe.split_at(padding_length_half);
        shifted_out[..padding_length_half].copy_from_slice(second_half);
        shifted_out[padding_length_half..].copy_from_slice(first_half);

        let scaled_shifted_out: Vec<C32> = shifted_out
            .iter_mut()
            .map(|val| *val * (fft_point as f32 / length as f32) * (512.0 / bandwidth))
            .collect();

        freq_rate_array.row_mut(i).assign(&ArrayView::from(&scaled_shifted_out));
    }

    (freq_rate_array, padding_length)
}

pub fn process_ifft(
    freq_rate_array: &Array2<C32>,
    fft_point: i32,
    padding_length: usize,
) -> Array2<C32> {
    let fft_point_usize = fft_point as usize;
    let mut delay_rate_array = Array2::<C32>::zeros((padding_length, fft_point_usize));

    for i in 0..freq_rate_array.dim().1 {
        let freq_data_col = freq_rate_array.column(i);
        let ifft_result = perform_ifft_on_vec(&freq_data_col.to_vec(), fft_point_usize);
        delay_rate_array.row_mut(i).assign(&ArrayView::from(&ifft_result));
    }

    delay_rate_array
}


pub fn perform_ifft_on_vec(input: &[C32], ifft_size: usize) -> Vec<C32> {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(ifft_size);

    let mut ifft_exe = vec![C32::new(0.0, 0.0); ifft_size];
    ifft_exe[..input.len()].copy_from_slice(input);

    ifft.process(&mut ifft_exe);

    let mut shifted_out = vec![C32::new(0.0, 0.0); ifft_size];
    let (first_half, second_half) = ifft_exe.split_at(ifft_size / 2);
    shifted_out[..ifft_size / 2].copy_from_slice(second_half);
    shifted_out[ifft_size / 2..].copy_from_slice(first_half);

    for val in &mut shifted_out {
        *val /= ifft_size as f32;
    }
    
    shifted_out.reverse(); // Common reverse operation

    shifted_out
}

/// Applies phase correction to input data
pub fn apply_phase_correction(
    input_data: &[Vec<Complex<f64>>],
    rate_hz_for_correction: f32,
    delay_samples_for_correction: f32,
    acel_hz_for_correction: f32,
    effective_integration_length: f32,
    sampling_speed: u32,
    fft_point: u32,
    start_time_offset_sec: f32,
) -> Vec<Vec<Complex<f64>>> {
    let mut corrected_data = input_data.to_vec();

    let n_rows_original = input_data.len();
    let n_cols_original = if n_rows_original > 0 { input_data[0].len() } else { 0 };

    let can_phase_correct = sampling_speed > 0 && fft_point >= 2 && (effective_integration_length as f64).abs() > 1e-9 && n_cols_original > 0;

    if can_phase_correct {
        let freq_resolution_hz = sampling_speed as f64 / fft_point as f64;
        let delay_seconds = delay_samples_for_correction as f64 / sampling_speed as f64;

        for r_orig in 0..n_rows_original {
            let time_for_rate_corr_sec = (r_orig as f64 * effective_integration_length as f64) + start_time_offset_sec as f64;
            let rate_corr_factor = Complex::new(0.0, -2.0 * PI * rate_hz_for_correction as f64 * time_for_rate_corr_sec).exp() * Complex::new(0.0, -1.0 * PI * acel_hz_for_correction as f64 * time_for_rate_corr_sec.powi(2)).exp();

            for c_orig in 0..n_cols_original {
                let freq_k_hz_for_delay_corr = c_orig as f64 * freq_resolution_hz;
                let delay_corr_factor = Complex::new(0.0, -2.0 * PI * delay_seconds * freq_k_hz_for_delay_corr).exp();
                
                corrected_data[r_orig][c_orig] *= rate_corr_factor * delay_corr_factor;
            }
        }
    }
    corrected_data
}
