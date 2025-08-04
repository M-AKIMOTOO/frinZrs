use ndarray::prelude::*;
use num_complex::Complex;
use rustfft::FftPlanner;

type C32 = Complex<f32>;

pub fn process_fft(
    complex_vec: &[C32],
    length: i32,
    fft_point: i32,
    rfi_ranges: &[(usize, usize)],
) -> (Array2<C32>, Array2<f32>, usize) {
    let length_usize = length as usize;
    let fft_point_half = (fft_point / 2) as usize;
    let padding_length = (length as u32).next_power_of_two() as usize * 4;
    let padding_length_half = padding_length / 2;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(padding_length);

    let complex_array = Array::from_shape_vec((length_usize, fft_point_half), complex_vec.to_vec()).unwrap();
    let mut freq_rate_array = Array2::<C32>::zeros((fft_point_half, padding_length));
    let mut freq_rate_data = Array2::<f32>::zeros((fft_point_half, padding_length));

    for i in 0..fft_point_half {
        let mut fft_in = vec![C32::new(0.0, 0.0); padding_length];
        let is_rfi_channel = rfi_ranges.iter().any(|(min, max)| i >= *min && i <= *max);

        if !is_rfi_channel {
            for (j, val) in complex_array.column(i).iter().enumerate() {
                fft_in[j] = *val;
            }
        }

        let mut fft_out = fft_in.clone();
        fft.process(&mut fft_out);

        let mut shifted_out = vec![C32::new(0.0, 0.0); padding_length];

        // FFT shift
        let (first_half, second_half) = fft_out.split_at(padding_length_half);
        shifted_out[..padding_length_half].copy_from_slice(second_half);
        shifted_out[padding_length_half..].copy_from_slice(first_half);

        let scaled_shifted_out: Vec<C32> = shifted_out
            .iter_mut()
            .map(|val| *val * (fft_point as f32 / length as f32))
            .collect();

        let shifted_norm: Vec<f32> = scaled_shifted_out.iter().map(|val| val.norm()).collect();

        freq_rate_array.row_mut(i).assign(&ArrayView::from(&scaled_shifted_out));
        freq_rate_data.row_mut(i).assign(&ArrayView::from(&shifted_norm));
    }

    (freq_rate_array, freq_rate_data, padding_length)
}

pub fn process_ifft(
    freq_rate_array: &Array2<C32>,
    fft_point: i32,
    padding_length: usize,
) -> (Array2<C32>, Array2<C32>) {
    let fft_point_usize = fft_point as usize;
    let fft_point_half = fft_point_usize / 2;

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(fft_point_usize);

    let mut delay_rate_array = Array2::<C32>::zeros((padding_length, fft_point_usize));

    for i in 0..freq_rate_array.dim().1 {
        let mut ifft_in = vec![C32::new(0.0, 0.0); fft_point_usize];
        let freq_data_col = freq_rate_array.column(i);

        ifft_in[..fft_point_half].copy_from_slice(&freq_data_col.slice(s![..fft_point_half]).to_vec());

        ifft.process(&mut ifft_in);

        // IFFT shift and normalization
        let mut shifted_out = vec![C32::new(0.0, 0.0); fft_point_usize];
        let (first_half, second_half) = ifft_in.split_at(fft_point_half);
        shifted_out[..fft_point_half].copy_from_slice(second_half);
        shifted_out[fft_point_half..].copy_from_slice(first_half);

        for val in &mut shifted_out {
            *val /= fft_point as f32;
        }

        delay_rate_array.row_mut(i).assign(&ArrayView::from(&shifted_out));
    }

    // Reverse the columns of delay_rate_array to match frinZ.py's behavior
    for mut row in delay_rate_array.rows_mut() {
        row.as_slice_mut().unwrap().reverse();
    }
    (delay_rate_array.clone(), delay_rate_array)
}
