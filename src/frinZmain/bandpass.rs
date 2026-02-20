use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::prelude::*;
use num_complex::Complex;
use std::fs::File;
use std::io::{self, BufReader};

use crate::output::write_complex_spectrum_npy;

type C32 = Complex<f32>;

pub fn read_bandpass_file(path: &std::path::Path) -> io::Result<Vec<C32>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut bandpass_data = Vec::new();

    while let Ok(real) = reader.read_f32::<LittleEndian>() {
        let imag = reader.read_f32::<LittleEndian>().map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Incomplete complex number at end of file",
                )
            } else {
                e
            }
        })?;
        bandpass_data.push(C32::new(real, imag));
    }

    Ok(bandpass_data)
}

pub fn write_complex_spectrum_binary(
    path: &std::path::Path,
    spectrum: &[C32],
    _fft_points: i32,
    _color_flag: i32,
) -> io::Result<()> {
    write_complex_spectrum_npy(path, spectrum)?;
    Ok(())
}

pub fn apply_bandpass_correction(freq_rate_array: &mut Array2<C32>, bandpass_data: &[C32]) {
    if bandpass_data.is_empty() {
        return;
    }
    const EPSILON: f32 = 1e-9;

    // The complex mean is used to rescale the corrected spectrum to maintain a similar overall power and phase.
    let bandpass_sum: C32 = bandpass_data.iter().copied().sum();
    let bandpass_mean = bandpass_sum / bandpass_data.len() as f32;

    for (mut row, &bp_val) in freq_rate_array
        .rows_mut()
        .into_iter()
        .zip(bandpass_data.iter())
    {
        // Avoid division by zero or near-zero values
        if bp_val.norm() > EPSILON {
            row.iter_mut()
                .for_each(|elem| *elem = (*elem / bp_val) * bandpass_mean);
        }
    }
}
