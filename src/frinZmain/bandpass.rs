use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::prelude::*;
use num_complex::Complex;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;

use crate::output::{write_complex_spectrum_npy, ComplexRiRow};

type C32 = Complex<f32>;

pub fn read_bandpass_file(path: &Path) -> io::Result<Vec<C32>> {
    let is_npy = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("npy"))
        .unwrap_or(false);

    if is_npy {
        return read_bandpass_npy(path);
    }

    read_bandpass_binary(path)
}

fn read_bandpass_binary(path: &Path) -> io::Result<Vec<C32>> {
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

fn read_bandpass_npy(path: &Path) -> io::Result<Vec<C32>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let npy = npyz::NpyFile::new(reader).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid NPY bandpass file '{}': {}", path.display(), e),
        )
    })?;
    let rows = npy.into_vec::<ComplexRiRow>().map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Failed to decode NPY bandpass '{}' as dtype [('real','<f4'),('imag','<f4')]: {}",
                path.display(),
                e
            ),
        )
    })?;

    Ok(rows
        .into_iter()
        .map(|row| C32::new(row.real, row.imag))
        .collect())
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

    // Phase+amplitude are corrected by division with complex bandpass.
    // Rescaling uses real mean amplitude to avoid injecting a global phase rotation.
    let bandpass_mean_amp =
        bandpass_data.iter().map(|bp| bp.norm()).sum::<f32>() / bandpass_data.len() as f32;

    for (mut row, &bp_val) in freq_rate_array
        .rows_mut()
        .into_iter()
        .zip(bandpass_data.iter())
    {
        // Avoid division by zero or near-zero values
        if bp_val.norm() > EPSILON {
            row.iter_mut()
                .for_each(|elem| *elem = (*elem / bp_val) * bandpass_mean_amp);
        }
    }
}
