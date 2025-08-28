use std::io::{self, BufReader, BufWriter, ErrorKind};
use std::fs::File;
use num_complex::Complex;
use byteorder::{ReadBytesExt, LittleEndian, WriteBytesExt};
use ndarray::prelude::*;
use plotters::prelude::*;

use crate::utils::safe_arg;

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
    fft_points: i32,
    flag: i8, // 0: bandpass, 1: cross-power spectrum
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write the complex spectrum data (interleaved real and imaginary parts)
    for val in spectrum {
        writer.write_f32::<LittleEndian>(val.re)?;
        writer.write_f32::<LittleEndian>(val.im)?;
    }

    if flag == 0 {
        // Plot the spectrum
        plot_bandpass_spectrum(path, spectrum, fft_points)?;
    }
    Ok(())
}

pub fn apply_bandpass_correction(
    freq_rate_array: &mut Array2<C32>,
    bandpass_data: &[C32],
) {
    if bandpass_data.is_empty() {
        return;
    }
    const EPSILON: f32 = 1e-9;

    // The mean is used to rescale the corrected spectrum to maintain a similar overall power level.
    let bandpass_mean =
        bandpass_data.iter().map(|c| c.norm()).sum::<f32>() / bandpass_data.len() as f32;

    for (mut row, &bp_val) in freq_rate_array.rows_mut().into_iter().zip(bandpass_data.iter()) {
        // Avoid division by zero or near-zero values
        if bp_val.norm() > EPSILON {
            row.iter_mut()
                .for_each(|elem| *elem = (*elem / bp_val) * bandpass_mean);
        }
    }
}

pub fn plot_bandpass_spectrum(
    path: &std::path::Path,
    spectrum: &[C32],
    fft_points: i32,
) -> io::Result<()> {
    const PLOT_WIDTH: u32 = 800;
    const PLOT_HEIGHT: u32 = 600;
    const UPPER_PLOT_HEIGHT: u32 = 180;
    const FONT_STYLE: (&str, i32) = ("sans-serif", 25);

    // Helper to convert plotters error to io::Error, reducing boilerplate
    fn to_io_error<E: std::fmt::Display>(e: E) -> io::Error {
        io::Error::new(ErrorKind::Other, e.to_string())
    }

    let output_file_path = path.with_extension("png"); // Change extension to png
    let root = BitMapBackend::new(&output_file_path, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root.fill(&WHITE).map_err(to_io_error)?;

    let (upper, lower) = root.split_vertically(UPPER_PLOT_HEIGHT);

    // --- Phase Plot (Top) ---
    let mut phase_chart = ChartBuilder::on(&upper)
        .margin(10)
        .y_label_area_size(90)
        .build_cartesian_2d(0..fft_points / 2, -180.0f32..180.0f32)
        .map_err(to_io_error)?;

    phase_chart
        .configure_mesh()
        .y_desc("Phase (deg)")
        .label_style(FONT_STYLE)
        .y_label_formatter(&|y| format!("{:.0}", y))
        .draw()
        .map_err(to_io_error)?;

    phase_chart
        .draw_series(LineSeries::new(
            spectrum.iter().enumerate().map(|(i, c)| (i as i32, safe_arg(c).to_degrees())),
            &RED,
        ))
        .map_err(to_io_error)?;

    // --- Amplitude Plot ---
    let max_amp = spectrum.iter().map(|c| c.norm()).fold(0.0f32, f32::max);
    // Add a small epsilon to the max amplitude to avoid a zero-range in case of all-zero spectrum
    let y_range_amp = 0.0f32..(max_amp * 1.1).max(1e-9);

    let mut amp_chart = ChartBuilder::on(&lower)
        .margin(10)
        .x_label_area_size(55)
        .y_label_area_size(90)
        .build_cartesian_2d(0..fft_points / 2, y_range_amp)
        .map_err(to_io_error)?;

    amp_chart
        .configure_mesh()
        .x_desc("Channels")
        .y_desc("Amplitude")
        .label_style(FONT_STYLE)
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .draw()
        .map_err(to_io_error)?;

    amp_chart
        .draw_series(LineSeries::new(
            spectrum.iter().enumerate().map(|(i, c)| (i as i32, c.norm())),
            &RED,
        ))
        .map_err(to_io_error)?;

    root.present().map_err(to_io_error)?;
    Ok(())
}
