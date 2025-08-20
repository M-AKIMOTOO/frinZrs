use std::io::{self, BufReader, BufWriter, ErrorKind};
use std::fs::File;
use num_complex::Complex;
use byteorder::{ReadBytesExt, LittleEndian, WriteBytesExt};
use ndarray::prelude::*;
use plotters::prelude::*;

type C32 = Complex<f32>;

pub fn read_bandpass_file(path: &std::path::Path) -> io::Result<Vec<C32>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    
    let mut bandpass_data = Vec::new();
    loop {
        let real = match reader.read_f32::<LittleEndian>() {
            Ok(val) => val,
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break, // End of file
            Err(e) => return Err(e),
        };
        let imag = match reader.read_f32::<LittleEndian>() {
            Ok(val) => val,
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // This case (real part read but imaginary part is EOF) should ideally not happen
                // in a correctly formatted file, but we handle it defensively.
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Incomplete complex number at end of file"));
            },
            Err(e) => return Err(e),
        };
        bandpass_data.push(C32::new(real, imag));
    }

    Ok(bandpass_data)
}

pub fn write_complex_spectrum_binary(
    path: &std::path::Path,
    spectrum: &[C32],
    fft_points: i32,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write the complex spectrum data (interleaved real and imaginary parts)
    for val in spectrum {
        writer.write_f32::<LittleEndian>(val.re)?;
        writer.write_f32::<LittleEndian>(val.im)?;
    }

    // Plot the spectrum
    plot_bandpass_spectrum(path, spectrum, fft_points)?;

    Ok(())
}

pub fn apply_bandpass_correction(
    freq_rate_array: &mut Array2<C32>,
    bandpass_data: &[C32],
) {
    let bandpass_mean = bandpass_data.iter().map(|c| c.norm()).sum::<f32>() / bandpass_data.len() as f32;

    for (i, mut row) in freq_rate_array.rows_mut().into_iter().enumerate() {
        if i < bandpass_data.len() {
            let bp_val = bandpass_data[i];
            if bp_val.norm() > 1e-9 { // Avoid division by zero
                for elem in row.iter_mut() {
                    *elem = (*elem / bp_val) * bandpass_mean;
                }
            }
        }
    }
}

pub fn plot_bandpass_spectrum(
    path: &std::path::Path,
    spectrum: &[C32],
    fft_points: i32,
) -> io::Result<()> {
    let output_file_path = path.with_extension("png"); // Change extension to png
    let root = BitMapBackend::new(&output_file_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| io::Error::new(ErrorKind::Other, e.to_string()))?;

    let (upper, lower) = root.split_vertically(250); // Split for two subplots

    // --- Amplitude Plot ---
    let mut amp_chart = ChartBuilder::on(&lower)
        .margin(10)
        .x_label_area_size(55)
        .y_label_area_size(90)
        //.caption("Amplitude Spectrum", ("sans-serif", 20).into_font())
        .build_cartesian_2d(0..fft_points/2, 0.0f32..spectrum.iter().map(|c| c.norm()).fold(0.0f32, |acc, x| acc.max(x)) * 1.1)
        .map_err(|e| io::Error::new(ErrorKind::Other, e.to_string()))?;

    amp_chart.configure_mesh()
        .x_desc("Channels")
        .y_desc("Amplitude")
        .label_style(("sans-serif", 25).into_font())
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .draw().map_err(|e| io::Error::new(ErrorKind::Other, e.to_string()))?;

    amp_chart.draw_series(LineSeries::new(
        spectrum.iter().enumerate().map(|(i, c)| (i as i32, c.norm())),
        &RED,
    )).map_err(|e| io::Error::new(ErrorKind::Other, e.to_string()))?;

    // --- Phase Plot ---
    let mut phase_chart = ChartBuilder::on(&upper)
        .margin(10)
        .x_label_area_size(0)
        .y_label_area_size(90)
        //.caption("Phase Spectrum", ("sans-serif", 20).into_font())
        .build_cartesian_2d(0..fft_points/2, -180.0f32..180.0f32)
        .map_err(|e| io::Error::new(ErrorKind::Other, e.to_string()))?;

    phase_chart.configure_mesh()
        //.x_desc("Channels")
        .y_desc("Phase (deg)")
        .label_style(("sans-serif", 25).into_font())
        .y_label_formatter(&|y| format!("{:.0}", y))
        .draw().map_err(|e| io::Error::new(ErrorKind::Other, e.to_string()))?;

    phase_chart.draw_series(LineSeries::new(
        spectrum.iter().enumerate().map(|(i, c)| (i as i32, c.arg().to_degrees())),
        &RED,
    )).map_err(|e| io::Error::new(ErrorKind::Other, e.to_string()))?;

    root.present().map_err(|e| io::Error::new(ErrorKind::Other, e.to_string()))?;
    Ok(())
}
