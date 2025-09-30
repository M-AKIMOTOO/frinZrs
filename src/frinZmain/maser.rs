use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::Path;
use num_complex::Complex;
use ndarray::{Array1, Axis};
use plotters::prelude::*;

use crate::args::Args;
use crate::header::{parse_header, CorHeader};
use crate::read::read_visibility_data;
use crate::fft::process_fft;
use crate::rfi::parse_rfi_ranges;

/// Extracts the cross-power spectrum at zero fringe rate from a .cor file.
fn get_spectrum_from_file(
    file_path: &Path,
    args: &Args,
) -> Result<(CorHeader, Array1<f32>), Box<dyn Error>> {
    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let mut cursor = Cursor::new(buffer.as_slice());

    let header = parse_header(&mut cursor)?;

    let (complex_vec, _, _) = read_visibility_data(
        &mut cursor,
        &header,
        header.number_of_sector,
        0,
        0,
        false,
        &[],
    )?;

    let rbw = (header.sampling_speed as f32 / header.fft_point as f32) / 1e6;
    let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw)?;

    let (freq_rate_array, padding_length) = process_fft(
        &complex_vec,
        header.number_of_sector,
        header.fft_point,
        header.sampling_speed,
        &rfi_ranges,
        args.rate_padding,
    );

    // Get spectrum at zero rate (center of rate dimension)
    let zero_rate_idx = padding_length / 2;
    let spectrum_complex = freq_rate_array.index_axis(Axis(1), zero_rate_idx);
    
    let spectrum_abs = spectrum_complex.mapv(|x| x.norm());

    Ok((header, spectrum_abs))
}

pub fn run_maser_analysis(args: &Args) -> Result<(), Box<dyn Error>> {
    println!("Running Maser Analysis...");
    let on_source_path = args.input.as_ref().unwrap();
    let off_source_path = args.maser.as_ref().unwrap();

    println!("  ON Source: {:?}", on_source_path);
    println!("  OFF Source: {:?}", off_source_path);

    // Get spectra
    let (header_on, spec_on) = get_spectrum_from_file(on_source_path, args)?;
    let (header_off, spec_off) = get_spectrum_from_file(off_source_path, args)?;

    // Create output directory
    let parent_dir = on_source_path.parent().unwrap_or_else(|| Path::new(""));
    let output_dir = parent_dir.join("frinZ").join("maser");
    fs::create_dir_all(&output_dir)?;
    let on_stem = on_source_path.file_stem().unwrap().to_str().unwrap();
    let off_stem = off_source_path.file_stem().unwrap().to_str().unwrap();

    let freq_resolution_mhz = (header_on.sampling_speed as f32 / 2.0 / 1e6) / (header_on.fft_point as f32 / 2.0);
    let freq_range: Vec<f32> = (0..header_on.fft_point as usize / 2)
        .map(|i| i as f32 * freq_resolution_mhz + (header_on.observing_frequency as f32 / 1e6))
        .collect();

    // --- Write data to CSV ---
    let csv_filename = output_dir.join(format!("{}_vs_{}_data.csv", on_stem, off_stem));
    let mut file = File::create(&csv_filename)?;
    writeln!(file, "Frequency,onsource,offsource")?;

    for i in 0..freq_range.len() {
        writeln!(file, "{},{},{}", freq_range[i], spec_on[i], spec_off[i])?;
    }
    println!("Maser data saved to: {:?}", csv_filename);
    // --- End of CSV writing ---

    // --- Plot for ON vs OFF ---
    let on_plot_data: Vec<(f32, f32)> = freq_range.iter().zip(spec_on.iter()).map(|(&x, &y)| (x, y)).collect();
    let off_plot_data: Vec<(f32, f32)> = freq_range.iter().zip(spec_off.iter()).map(|(&x, &y)| (x, y)).collect();
    
    let on_off_plot_filename = output_dir.join(format!("{}_vs_{}_on_off.png", on_stem, off_stem));

    plot_on_off_spectra(
        &on_off_plot_filename,
        &on_plot_data,
        &off_plot_data,
    )?;
    println!("ON vs OFF source spectra plot saved to: {:?}", on_off_plot_filename);
    // --- End of plot section ---

    // Validate headers
    if header_on.fft_point != header_off.fft_point {
        return Err("ON and OFF source files must have the same fft_point.".into());
    }
    if header_on.sampling_speed != header_off.sampling_speed {
        return Err("ON and OFF source files must have the same sampling_speed.".into());
    }

    // Calculate (ON - OFF) / OFF
    if spec_on.len() != spec_off.len() {
        return Err("ON and OFF spectra have different lengths.".into());
    }

    let mut normalized_spec = Array1::<f32>::zeros(spec_on.len());
    for i in 0..spec_on.len() {
        if spec_off[i] > 1e-9 { // Avoid division by zero
            normalized_spec[i] = (spec_on[i] - spec_off[i]) / spec_off[i];
        } else {
            normalized_spec[i] = 0.0;
        }
    }

    // Find peak
    let mut peak_val = f32::NEG_INFINITY;
    let mut peak_idx = 0;
    for (i, &val) in normalized_spec.iter().enumerate() {
        if val > peak_val {
            peak_val = val;
            peak_idx = i;
        }
    }

    let peak_freq_mhz = peak_idx as f32 * freq_resolution_mhz + (header_on.observing_frequency as f32 / 1e6);

    // Calculate median and SNR
    let mut sorted_spec = normalized_spec.to_vec();
    sorted_spec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_spec[sorted_spec.len() / 2];
    let snr = if median > 1e-9 { peak_val / median } else { 0.0 };

    // Plotting normalized spectrum
    let plot_filename = output_dir.join(format!("{}_vs_{}_maser.png", on_stem, off_stem));
    
    let plot_data: Vec<(f32, f32)> = freq_range.iter().zip(normalized_spec.iter()).map(|(&x, &y)| (x, y)).collect();

    plot_maser_spectrum(
        &plot_filename,
        &plot_data,
        peak_freq_mhz,
        peak_val,
        snr,
    )?;

    println!("Maser analysis plot saved to: {:?}", plot_filename);

    Ok(())
}

fn plot_maser_spectrum(
    output_path: &Path,
    data: &[(f32, f32)],
    peak_freq: f32,
    peak_val: f32,
    snr: f32,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let (min_freq, max_freq) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), (f, _)| (min.min(*f), max.max(*f)));
    let (min_val, max_val) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), (_, v)| (min.min(*v), max.max(*v)));

    let y_margin = (max_val - min_val) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Maser Analysis: (ON-OFF)/OFF Spectrum", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(min_freq..max_freq, (min_val - y_margin)..(max_val + y_margin))?;

    chart.configure_mesh()
        .x_desc("Frequency [MHz]")
        .y_desc("Normalized Intensity")
        .label_style(("sans-serif", 20))
        .draw()?;

    chart.draw_series(LineSeries::new(data.iter().cloned(), &BLUE))?;

    // Draw legend as text
    let style = TextStyle::from(("sans-serif", 20)).color(&BLACK);
    let legend_lines = vec![
        format!("Peak Freq: {:.4} MHz", peak_freq),
        format!("Peak Value: {:.4}", peak_val),
        format!("SNR: {:.2}", snr),
    ];
    let mut y_pos = 40;
    for line in legend_lines {
        root.draw(&Text::new(line, (800, y_pos), style.clone()))?;
        y_pos += 25;
    }

    root.present()?;
    Ok(())
}

fn plot_on_off_spectra(
    output_path: &Path,
    on_data: &[(f32, f32)],
    off_data: &[(f32, f32)],
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let (min_freq, max_freq) = on_data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), (f, _)| (min.min(*f), max.max(*f)));
    
    let on_max = on_data.iter().fold(f32::NEG_INFINITY, |max, (_, v)| max.max(*v));
    let off_max = off_data.iter().fold(f32::NEG_INFINITY, |max, (_, v)| max.max(*v));
    let max_val = on_max.max(off_max);

    let on_min = on_data.iter().fold(f32::INFINITY, |min, (_, v)| min.min(*v));
    let off_min = off_data.iter().fold(f32::INFINITY, |min, (_, v)| min.min(*v));
    let min_val = on_min.min(off_min);

    let y_margin = (max_val - min_val) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("ON-source vs OFF-source Spectrum", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(min_freq..max_freq, (min_val - y_margin)..(max_val + y_margin))?;

    chart.configure_mesh()
        .x_desc("Frequency [MHz]")
        .y_desc("Intensity")
        .label_style(("sans-serif", 20))
        .draw()?;

    chart.draw_series(LineSeries::new(on_data.iter().cloned(), &RED))?
        .label("ON Source")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));

    chart.draw_series(LineSeries::new(off_data.iter().cloned(), &BLUE))?
        .label("OFF Source")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
