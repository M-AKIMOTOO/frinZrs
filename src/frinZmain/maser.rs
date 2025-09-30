use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use ndarray::{Array1, Axis};
use plotters::prelude::*;

use crate::args::Args;
use crate::header::{parse_header, CorHeader};
use crate::read::read_visibility_data;
use crate::fft::process_fft;
use crate::rfi::parse_rfi_ranges;

const C_KM_S: f64 = 299792.458; // Speed of light in km/s

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
    // 1. Parse args
    println!("Running Maser Analysis...");
    let on_source_path = args.input.as_ref().unwrap();
    let off_source_path = PathBuf::from(&args.maser[0]);
    let velocity_correction: f64 = if args.maser.len() > 1 { args.maser[1].parse()? } else { 0.0 };
    let rest_freq_mhz: f64 = if args.maser.len() > 2 { args.maser[2].parse()? } else { 6668.5192 };

    println!("  ON Source: {:?}", on_source_path);
    println!("  OFF Source: {:?}", off_source_path);
    println!("  Rest Frequency: {} MHz", rest_freq_mhz);
    println!("  Velocity Correction: {} km/s", velocity_correction);

    // 2. Get full spectra
    let (header_on, spec_on) = get_spectrum_from_file(on_source_path, args)?;
    let (_header_off, spec_off) = get_spectrum_from_file(&off_source_path, args)?;

    // 3. Setup paths and full-range vectors
    let parent_dir = on_source_path.parent().unwrap_or_else(|| Path::new(""));
    let output_dir = parent_dir.join("frinZ").join("maser");
    fs::create_dir_all(&output_dir)?;
    let on_stem = on_source_path.file_stem().unwrap().to_str().unwrap();
    let off_stem = off_source_path.file_stem().unwrap().to_str().unwrap();

    let freq_resolution_mhz = (header_on.sampling_speed as f64 / 2.0 / 1e6) / (header_on.fft_point as f64 / 2.0);
    let freq_range_mhz: Vec<f64> = (0..header_on.fft_point as usize / 2)
        .map(|i| i as f64 * freq_resolution_mhz + (header_on.observing_frequency / 1e6))
        .collect();
    let velocity_range_kms: Vec<f64> = freq_range_mhz.iter()
        .map(|&f_obs| C_KM_S * (rest_freq_mhz - f_obs) / rest_freq_mhz - velocity_correction)
        .collect();


    // 5. Conditional analysis range selection
    let analysis_indices: Vec<usize> = if rest_freq_mhz >= 6600.0 && rest_freq_mhz <= 7112.0 {
        println!("  C-band maser detected. Restricting analysis to 6660-6675 MHz range.");
        freq_range_mhz.iter().enumerate()
            .filter(|(_, &freq)| freq >= 6660.0 && freq <= 6675.0)
            .map(|(i, _)| i)
            .collect()
    } else {
        (0..freq_range_mhz.len()).collect()
    };

    if analysis_indices.is_empty() {
        return Err("No data found in the specified frequency range for analysis.".into());
    }

    // 6. Create data slices for analysis
    let analysis_freq_mhz: Vec<f64> = analysis_indices.iter().map(|&i| freq_range_mhz[i]).collect();
    let analysis_velocity_kms: Vec<f64> = analysis_indices.iter().map(|&i| velocity_range_kms[i]).collect();
    let analysis_spec_on: Vec<f32> = analysis_indices.iter().map(|&i| spec_on[i]).collect();
    let analysis_spec_off: Vec<f32> = analysis_indices.iter().map(|&i| spec_off[i]).collect();

    // 7. Write NARROWED data to TSV
    let tsv_filename = output_dir.join(format!("{}_vs_{}_data.tsv", on_stem, off_stem));
    let mut file = File::create(&tsv_filename)?;
    let base_freq_mhz = header_on.observing_frequency / 1e6;
    writeln!(file, "# Base Frequency (MHz): {}", base_freq_mhz)?;
    writeln!(file, "Frequency_Offset_MHz\tVelocity_km/s\tonsourc\toffsource")?;
    for i in 0..analysis_freq_mhz.len() {
        let freq_offset_mhz = analysis_freq_mhz[i] - base_freq_mhz;
        writeln!(file, "{}\t{}\t{}\t{}", freq_offset_mhz, analysis_velocity_kms[i], analysis_spec_on[i], analysis_spec_off[i])?;
    }
    println!("Maser data saved to: {:?}", tsv_filename);

    let mut normalized_spec = Array1::<f32>::zeros(analysis_indices.len());
    for i in 0..analysis_indices.len() {
        if analysis_spec_off[i] > 1e-9 {
            normalized_spec[i] = (analysis_spec_on[i] - analysis_spec_off[i]) / analysis_spec_off[i];
        } else {
            normalized_spec[i] = 0.0;
        }
    }

    // 7. Find peak on (potentially narrowed) normalized spectrum
    let mut peak_val = f32::NEG_INFINITY;
    let mut peak_idx_in_analysis = 0;
    for (i, &val) in normalized_spec.iter().enumerate() {
        if val > peak_val {
            peak_val = val;
            peak_idx_in_analysis = i;
        }
    }
    let peak_freq_mhz = analysis_freq_mhz[peak_idx_in_analysis];
    let peak_velocity_kms = analysis_velocity_kms[peak_idx_in_analysis];
    let mut sorted_spec = normalized_spec.to_vec();
    sorted_spec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_spec[sorted_spec.len() / 2];
    let snr = if median > 1e-9 { peak_val / median } else { 0.0 };

    // 8. Generate plots
    // Plot 1: ON/OFF vs Frequency (Full Range)
    let on_plot_data_freq: Vec<(f64, f32)> = freq_range_mhz.iter().zip(spec_on.iter()).map(|(&x, &y)| (x, y)).collect();
    let off_plot_data_freq: Vec<(f64, f32)> = freq_range_mhz.iter().zip(spec_off.iter()).map(|(&x, &y)| (x, y)).collect();
    let on_off_freq_plot_filename = output_dir.join(format!("{}_vs_{}_on_off_freq.png", on_stem, off_stem));
    plot_on_off_spectra(
        &on_off_freq_plot_filename,
        &on_plot_data_freq,
        &off_plot_data_freq,
        "Frequency [MHz]",
        peak_velocity_kms, // Use peak velocity found in analysis range
    )?;
    println!("ON vs OFF source frequency spectra plot saved to: {:?}", on_off_freq_plot_filename);

    // Plot 2: (ON-OFF)/OFF vs Frequency (Analysis Range)
    let normalized_plot_data_freq: Vec<(f64, f32)> = analysis_freq_mhz.iter().zip(normalized_spec.iter()).map(|(&x, &y)| (x, y)).collect();
    let maser_freq_plot_filename = output_dir.join(format!("{}_vs_{}_maser_freq.png", on_stem, off_stem));
    plot_maser_spectrum(
        &maser_freq_plot_filename,
        &normalized_plot_data_freq,
        "Frequency [MHz]",
        peak_freq_mhz,
        peak_velocity_kms,
        peak_val,
        snr,
    )?;
    println!("Maser frequency analysis plot saved to: {:?}", maser_freq_plot_filename);

    // Plot 3: (ON-OFF)/OFF vs Velocity (Analysis Range)
    let normalized_plot_data_vel: Vec<(f64, f32)> = analysis_velocity_kms.iter().zip(normalized_spec.iter()).map(|(&x, &y)| (x, y)).collect();
    let maser_vel_plot_filename = output_dir.join(format!("{}_vs_{}_maser_vel.png", on_stem, off_stem));
    plot_maser_spectrum(
        &maser_vel_plot_filename,
        &normalized_plot_data_vel,
        "Velocity [km/s]",
        peak_freq_mhz,
        peak_velocity_kms,
        peak_val,
        snr,
    )?;
    println!("Maser velocity analysis plot saved to: {:?}", maser_vel_plot_filename);

    // Plot 4: Zoomed-in (ON-OFF)/OFF vs Velocity
    let vel_window_kms = 10.0;
    let min_zoom_vel = peak_velocity_kms - vel_window_kms;
    let max_zoom_vel = peak_velocity_kms + vel_window_kms;
    let zoomed_plot_data: Vec<(f64, f32)> = analysis_velocity_kms.iter() // Use analysis range for zoom
        .zip(normalized_spec.iter())
        .filter(|(&vel, _)| vel >= min_zoom_vel && vel <= max_zoom_vel)
        .map(|(&vel, &norm_val)| (vel, norm_val))
        .collect();

    if !zoomed_plot_data.is_empty() {
        let maser_zoom_plot_filename = output_dir.join(format!("{}_vs_{}_maser_vel_zoom.png", on_stem, off_stem));
        plot_maser_spectrum(
            &maser_zoom_plot_filename,
            &zoomed_plot_data,
            "Velocity [km/s]",
            peak_freq_mhz,
            peak_velocity_kms,
            peak_val,
            snr,
        )?;
        println!("Zoomed maser velocity plot saved to: {:?}", maser_zoom_plot_filename);
    }

    Ok(())
}

fn plot_maser_spectrum(
    output_path: &Path,
    data: &[(f64, f32)],
    x_label: &str,
    peak_freq: f64,
    peak_velocity: f64,
    peak_val: f32,
    snr: f32,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let (min_x, max_x) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| (min.min(*x), max.max(*x)));
    let (min_y, max_y) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), (_, y)| (min.min(*y), max.max(*y)));

    let y_margin = (max_y - min_y) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Maser Analysis: (ON-OFF)/OFF Spectrum", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(min_x..max_x, (min_y - y_margin)..(max_y + y_margin))?;

    chart.configure_mesh()
        .x_desc(x_label)
        .y_desc("Normalized Intensity")
        .label_style(("sans-serif", 20))
        .draw()?;

    chart.draw_series(LineSeries::new(data.iter().cloned(), &BLUE))?;

    // Draw legend as text
    let style = TextStyle::from(("sans-serif", 20)).color(&BLACK);
    let legend_lines = vec![
        format!("Peak Freq: {:.4} MHz", peak_freq),
        format!("Peak Velocity: {:.2} km/s", peak_velocity),
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
    on_data: &[(f64, f32)],
    off_data: &[(f64, f32)],
    x_label: &str,
    peak_velocity: f64,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let (min_x, max_x) = on_data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| (min.min(*x), max.max(*x)));
    
    let on_max = on_data.iter().fold(f32::NEG_INFINITY, |max, (_, v)| max.max(*v));
    let off_max = off_data.iter().fold(f32::NEG_INFINITY, |max, (_, v)| max.max(*v));
    let max_y = on_max.max(off_max);

    let on_min = on_data.iter().fold(f32::INFINITY, |min, (_, v)| min.min(*v));
    let off_min = off_data.iter().fold(f32::INFINITY, |min, (_, v)| min.min(*v));
    let min_y = on_min.min(off_min);

    let y_margin = (max_y - min_y) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("ON-source vs OFF-source Spectrum", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(min_x..max_x, (min_y - y_margin)..(max_y + y_margin))?;

    chart.configure_mesh()
        .x_desc(x_label)
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

    // Draw legend as text
    let style = TextStyle::from(("sans-serif", 20)).color(&BLACK);
    let legend_text = format!("Peak Velocity: {:.2} km/s", peak_velocity);
    root.draw(&Text::new(legend_text, (800, 40), style))?;

    root.present()?;
    Ok(())
}
