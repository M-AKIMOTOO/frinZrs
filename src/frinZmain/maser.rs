use ndarray::{Array1, Axis};
use plotters::prelude::*;
use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use crate::args::Args;
use crate::fft::process_fft;
use crate::header::{parse_header, CorHeader};
use crate::read::read_visibility_data;
use crate::rfi::parse_rfi_ranges;

// New use statements for LSR velocity correction
use astro::ecliptic;
use astro::nutation;
use astro::planet::{self, earth, Planet};
use astro::time::{self, Date};

use chrono::{DateTime, Datelike, Timelike, Utc};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{storage::Owned, DMatrix, DVector, Dyn, Matrix3, Vector3};

const C_KM_S: f64 = 299792.458; // Speed of light in km/s
const FWHM_TO_SIGMA: f64 = 0.42466090014400953; // 1 / (2 * sqrt(2 ln 2))
const MIN_FWHM_KMS: f64 = 1.0e-3; // clamp to avoid zero-width components

#[derive(Debug)]
struct GaussianFitResult {
    components: Vec<(f64, f64, f64)>,
    residual_norm: f64,
    termination: TerminationReason,
    evaluations: usize,
}

struct GaussianMixtureProblem<'a> {
    velocities: &'a [f64],
    values: &'a [f64],
    params: DVector<f64>,
    components: usize,
}

impl<'a> GaussianMixtureProblem<'a> {
    fn new(velocities: &'a [f64], values: &'a [f64], initial: &[f64]) -> Self {
        let params = DVector::from_column_slice(initial);
        let mut problem = Self {
            velocities,
            values,
            params,
            components: initial.len() / 3,
        };
        problem.enforce_constraints();
        problem
    }

    fn enforce_constraints(&mut self) {
        for idx in 0..self.components {
            let width_idx = idx * 3 + 2;
            let val = self.params[width_idx].abs().max(MIN_FWHM_KMS);
            self.params[width_idx] = val;
        }
    }

    fn components(&self) -> Vec<(f64, f64, f64)> {
        let mut out = Vec::with_capacity(self.components);
        for idx in 0..self.components {
            let base = idx * 3;
            out.push((
                self.params[base],
                self.params[base + 1],
                self.params[base + 2].max(MIN_FWHM_KMS),
            ));
        }
        out
    }

    fn predicted_value(params: &DVector<f64>, components: usize, vel: f64) -> f64 {
        let mut total = 0.0;
        for idx in 0..components {
            let base = idx * 3;
            let amp = params[base];
            let center = params[base + 1];
            let fwhm = params[base + 2].abs().max(MIN_FWHM_KMS);
            let sigma = fwhm * FWHM_TO_SIGMA;
            let delta = vel - center;
            let sigma_sq = sigma * sigma;
            let exponent = -delta * delta / (2.0 * sigma_sq);
            total += amp * exponent.exp();
        }
        total
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, Dyn> for GaussianMixtureProblem<'a> {
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    type ParameterStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.params.copy_from(x);
        self.enforce_constraints();
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        if self.velocities.len() != self.values.len() {
            return None;
        }
        let mut residuals = DVector::zeros(self.velocities.len());
        for (idx, (&vel, &target)) in self.velocities.iter().zip(self.values.iter()).enumerate() {
            let predicted = Self::predicted_value(&self.params, self.components, vel);
            residuals[idx] = predicted - target;
        }
        Some(residuals)
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        if self.velocities.len() != self.values.len() {
            return None;
        }
        let rows = self.velocities.len();
        let cols = self.components * 3;
        let mut jac = DMatrix::zeros(rows, cols);
        for (row_idx, &vel) in self.velocities.iter().enumerate() {
            for component_idx in 0..self.components {
                let base = component_idx * 3;
                let amp = self.params[base];
                let center = self.params[base + 1];
                let fwhm = self.params[base + 2].abs().max(MIN_FWHM_KMS);
                let sigma = fwhm * FWHM_TO_SIGMA;
                let delta = vel - center;
                let sigma_sq = sigma * sigma;
                let exp_term = (-delta * delta / (2.0 * sigma_sq)).exp();
                let value = amp * exp_term;

                jac[(row_idx, base)] = exp_term; // ∂/∂amp
                jac[(row_idx, base + 1)] = value * (delta / sigma_sq); // ∂/∂center

                let sigma_cubed = sigma_sq * sigma;
                jac[(row_idx, base + 2)] = value * ((delta * delta) / sigma_cubed) * FWHM_TO_SIGMA;
                // ∂/∂fwhm
            }
        }
        Some(jac)
    }
}

fn flatten_gaussian_components(components: &[(f64, f64, f64)]) -> Vec<f64> {
    let mut flat = Vec::with_capacity(components.len() * 3);
    for &(amp, center, fwhm) in components {
        flat.push(amp);
        flat.push(center);
        flat.push(fwhm);
    }
    flat
}

fn evaluate_gaussian_mixture(
    velocities: &[f64],
    components: &[(f64, f64, f64)],
) -> Vec<(f64, f32)> {
    velocities
        .iter()
        .map(|&vel| {
            let mut total = 0.0;
            for &(amp, center, fwhm) in components {
                let sigma = fwhm.abs().max(MIN_FWHM_KMS) * FWHM_TO_SIGMA;
                let delta = vel - center;
                let sigma_sq = sigma * sigma;
                total += amp * (-delta * delta / (2.0 * sigma_sq)).exp();
            }
            (vel, total as f32)
        })
        .collect()
}

fn fit_gaussian_mixture(
    velocities: &[f64],
    values: &[f64],
    initial_components: &[(f64, f64, f64)],
) -> Result<GaussianFitResult, Box<dyn Error>> {
    if velocities.len() != values.len() {
        return Err("Velocity and spectrum length mismatch.".into());
    }
    if initial_components.is_empty() {
        return Err("No Gaussian components provided.".into());
    }

    let initial_flat = flatten_gaussian_components(initial_components);
    let problem = GaussianMixtureProblem::new(velocities, values, &initial_flat);
    let (problem, report) = LevenbergMarquardt::new().minimize(problem);

    let residual_vec = problem
        .residuals()
        .unwrap_or_else(|| DVector::zeros(velocities.len()));
    let result = GaussianFitResult {
        components: problem.components(),
        residual_norm: residual_vec.norm(),
        termination: report.termination,
        evaluations: report.number_of_evaluations,
    };
    Ok(result)
}

/// Extracts the cross-power spectrum at zero fringe rate from a .cor file.
fn get_spectrum_from_file(
    file_path: &Path,
    args: &Args,
    sampling_scale: f64,
) -> Result<(CorHeader, Array1<f32>), Box<dyn Error>> {
    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let mut cursor = Cursor::new(buffer.as_slice());

    let header = parse_header(&mut cursor)?;

    let scaled_sampling_speed = (header.sampling_speed as f64 * sampling_scale) as f32;
    let sampling_speed_for_fft = (header.sampling_speed as f64 * sampling_scale).round() as i32;

    let (complex_vec, _, _) = read_visibility_data(
        &mut cursor,
        &header,
        header.number_of_sector,
        0,
        0,
        false,
        &[],
    )?;

    let rbw = (scaled_sampling_speed / header.fft_point as f32) / 1e6;
    let rfi_ranges = parse_rfi_ranges(&args.rfi, rbw)?;

    let (freq_rate_array, padding_length) = process_fft(
        &complex_vec,
        header.number_of_sector,
        header.fft_point,
        sampling_speed_for_fft,
        &rfi_ranges,
        args.rate_padding,
    );

    // Get spectrum at zero rate (center of rate dimension)
    let zero_rate_idx = padding_length / 2;
    let spectrum_complex = freq_rate_array.index_axis(Axis(1), zero_rate_idx);

    let spectrum_abs = spectrum_complex.mapv(|x| x.norm());

    Ok((header, spectrum_abs))
}

fn plot_maser_spectrum(
    output_path: &Path,
    data: &[(f64, f32)],
    x_label: &str,
    peak_freq: f64,
    peak_velocity: f64,
    peak_val: f32,
    snr: f32,
    freq_resolution_mhz: f64,
    channel_width_kms: f64,
    gaussian_fit: Option<&[(f64, f32)]>,
    gaussian_params: Option<&[(f64, f64, f64)]>,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let (min_x, max_x) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| {
            (min.min(*x), max.max(*x))
        });
    let (min_y, max_y) = data
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), (_, y)| {
            (min.min(*y), max.max(*y))
        });

    let y_margin = (max_y - min_y) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Maser Analysis: (ON-OFF)/OFF Spectrum",
            ("sans-serif", 30).into_font(),
        )
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(min_x..max_x, (min_y - y_margin)..(max_y + y_margin))?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc("Normalized Intensity")
        .label_style(("sans-serif", 20))
        .draw()?;

    chart.draw_series(LineSeries::new(data.iter().cloned(), &BLUE))?;

    if let Some(fit_data) = gaussian_fit {
        chart.draw_series(LineSeries::new(fit_data.iter().cloned(), &RED))?;
    }

    // Draw legend as text
    let style = TextStyle::from(("sans-serif", 20)).color(&BLACK);
    let legend_lines = vec![
        format!("Peak Freq: {:.4} MHz", peak_freq),
        format!("Peak Velocity: {:.2} km/s", peak_velocity),
        format!("Peak Value: {:.4}", peak_val),
        format!("SNR: {:.2}", snr),
        format!("Δf: {:.4} MHz", freq_resolution_mhz),
        format!("Δv: {:.3} km/s", channel_width_kms),
    ];
    let mut y_pos = 40;
    for line in legend_lines {
        root.draw(&Text::new(line, (800, y_pos), style.clone()))?;
        y_pos += 25;
    }

    if let Some(params) = gaussian_params {
        root.draw(&Text::new(
            "Gaussian Components:",
            (800, y_pos),
            style.clone(),
        ))?;
        y_pos += 25;
        for (idx, (amp, center, fwhm)) in params.iter().enumerate() {
            let text = format!(
                "G{}: amp={:.4}, v={:.3} km/s, FWHM={:.3} km/s",
                idx + 1,
                amp,
                center,
                fwhm
            );
            root.draw(&Text::new(text, (800, y_pos), style.clone()))?;
            y_pos += 25;
        }
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

    let (min_x, max_x) = on_data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| {
            (min.min(*x), max.max(*x))
        });

    let on_max = on_data
        .iter()
        .fold(f32::NEG_INFINITY, |max, (_, v)| max.max(*v));
    let off_max = off_data
        .iter()
        .fold(f32::NEG_INFINITY, |max, (_, v)| max.max(*v));
    let max_y = on_max.max(off_max);

    let on_min = on_data
        .iter()
        .fold(f32::INFINITY, |min, (_, v)| min.min(*v));
    let off_min = off_data
        .iter()
        .fold(f32::INFINITY, |min, (_, v)| min.min(*v));
    let min_y = on_min.min(off_min);

    let y_margin = (max_y - min_y) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "ON-source vs OFF-source Spectrum",
            ("sans-serif", 30).into_font(),
        )
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(min_x..max_x, (min_y - y_margin)..(max_y + y_margin))?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc("Intensity")
        .label_style(("sans-serif", 20))
        .draw()?;

    chart
        .draw_series(LineSeries::new(on_data.iter().cloned(), &RED))?
        .label("ON Source")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));

    chart
        .draw_series(LineSeries::new(off_data.iter().cloned(), &BLUE))?
        .label("OFF Source")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));

    chart
        .configure_series_labels()
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

pub fn run_maser_analysis(args: &Args) -> Result<(), Box<dyn Error>> {
    // 1. Parse args
    println!("Running Maser Analysis...");
    let on_source_path = args.input.as_ref().unwrap();

    let mut off_source_path: Option<PathBuf> = None;
    let mut rest_freq_mhz: f64 = 6668.5192;
    let mut rest_freq_overridden = false;
    let mut override_vlsr: Option<f64> = None;
    let mut corrfreq: f64 = 1.0;
    let mut user_band_range: Option<(f64, f64)> = None;
    let mut gaussian_initial_components: Vec<(f64, f64, f64)> = Vec::new();
    let mut positional_args: Vec<&String> = Vec::new();

    for entry in &args.maser {
        if let Some((key, value)) = entry.split_once(':') {
            match key.trim().to_lowercase().as_str() {
                "off" => {
                    off_source_path = Some(PathBuf::from(value.trim()));
                }
                "rest" => {
                    rest_freq_mhz = value.trim().parse()?;
                    rest_freq_overridden = true;
                }
                "vlst" => {
                    override_vlsr = Some(value.trim().parse()?);
                }
                "corrfreq" => {
                    corrfreq = value.trim().parse()?;
                }
                "band" => {
                    let mut parts = value.trim().split('-');
                    let start: f64 = parts
                        .next()
                        .ok_or("Error: band requires start-end in MHz offset.")?
                        .parse()?;
                    let end: f64 = parts
                        .next()
                        .ok_or("Error: band requires start-end in MHz offset.")?
                        .parse()?;
                    if parts.next().is_some() {
                        return Err(
                            "Error: band accepts exactly one start-end pair (e.g., band:60-70)."
                                .into(),
                        );
                    }
                    if start >= end {
                        return Err("Error: band start must be less than band end.".into());
                    }
                    user_band_range = Some((start, end));
                }
                "gauss" => {
                    let params: Vec<&str> = value
                        .split(',')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if params.is_empty() {
                        return Err("Error: gauss requires amp,Vlst,fwhm values.".into());
                    }
                    if params.len() % 3 != 0 {
                        return Err(
                            format!(
                                "Error: gauss expects triples of amp,Vlst,fwhm. Received {} entries: {}",
                                params.len(),
                                params.join(", ")
                            )
                            .into(),
                        );
                    }
                    for chunk in params.chunks(3) {
                        let amp: f64 = chunk[0].parse()?;
                        let center: f64 = chunk[1].parse()?;
                        let fwhm: f64 = chunk[2].parse()?;
                        if fwhm <= 0.0 {
                            return Err("Error: gauss FWHM must be positive.".into());
                        }
                        gaussian_initial_components.push((amp, center, fwhm));
                    }
                }
                other => {
                    return Err(format!(
                        "Unknown --maser parameter '{}'. Expected off, rest, Vlst, corrfreq, band, gauss.",
                        other
                    )
                    .into());
                }
            }
        } else {
            positional_args.push(entry);
        }
    }

    if off_source_path.is_none() {
        if let Some(first) = positional_args.get(0) {
            off_source_path = Some(PathBuf::from(first.as_str()));
        }
    }

    if off_source_path.is_none() {
        return Err(
            "Error: --maser requires an off-source file via 'off:<PATH>' or positional argument."
                .into(),
        );
    }

    if positional_args.len() >= 2 {
        if let Ok(rest_val) = positional_args[1].parse::<f64>() {
            rest_freq_mhz = rest_val;
            rest_freq_overridden = true;
        }
    }

    let off_source_path = off_source_path.unwrap();

    println!("  ON Source: {:?}", on_source_path);
    println!("  OFF Source: {:?}", off_source_path);
    println!("  Rest Frequency: {} MHz", rest_freq_mhz);
    println!("  Frequency Correction Factor: {:.6}", corrfreq);
    if let Some(v) = override_vlsr {
        println!("  Override LSR Velocity Correction: {:.6} km/s", v);
    }
    if let Some((start, end)) = user_band_range {
        println!(
            "  Frequency window offsets: {:.3} MHz to {:.3} MHz relative to observing frequency",
            start, end
        );
    }
    if !gaussian_initial_components.is_empty() {
        println!("  Gaussian initial guesses (amp, center[km/s], fwhm[km/s]):");
        for (amp, center, fwhm) in &gaussian_initial_components {
            println!("    {:.4}, {:.4}, {:.4}", amp, center, fwhm);
        }
    }

    // Get full spectra
    let (header_on, spec_on) = get_spectrum_from_file(on_source_path, args, corrfreq)?;
    let (header_off, spec_off) = get_spectrum_from_file(&off_source_path, args, corrfreq)?;

    if header_on.fft_point != header_off.fft_point
        || header_on.observing_frequency.to_bits() != header_off.observing_frequency.to_bits()
        || header_on.sampling_speed != header_off.sampling_speed
    {
        return Err(format!(
            "Error: Header mismatch between {:?} and {:?}.\n  fft_point: {} (ON) vs {} (OFF)\n  observing_frequency: {:.6} Hz (ON) vs {:.6} Hz (OFF)\n  sampling_speed: {} Hz (ON) vs {} Hz (OFF)",
            on_source_path,
            off_source_path,
            header_on.fft_point,
            header_off.fft_point,
            header_on.observing_frequency,
            header_off.observing_frequency,
            header_on.sampling_speed,
            header_off.sampling_speed
        )
        .into());
    }

    // Get obs_time from the first sector of the ON source file
    let mut on_file = File::open(on_source_path)?;
    let mut on_buffer = Vec::new();
    on_file.read_to_end(&mut on_buffer)?;
    let mut on_cursor = Cursor::new(on_buffer.as_slice());
    let (_, on_obs_time, _) = crate::read::read_visibility_data(
        &mut on_cursor,
        &header_on,
        1, // length
        0, // skip
        0, // loop_index
        false,
        &[],
    )?;

    // Calculate LSR velocity correction
    let mut lsr_vel_corr = calculate_lsr_velocity_correction(
        header_on.station1_position, // Using station1_position as the observer location
        &on_obs_time,
        header_on.source_position_ra,
        header_on.source_position_dec,
    );
    if let Some(v) = override_vlsr {
        lsr_vel_corr = v;
    }
    println!(
        "  Calculated LSR Velocity Correction: {:.6} km/s",
        lsr_vel_corr
    );

    if !rest_freq_overridden {
        let obs_freq_mhz = header_on.observing_frequency / 1e6;
        if (11923.0..=12435.0).contains(&obs_freq_mhz) {
            rest_freq_mhz = 1217.8597;
            println!(
                "  Rest frequency automatically set to 1217.8597 MHz for 12.2 GHz methanol maser analysis."
            );
        }
    }

    // 3. Setup paths and full-range vectors
    let parent_dir = on_source_path.parent().unwrap_or_else(|| Path::new(""));
    let output_dir = parent_dir.join("frinZ").join("maser");
    fs::create_dir_all(&output_dir)?;
    let on_stem = on_source_path.file_stem().unwrap().to_str().unwrap();

    let freq_resolution_mhz = (header_on.sampling_speed as f64 * corrfreq / 2.0 / 1e6)
        / (header_on.fft_point as f64 / 2.0);
    let channel_width_kms = C_KM_S * freq_resolution_mhz / rest_freq_mhz;
    println!("  Frequency Resolution: {:.6} MHz", freq_resolution_mhz);
    println!("  Channel Width: {:.6} km/s", channel_width_kms);
    let base_freq_mhz = header_on.observing_frequency / 1e6;
    let freq_range_mhz: Vec<f64> = (0..header_on.fft_point as usize / 2)
        .map(|i| i as f64 * freq_resolution_mhz + base_freq_mhz)
        .collect();
    let velocity_range_kms: Vec<f64> = freq_range_mhz
        .iter()
        .map(|&f_obs| C_KM_S * (rest_freq_mhz - f_obs) / rest_freq_mhz + lsr_vel_corr)
        .collect();

    // 5. Conditional analysis range selection
    let analysis_indices: Vec<usize> = if let Some((start_offset, end_offset)) = user_band_range {
        let min_freq = base_freq_mhz + start_offset;
        let max_freq = base_freq_mhz + end_offset;
        freq_range_mhz
            .iter()
            .enumerate()
            .filter(|(_, &freq)| freq >= min_freq && freq <= max_freq)
            .map(|(i, _)| i)
            .collect()
    } else if rest_freq_mhz >= 6600.0 && rest_freq_mhz <= 7112.0 {
        println!("  C-band maser detected. Restricting analysis to 6660-6675 MHz range.");
        freq_range_mhz
            .iter()
            .enumerate()
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
    let analysis_freq_mhz: Vec<f64> = analysis_indices
        .iter()
        .map(|&i| freq_range_mhz[i])
        .collect();
    let analysis_velocity_kms: Vec<f64> = analysis_indices
        .iter()
        .map(|&i| velocity_range_kms[i])
        .collect();
    let analysis_spec_on: Vec<f32> = analysis_indices.iter().map(|&i| spec_on[i]).collect();
    let analysis_spec_off: Vec<f32> = analysis_indices.iter().map(|&i| spec_off[i]).collect();

    let mut normalized_spec = Array1::<f32>::zeros(analysis_indices.len());
    for i in 0..analysis_indices.len() {
        if analysis_spec_off[i] > 1e-9 {
            normalized_spec[i] =
                (analysis_spec_on[i] - analysis_spec_off[i]) / analysis_spec_off[i];
        } else {
            normalized_spec[i] = 0.0;
        }
    }

    let normalized_spec_f64: Vec<f64> = normalized_spec.iter().map(|&v| v as f64).collect();
    let mut gaussian_fit_summary: Option<GaussianFitResult> = None;
    let mut gaussian_fit_components: Option<Vec<(f64, f64, f64)>> = None;
    let mut gaussian_fit_data_vel: Option<Vec<(f64, f32)>> = None;

    if !gaussian_initial_components.is_empty() {
        match fit_gaussian_mixture(
            &analysis_velocity_kms,
            &normalized_spec_f64,
            &gaussian_initial_components,
        ) {
            Ok(result) => {
                println!(
                    "  Gaussian fit termination: {:?}, residual norm: {:.6}, evaluations: {}",
                    result.termination, result.residual_norm, result.evaluations
                );
                gaussian_fit_data_vel = Some(evaluate_gaussian_mixture(
                    &analysis_velocity_kms,
                    &result.components,
                ));
                gaussian_fit_components = Some(result.components.clone());
                gaussian_fit_summary = Some(result);
            }
            Err(err) => {
                eprintln!(
                    "Warning: Gaussian fit failed ({}). Using initial parameters for overlay.",
                    err
                );
                gaussian_fit_data_vel = Some(evaluate_gaussian_mixture(
                    &analysis_velocity_kms,
                    &gaussian_initial_components,
                ));
                gaussian_fit_components = Some(gaussian_initial_components.clone());
            }
        }
    }

    if let Some(ref comps) = gaussian_fit_components {
        println!("  Gaussian fit result (amp, center[km/s], fwhm[km/s]):");
        for (idx, (amp, center, fwhm)) in comps.iter().enumerate() {
            println!("    G{}: {:.4}, {:.4}, {:.4}", idx + 1, amp, center, fwhm);
        }
    }

    if let Some(summary) = &gaussian_fit_summary {
        println!(
            "  Gaussian fit residual norm: {:.6} (objective: {:.6})",
            summary.residual_norm,
            0.5 * summary.residual_norm * summary.residual_norm
        );
    }

    // Write NARROWED data and final fit parameters to TSV
    let tsv_filename = output_dir.join(format!("{}_maser_data.tsv", on_stem));
    let mut file = File::create(&tsv_filename)?;
    writeln!(file, "# Base Frequency (MHz): {}", base_freq_mhz)?;
    writeln!(
        file,
        "Frequency_Offset_MHz\tVelocity_km/s\tonsourc\toffsource"
    )?;
    for i in 0..analysis_freq_mhz.len() {
        let freq_offset_mhz = analysis_freq_mhz[i] - base_freq_mhz;
        writeln!(
            file,
            "{}\t{}\t{}\t{}",
            freq_offset_mhz, analysis_velocity_kms[i], analysis_spec_on[i], analysis_spec_off[i]
        )?;
    }

    if let Some(ref comps) = gaussian_fit_components {
        let fit_filename = output_dir.join(format!("{}_maser_fit.txt", on_stem));
        let mut fit_file = File::create(&fit_filename)?;
        writeln!(fit_file, "Gaussian Fit Summary")?;
        if let Some(summary) = &gaussian_fit_summary {
            writeln!(fit_file, "Termination: {:?}", summary.termination)?;
            writeln!(fit_file, "Evaluations: {}", summary.evaluations)?;
            writeln!(fit_file, "Residual norm: {:.6}", summary.residual_norm)?;
            writeln!(
                fit_file,
                "Objective (0.5 * residual_norm^2): {:.6}",
                0.5 * summary.residual_norm * summary.residual_norm
            )?;
        } else {
            writeln!(
                fit_file,
                "Warning: Fit failed to converge. Recording initial parameters."
            )?;
        }
        writeln!(fit_file, "")?;
        writeln!(
            fit_file,
            "# Gaussian Components (amp, velocity[km/s], FWHM[km/s])"
        )?;
        for (idx, (amp, center, fwhm)) in comps.iter().enumerate() {
            writeln!(
                fit_file,
                "G{}: {:.6}\t{:.6}\t{:.6}",
                idx + 1,
                amp,
                center,
                fwhm
            )?;
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
    let snr = if median > 1e-9 {
        peak_val / median
    } else {
        0.0
    };

    // 8. Generate plots
    // Plot 1: ON/OFF vs Frequency (Full Range)
    let on_plot_data_freq: Vec<(f64, f32)> = freq_range_mhz
        .iter()
        .zip(spec_on.iter())
        .map(|(&x, &y)| (x, y))
        .collect();
    let off_plot_data_freq: Vec<(f64, f32)> = freq_range_mhz
        .iter()
        .zip(spec_off.iter())
        .map(|(&x, &y)| (x, y))
        .collect();
    let on_off_freq_plot_filename = output_dir.join(format!("{}_on_off_freq.png", on_stem));
    plot_on_off_spectra(
        &on_off_freq_plot_filename,
        &on_plot_data_freq,
        &off_plot_data_freq,
        "Frequency [MHz]",
        peak_velocity_kms, // Use peak velocity found in analysis range
    )?;

    // Plot 2: (ON-OFF)/OFF vs Frequency (Analysis Range)
    let normalized_plot_data_freq: Vec<(f64, f32)> = analysis_freq_mhz
        .iter()
        .zip(normalized_spec.iter())
        .map(|(&x, &y)| (x, y))
        .collect();
    let maser_freq_plot_filename = output_dir.join(format!("{}_maser_freq.png", on_stem));
    plot_maser_spectrum(
        &maser_freq_plot_filename,
        &normalized_plot_data_freq,
        "Frequency [MHz]",
        peak_freq_mhz,
        peak_velocity_kms,
        peak_val,
        snr,
        freq_resolution_mhz,
        channel_width_kms,
        None,
        None,
    )?;

    // Plot 3: (ON-OFF)/OFF vs Velocity (Analysis Range)
    let normalized_plot_data_vel: Vec<(f64, f32)> = analysis_velocity_kms
        .iter()
        .zip(normalized_spec.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    let maser_vel_plot_filename = output_dir.join(format!("{}_maser_vel.png", on_stem));
    plot_maser_spectrum(
        &maser_vel_plot_filename,
        &normalized_plot_data_vel,
        "LSR Velocity [km/s]",
        peak_freq_mhz,
        peak_velocity_kms,
        peak_val,
        snr,
        freq_resolution_mhz,
        channel_width_kms,
        gaussian_fit_data_vel.as_ref().map(|v| v.as_slice()),
        gaussian_fit_components
            .as_ref()
            .map(|components| components.as_slice()),
    )?;

    // Plot 4: Zoomed-in (ON-OFF)/OFF vs Velocity
    let vel_window_kms = 10.0;
    let min_zoom_vel = peak_velocity_kms - vel_window_kms;
    let max_zoom_vel = peak_velocity_kms + vel_window_kms;
    let zoomed_plot_data: Vec<(f64, f32)> = analysis_velocity_kms
        .iter() // Use analysis range for zoom
        .zip(normalized_spec.iter())
        .filter(|(&vel, _)| vel >= min_zoom_vel && vel <= max_zoom_vel)
        .map(|(&vel, &norm_val)| (vel, norm_val))
        .collect();

    if !zoomed_plot_data.is_empty() {
        let gaussian_fit_zoom: Option<Vec<(f64, f32)>> =
            gaussian_fit_data_vel.as_ref().map(|fit_data| {
                fit_data
                    .iter()
                    .filter(|(vel, _)| *vel >= min_zoom_vel && *vel <= max_zoom_vel)
                    .cloned()
                    .collect()
            });
        let maser_zoom_plot_filename = output_dir.join(format!("{}_maser_vel_zoom.png", on_stem));
        plot_maser_spectrum(
            &maser_zoom_plot_filename,
            &zoomed_plot_data,
            "LSR Velocity [km/s]",
            peak_freq_mhz,
            peak_velocity_kms,
            peak_val,
            snr,
            freq_resolution_mhz,
            channel_width_kms,
            gaussian_fit_zoom.as_ref().map(|v| v.as_slice()),
            gaussian_fit_components
                .as_ref()
                .map(|components| components.as_slice()),
        )?;
    }

    println!("make some plots in {:?}", output_dir);

    Ok(())
}

// New function starts here
pub fn calculate_lsr_velocity_correction(
    ant_position_geocentric: [f64; 3], // meters
    time: &DateTime<Utc>,
    obs_ra_rad: f64,  // radians
    obs_dec_rad: f64, // radians
) -> f64 {
    // returns correction in km/s

    // Constants for solar motion w.r.t. LSR (from Python script)
    // These are typically given in a specific coordinate system (e.g., ICRS-aligned at Sun's position)
    // Assuming these are in the same equatorial frame as the calculated v_obs_helio
    // The solar motion vector (v_sun_lsr) is derived from a standard solar motion model.
    // The components (U, V, W) are typically given in a coordinate system aligned with ICRS at the Sun's position.
    //
    // For reference, these values can be calculated using astropy as follows:
    //
    // # Define the direction of the Sun's motion
    // from astropy.coordinates import SkyCoord, LSR, CartesianDifferential
    // from astropy.time import Time
    // import astropy.units as u
    // import numpy as np
    //
    // dir_sun = SkyCoord(
    //     ra = 18 * 15 * u.deg,                   # R.A. = 18 h
    //     dec = 30 * u.deg,                       # Dec. = 30 deg
    //     frame = 'fk4',
    //     equinox = Time('B1900'), # 1900 年分点を指定する
    // ).galactic
    // # >>> dir_sun
    // # <SkyCoord (Galactic): (l, b) in deg
    // #     (56.15745659, 22.76480182)>
    //
    // # Define the speed of the Sun
    // v_sun = 20 * u.km / u.s
    //
    // # Decompose the solar velocity into components (U, V, W) in a coordinate system
    // # orthogonal to the galactic plane
    // U = v_sun * np.cos(dir_sun.b) * np.cos(dir_sun.l)
    // V = v_sun * np.cos(dir_sun.b) * np.sin(dir_sun.l)
    // W = v_sun * np.sin(dir_sun.b)
    //
    // # Convert to CartesianDifferential type
    // v_bary = CartesianDifferential(U, V, W)
    // # >>> v_bary
    // # <CartesianDifferential (d_x, d_y, d_z) in km / s
    // #     (10.27059164, 15.31741091, 7.73898381)>
    //
    // These components are then used as v_sun_lsr.
    // 銀河座標系 (U, V, W) → J2000 等赤道座標系 (ICRS) への変換行列（IAU 1958 定義）
    // 参考: Astropy 実装
    let gal_to_icrs = Matrix3::new(
        -0.054_875_560_416_215_4,
        0.494_109_427_875_583_7,
        -0.867_666_149_019_004_7,
        -0.873_437_090_234_885_0,
        -0.444_829_629_960_011_2,
        -0.198_076_373_431_201_5,
        -0.483_835_015_548_713_2,
        0.746_982_244_497_218_9,
        0.455_983_776_175_066_9,
    );

    let v_sun_lsr_gal = Vector3::new(10.270_591_64, 15.317_410_91, 7.738_983_81); // km/s (U, V, W)
    let v_sun_lsr = gal_to_icrs * v_sun_lsr_gal; // km/s（ICRS）

    // 1. Time conversion to Julian Day
    let decimal_day = time.day() as f64
        + time.hour() as f64 / 24.0
        + time.minute() as f64 / 1440.0
        + (time.second() as f64 + time.nanosecond() as f64 / 1e9) / 86400.0;
    let date = Date {
        year: time.year() as i16,
        month: time.month() as u8,
        decimal_day,
        cal_type: time::CalType::Gregorian,
    };
    let jd = time::julian_day(&date);

    // 2. Earth's orbital velocity v_orb (heliocentric equatorial frame)
    // Calculated by finite difference of Earth's heliocentric position
    let delta_jd = 0.001; // days for finite difference
    let au_to_km = 149597870.7; // km

    let sph_to_rect = |lon: f64, lat: f64, radius_au: f64| {
        let r_km = radius_au * au_to_km;
        let cos_lat = lat.cos();
        Vector3::new(
            r_km * cos_lat * lon.cos(),
            r_km * cos_lat * lon.sin(),
            r_km * lat.sin(),
        )
    };

    let (lon_minus, lat_minus, radius_minus) =
        planet::heliocent_coords(&Planet::Earth, jd - delta_jd);
    let pos_minus_ecl = sph_to_rect(lon_minus, lat_minus, radius_minus);

    let (lon_plus, lat_plus, radius_plus) = planet::heliocent_coords(&Planet::Earth, jd + delta_jd);
    let pos_plus_ecl = sph_to_rect(lon_plus, lat_plus, radius_plus);

    // Obliquity of the ecliptic
    let epsilon = ecliptic::mn_oblq_IAU(jd);

    // Rotation matrix from ecliptic to equatorial coordinates
    let rot_matrix_ecl_to_eq = nalgebra::Matrix3::new(
        1.0,
        0.0,
        0.0,
        0.0,
        epsilon.cos(),
        -epsilon.sin(),
        0.0,
        epsilon.sin(),
        epsilon.cos(),
    );

    let pos_minus_eq = rot_matrix_ecl_to_eq * pos_minus_ecl;
    let pos_plus_eq = rot_matrix_ecl_to_eq * pos_plus_ecl;

    // Orbital velocity (km/s)
    let v_orb = (pos_plus_eq - pos_minus_eq) / (2.0 * delta_jd * 86400.0); // 2*delta_jd is in days, convert to seconds

    // 3. Earth's rotational velocity v_rot (geocentric equatorial frame)
    let omega_magnitude_rad_per_day = earth::rot_angular_velocity(); // No arguments
    let omega_magnitude_rad_per_s = omega_magnitude_rad_per_day / 86400.0; // rad/s

    // Construct omega_vec assuming alignment with Z-axis in equatorial coordinates
    let omega_vec = Vector3::new(0.0, 0.0, omega_magnitude_rad_per_s);

    // Observer's geocentric position in km (ECEF frame)
    let r_obs_ecef_km = Vector3::new(
        ant_position_geocentric[0] / 1000.0,
        ant_position_geocentric[1] / 1000.0,
        ant_position_geocentric[2] / 1000.0,
    );

    // Transform r_obs_ecef_km from ECEF to equatorial (ICRS) frame
    // Incorporating precession and nutation (polar motion is ignored)

    // 1. Greenwich Mean Sidereal Time (GMST)
    let gmst_rad = time::mn_sidr(jd);

    // 2. Nutation angles
    let (delta_psi, delta_epsilon) = nutation::nutation(jd);

    // 3. Precession angles (ignored as per user's request)
    // let (zeta, z, theta) = precess::precession_angles(jd);

    // Mean obliquity of the ecliptic
    let mean_obliquity = ecliptic::mn_oblq_IAU(jd);

    // Rotation matrix for Earth Rotation Angle (GMST)
    let r_era = nalgebra::Matrix3::new(
        gmst_rad.cos(),
        -gmst_rad.sin(),
        0.0,
        gmst_rad.sin(),
        gmst_rad.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );

    // Nutation matrix (simplified, using mean obliquity)
    // This transforms from mean equinox of date to true equinox of date.
    // R_x(angle) = | 1  0           0          |
    //              | 0  cos(angle) -sin(angle) |
    //              | 0  sin(angle)  cos(angle) |
    let r_x_mean_obl = nalgebra::Matrix3::new(
        1.0,
        0.0,
        0.0,
        0.0,
        mean_obliquity.cos(),
        -mean_obliquity.sin(),
        0.0,
        mean_obliquity.sin(),
        mean_obliquity.cos(),
    );
    let r_x_mean_obl_plus_delta = nalgebra::Matrix3::new(
        1.0,
        0.0,
        0.0,
        0.0,
        (mean_obliquity + delta_epsilon).cos(),
        -(mean_obliquity + delta_epsilon).sin(),
        0.0,
        (mean_obliquity + delta_epsilon).sin(),
        (mean_obliquity + delta_epsilon).cos(),
    );
    let r_z_delta_psi = nalgebra::Matrix3::new(
        delta_psi.cos(),
        -delta_psi.sin(),
        0.0,
        delta_psi.sin(),
        delta_psi.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let r_nutation = r_x_mean_obl_plus_delta * r_z_delta_psi * r_x_mean_obl.transpose(); // N = R_x(-e-de) * R_z(-dp) * R_x(e)

    // Precession matrix (ignored as per user's request)
    // P = R_z(-z) * R_y(theta) * R_z(-zeta)
    // let r_z_minus_zeta = nalgebra::Matrix3::new(
    //     (-zeta).cos(), -(-zeta).sin(), 0.0,
    //     (-zeta).sin(), (-zeta).cos(),  0.0,
    //     0.0,           0.0,            1.0,
    // );
    // let r_y_theta = nalgebra::Matrix3::new(
    //     theta.cos(), 0.0, theta.sin(),
    //     0.0,         1.0, 0.0,
    //     -theta.sin(), 0.0, theta.cos(),
    // );
    // let r_z_minus_z = nalgebra::Matrix3::new(
    //     (-z).cos(), -(-z).sin(), 0.0,
    //     (-z).sin(), (-z).cos(),  0.0,
    //     0.0,        0.0,         1.0,
    // );
    // let r_precession = r_z_minus_z * r_y_theta * r_z_minus_zeta;

    // Combined transformation matrix from ECEF to ICRS (J2000)
    // R_ICRS = N * R_ERA * R_ECEF (Precession ignored)
    // This is for position. For velocity, it's more complex.
    // Let's assume the rotation matrix for position is sufficient for now.
    let total_rotation_matrix = r_nutation * r_era;

    let r_obs_eq_km = total_rotation_matrix * r_obs_ecef_km;

    // Rotational velocity (km/s)
    let v_rot = omega_vec.cross(&r_obs_eq_km); // Use transformed position

    // 4. Observer's velocity w.r.t. heliocenter v_obs_helio
    let v_obs_helio = v_orb + v_rot; // km/s

    // 5. Observer's velocity w.r.t. LSR v_obs_lsr
    // v_obs_lsr = v_obs_helio + v_sun_lsr (as per astropy's LSR frame definition)
    let v_obs_lsr = v_obs_helio + v_sun_lsr; // km/s

    // 6. Target direction unit vector k
    let k = Vector3::new(
        obs_dec_rad.cos() * obs_ra_rad.cos(),
        obs_dec_rad.cos() * obs_ra_rad.sin(),
        obs_dec_rad.sin(),
    );

    // 7. Final correction v_correction (dot product of observer's LSR velocity and target direction)
    let v_correction = v_obs_lsr.dot(&k); // km/s

    v_correction
}
