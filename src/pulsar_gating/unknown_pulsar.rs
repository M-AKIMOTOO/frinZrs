use super::known_pulsar::{
    self, build_frequency_axis_mhz, fold_profile, load_sectors_with_limits,
    prepare_output_directory, KnownArgs, SectorData,
};
use anyhow::{anyhow, Result};
use frinZ::fft::{self, process_fft};
use frinZ::header::parse_header;
use ndarray::Axis;
use num_complex::Complex;
use plotters::prelude::*;
use rustfft::FftPlanner;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct UnknownArgs {
    pub input: PathBuf,
    pub bins: usize,
    pub skip: u32,
    pub length: u32,
    pub on_duty: f64,
}

pub fn run(args: UnknownArgs) -> Result<()> {
    if args.bins == 0 {
        return Err(anyhow!("--bins must be greater than 0"));
    }
    if !(0.0..=1.0).contains(&args.on_duty) {
        return Err(anyhow!("--on-duty must be within [0, 1]"));
    }

    let UnknownArgs {
        input,
        bins,
        skip,
        length,
        on_duty,
    } = args;

    let buffer = fs::read(&input)
        .map_err(|e| anyhow!("failed to read input file {}: {e}", input.display()))?;
    let mut cursor = Cursor::new(buffer.as_slice());

    let header = parse_header(&mut cursor)?;
    let sectors = load_sectors_with_limits(&mut cursor, &header, skip, length)?;
    if sectors.len() < 4 {
        return Err(anyhow!(
            "at least 4 sectors are required to estimate pulsar parameters"
        ));
    }

    let freq_axis_mhz = build_frequency_axis_mhz(&header);
    if freq_axis_mhz.is_empty() {
        return Err(anyhow!("frequency axis is empty"));
    }

    let samples_per_sector = freq_axis_mhz.len();
    let (channel_series, durations) = build_time_series(&sectors, samples_per_sector);

    const RATE_PADDING: u32 = 1;
    let fringe = compute_rate_spectrum(
        &sectors,
        samples_per_sector,
        header.fft_point,
        header.sampling_speed,
        RATE_PADDING,
    )?;
    let rate_spectrum_complex = fringe.rate_spectrum.clone();
    let sample_dt = durations.iter().copied().sum::<f64>() / durations.len().max(1) as f64;
    let time_profile = ifft_rate_spectrum(&rate_spectrum_complex)?;
    let period_info = extract_period_from_time_profile(&time_profile, sample_dt)?;
    println!("Estimated period [s] : {:.9}", period_info.period);

    let output_dir = prepare_output_directory(&input)?;
    let stem = file_stem(&input);

    let rate_plot = output_dir.join(format!("{stem}_rate_spectrum.png"));
    plot_rate_spectrum(&rate_plot, &rate_spectrum_complex, sample_dt)?;

    if let Some(rate_hz) = estimate_peak_rate(&rate_spectrum_complex, sample_dt) {
        println!(
            "Estimated fringe rate [Hz]: {:+.6} (period {:.9} s)",
            rate_hz,
            1.0 / rate_hz.abs()
        );
    }

    let time_plot = output_dir.join(format!("{stem}_rate_time_profile.png"));
    plot_time_profile(&time_plot, &time_profile, sample_dt, period_info.highlight)?;

    let dm_estimate = estimate_dispersion_measure(
        &channel_series,
        &durations,
        &freq_axis_mhz,
        period_info.period,
        bins,
    )?;
    if let Some(dm) = dm_estimate {
        println!("Estimated DM [pc cm^-3]: {:.6}", dm);
    } else {
        println!("Estimated DM [pc cm^-3]: n/a");
    }

    let known_args = KnownArgs {
        input,
        period: period_info.period,
        dm: dm_estimate,
        bins,
        skip,
        length,
        on_duty,
    };
    known_pulsar::run(known_args)
}

fn build_time_series(
    sectors: &[SectorData],
    samples_per_sector: usize,
) -> (Vec<Vec<(f64, f64)>>, Vec<f64>) {
    let mut channel_series = vec![Vec::with_capacity(sectors.len()); samples_per_sector];
    let mut durations = Vec::with_capacity(sectors.len());
    let mut cumulative_time = 0.0f64;

    for sector in sectors {
        let duration = sector.integ_time.max(1e-9);
        let center = cumulative_time + duration / 2.0;
        cumulative_time += duration;
        durations.push(duration);

        for chan_idx in 0..samples_per_sector {
            let value = sector
                .spectra
                .get(chan_idx)
                .copied()
                .unwrap_or_else(|| Complex::new(0.0, 0.0));
            let amp = value.norm() as f64;
            channel_series[chan_idx].push((center, amp));
        }
    }

    (channel_series, durations)
}

fn compute_rate_spectrum(
    sectors: &[SectorData],
    samples_per_sector: usize,
    fft_point: i32,
    sampling_speed: i32,
    rate_padding: u32,
) -> Result<FringeSpectra> {
    if sectors.is_empty() || samples_per_sector == 0 {
        return Err(anyhow!(
            "insufficient data to perform fringe search (need > 0 sectors and channels)"
        ));
    }
    let time_len = sectors.len();
    let mut combined = Vec::with_capacity(time_len * samples_per_sector);
    for sector in sectors {
        combined.extend_from_slice(&sector.spectra);
    }

    let padding_limit = fft::compute_padding_limit(sectors.len() as i32);
    let (freq_rate_array, padding_length) = process_fft(
        &combined,
        time_len as i32,
        fft_point,
        sampling_speed,
        &[],
        rate_padding,
        padding_limit,
    );

    let mut rate_accum = vec![Complex::new(0.0f32, 0.0f32); padding_length];
    for row in freq_rate_array.axis_iter(Axis(0)) {
        for (idx, value) in row.iter().enumerate() {
            rate_accum[idx] += *value;
        }
    }

    Ok(FringeSpectra {
        rate_spectrum: rate_accum,
    })
}

fn ifft_rate_spectrum(rate_spectrum: &[Complex<f32>]) -> Result<Vec<f64>> {
    if rate_spectrum.is_empty() {
        return Err(anyhow!("rate spectrum is empty"));
    }
    let len = rate_spectrum.len();
    let mut buffer: Vec<Complex<f32>> = rate_spectrum
        .iter()
        .map(|c| Complex::new(c.norm(), 0.0))
        .collect();
    let mean: f32 = buffer.iter().map(|c| c.re).sum::<f32>() / len as f32;
    for sample in &mut buffer {
        sample.re -= mean;
    }
    if buffer.iter().all(|c| c.re.abs() <= f32::EPSILON) {
        return Err(anyhow!(
            "rate spectrum lacks significant structure for period estimation"
        ));
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft_inv = planner.plan_fft_inverse(len);
    fft_inv.process(&mut buffer);

    let scale = 1.0 / len as f32;
    Ok(buffer
        .into_iter()
        .map(|c| (c * scale).norm() as f64)
        .collect())
}

struct FringeSpectra {
    rate_spectrum: Vec<Complex<f32>>,
}

struct PeriodEstimation {
    period: f64,
    highlight: Option<(f64, f64)>,
}

fn extract_period_from_time_profile(
    time_profile: &[f64],
    sample_dt: f64,
) -> Result<PeriodEstimation> {
    if time_profile.len() < 2 || !sample_dt.is_finite() || sample_dt <= 0.0 {
        return Err(anyhow!(
            "not enough samples to determine pulsar period from fringe search"
        ));
    }

    let mut samples: Vec<(usize, f64)> = Vec::new();
    let len = time_profile.len();
    let nyquist = if len % 2 == 0 { Some(len / 2) } else { None };

    for idx in 1..len {
        if let Some(ny) = nyquist {
            if idx == ny {
                continue;
            }
        }
        let amp = time_profile[idx].abs();
        if amp.is_finite() {
            samples.push((idx, amp));
        }
    }
    if samples.is_empty() {
        return Err(anyhow!(
            "insufficient valid samples for time profile analysis"
        ));
    }

    let mut amplitudes: Vec<f64> = samples.iter().map(|&(_, amp)| amp).collect();
    amplitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = median_of_sorted(&amplitudes);

    let mut deviations: Vec<f64> = amplitudes.iter().map(|a| (a - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = median_of_sorted(&deviations);
    let sigma = mad * 1.4826;

    if sigma <= f64::EPSILON {
        return Err(anyhow!(
            "time profile lacks variance for robust period estimation"
        ));
    }

    let threshold = median + 10.0 * sigma;
    let mut candidate: Option<(f64, f64)> = None;
    let total_time = len as f64 * sample_dt;
    for (idx, amp) in samples {
        if amp >= threshold {
            let raw_time = idx as f64 * sample_dt;
            let folded_time = if raw_time > total_time / 2.0 {
                total_time - raw_time
            } else {
                raw_time
            };
            if folded_time > sample_dt {
                if let Some((best_folded, _)) = candidate {
                    if folded_time < best_folded {
                        candidate = Some((folded_time, amp));
                    }
                } else {
                    candidate = Some((folded_time, amp));
                }
            }
        }
    }
    let Some((earliest, amp)) = candidate else {
        return Err(anyhow!(
            "no significant peaks above 10σ found in time-domain fringe profile"
        ));
    };
    Ok(PeriodEstimation {
        period: earliest,
        highlight: Some((earliest, amp)),
    })
}

fn estimate_dispersion_measure(
    channel_series: &[Vec<(f64, f64)>],
    durations: &[f64],
    freq_axis_mhz: &[f64],
    period: f64,
    bins: usize,
) -> Result<Option<f64>> {
    if channel_series.is_empty() || freq_axis_mhz.is_empty() || period <= 0.0 {
        return Ok(None);
    }

    let mut channel_phases: Vec<Option<f64>> = Vec::with_capacity(channel_series.len());
    for series in channel_series {
        if series.is_empty() {
            channel_phases.push(None);
            continue;
        }
        let folded = fold_profile(series, durations, period, bins)?;
        let mut best_phase = None;
        let mut best_amp = f64::NEG_INFINITY;
        for &(phase, amp) in &folded {
            if amp.is_finite() && amp > best_amp {
                best_amp = amp;
                best_phase = Some(phase);
            }
        }
        channel_phases.push(best_phase);
    }

    let mut ref_idx = freq_axis_mhz.len().saturating_sub(1);
    while ref_idx > 0 && channel_phases.get(ref_idx).and_then(|p| *p).is_none() {
        ref_idx -= 1;
    }
    let Some(ref_phase) = channel_phases.get(ref_idx).and_then(|p| *p) else {
        return Ok(None);
    };
    let ref_freq = freq_axis_mhz
        .get(ref_idx)
        .copied()
        .filter(|f| *f > 0.0 && f.is_finite())
        .ok_or_else(|| anyhow!("reference channel frequency is invalid"))?;

    const DISPERSION_CONSTANT_SEC: f64 = 4.148_808;
    let mut sum_xx = 0.0f64;
    let mut sum_xy = 0.0f64;

    for (idx, phase_opt) in channel_phases.iter().enumerate() {
        if idx == ref_idx {
            continue;
        }
        let Some(phase) = phase_opt else {
            continue;
        };
        let freq = *freq_axis_mhz
            .get(idx)
            .ok_or_else(|| anyhow!("frequency axis and time series length mismatch"))?;
        if !freq.is_finite() || freq <= 0.0 {
            continue;
        }

        let mut phase_diff = phase - ref_phase;
        if phase_diff > 0.5 {
            phase_diff -= 1.0;
        } else if phase_diff < -0.5 {
            phase_diff += 1.0;
        }
        let delay = phase_diff * period;
        if !delay.is_finite() {
            continue;
        }

        let dispersion_term = (1.0 / (freq * freq)) - (1.0 / (ref_freq * ref_freq));
        if dispersion_term.abs() < 1e-12 {
            continue;
        }
        sum_xx += dispersion_term * dispersion_term;
        sum_xy += dispersion_term * delay;
    }

    if sum_xx <= 0.0 {
        return Ok(None);
    }
    let dm = sum_xy / (DISPERSION_CONSTANT_SEC * sum_xx);
    if dm.is_finite() {
        Ok(Some(dm))
    } else {
        Ok(None)
    }
}

fn file_stem(input: &Path) -> String {
    input
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "pulsar".to_string())
}

fn generate_rate_axis(len: usize, sample_dt: f64) -> Vec<f64> {
    let mut axis = Vec::with_capacity(len);
    if len == 0 || !sample_dt.is_finite() || sample_dt <= 0.0 {
        return axis;
    }
    let sample_rate = 1.0 / sample_dt;
    let bin_width = sample_rate / len as f64;
    let half = (len / 2) as isize;
    for idx in 0..len {
        let offset = idx as isize - half;
        axis.push(offset as f64 * bin_width);
    }
    axis
}

fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

fn estimate_peak_rate(rate_spectrum: &[Complex<f32>], sample_dt: f64) -> Option<f64> {
    if rate_spectrum.len() < 2 || !sample_dt.is_finite() || sample_dt <= 0.0 {
        return None;
    }
    let axis = generate_rate_axis(rate_spectrum.len(), sample_dt);
    let mut best_amp = 0.0f64;
    let mut best_freq = None;

    for (freq, value) in axis.into_iter().zip(rate_spectrum.iter()) {
        if freq.abs() < f64::EPSILON {
            continue;
        }
        let amp = value.norm() as f64;
        if amp > best_amp {
            best_amp = amp;
            best_freq = Some(freq);
        }
    }

    best_freq
}

fn plot_rate_spectrum(path: &Path, spectrum: &[Complex<f32>], sample_dt: f64) -> Result<()> {
    if spectrum.len() < 2 || !sample_dt.is_finite() || sample_dt <= 0.0 {
        return Ok(());
    }
    let len = spectrum.len();
    let rate_axis = generate_rate_axis(len, sample_dt);
    let amplitudes: Vec<f64> = spectrum.iter().map(|c| c.norm() as f64).collect();
    plot_line_chart(
        path,
        &rate_axis,
        &amplitudes,
        "Fringe rate spectrum",
        "Rate [Hz]",
        "Amplitude (a.u.)",
    )
}

fn plot_time_profile(
    path: &Path,
    profile: &[f64],
    sample_dt: f64,
    highlight: Option<(f64, f64)>,
) -> Result<()> {
    if profile.len() < 2 || !sample_dt.is_finite() || sample_dt <= 0.0 {
        return Ok(());
    }
    let len = profile.len();
    let time_axis: Vec<f64> = (0..len).map(|idx| idx as f64 * sample_dt).collect();
    let nyquist = if len % 2 == 0 { Some(len / 2) } else { None };
    let mut line_values = Vec::with_capacity(len);
    for (idx, &amp) in profile.iter().enumerate() {
        if idx == 0 || nyquist.is_some_and(|ny| idx == ny) {
            line_values.push(0.0);
        } else {
            line_values.push(amp);
        }
    }

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &amp in &line_values {
        if amp.is_finite() {
            y_min = y_min.min(amp);
            y_max = y_max.max(amp);
        }
    }
    if let Some((_, amp)) = highlight {
        if amp.is_finite() {
            y_min = y_min.min(amp);
            y_max = y_max.max(amp);
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        return Ok(());
    }
    let span = (y_max - y_min).abs().max(1e-9);

    let root = BitMapBackend::new(path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Time-domain fringe profile", ("sans-serif", 28))
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(
            *time_axis.first().unwrap_or(&0.0)..*time_axis.last().unwrap_or(&0.0),
            (y_min - 0.05 * span)..(y_max + 0.05 * span),
        )?;

    chart
        .configure_mesh()
        .x_desc("Time [s]")
        .y_desc("Amplitude (a.u.)")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            time_axis.iter().copied().zip(line_values.iter().copied()),
            &BLUE,
        ))?
        .label("Time series (DC & Nyquist removed)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], &BLUE));

    if let Some((t, amp)) = highlight {
        chart
            .draw_series(std::iter::once(Circle::new((t, amp), 5, RED.filled())))?
            .label("Estimated period (≥10σ)")
            .legend(|(x, y)| Circle::new((x + 10, y), 5, RED.filled()));
    }

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}

fn plot_line_chart(
    path: &Path,
    x_values: &[f64],
    y_values: &[f64],
    title: &str,
    x_label: &str,
    y_label: &str,
) -> Result<()> {
    if x_values.len() != y_values.len() || x_values.is_empty() {
        return Ok(());
    }

    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for (&x, &y) in x_values.iter().zip(y_values.iter()) {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }

    if !x_min.is_finite()
        || !x_max.is_finite()
        || !y_min.is_finite()
        || !y_max.is_finite()
        || (x_max - x_min).abs() < 1e-12
        || (y_max - y_min).abs() < 1e-12
    {
        return Ok(());
    }

    let x_range = (x_max - x_min).abs().max(1e-12);
    let y_range = (y_max - y_min).abs().max(1e-12);

    let root = BitMapBackend::new(path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(title, ("sans-serif", 28))
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(
            (x_min - 0.05 * x_range)..(x_max + 0.05 * x_range),
            (y_min - 0.05 * y_range)..(y_max + 0.05 * y_range),
        )?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;
    chart.draw_series(LineSeries::new(
        x_values.iter().copied().zip(y_values.iter().copied()),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}
