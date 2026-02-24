#![allow(non_local_definitions, unexpected_cfgs)]

use byteorder::{LittleEndian, WriteBytesExt};
use chrono::{DateTime, Utc};
use npyz::WriterBuilder;
use num_complex::Complex;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::analysis::AnalysisResults;
use crate::header::CorHeader;

type C32 = Complex<f32>;

#[allow(non_local_definitions, unexpected_cfgs)]
#[derive(npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize, Clone, Copy)]
pub struct ComplexRiRow {
    pub real: f32,
    pub imag: f32,
}

#[allow(non_local_definitions, unexpected_cfgs)]
#[derive(npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize, Clone, Copy)]
struct AddPlotRow {
    elapsed_time_s: f32,
    amplitude_pct: f32,
    snr: f32,
    phase_deg: f32,
    noise_level_pct: f32,
    res_delay_samp: f32,
    res_rate_hz: f32,
}

pub fn npy<T: npyz::AutoSerialize>(output_path: &Path, rows: &[T]) -> io::Result<()> {
    let file = File::create(output_path)?;
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .writer(file)
        .begin_1d()?;
    for row in rows {
        writer.push(row)?;
    }
    writer.finish()?;
    Ok(())
}

pub fn npy_f32_2d(output_path: &Path, rows: usize, cols: usize, values: &[f32]) -> io::Result<()> {
    if rows.checked_mul(cols).unwrap_or(0) != values.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "rows*cols must match values.len()",
        ));
    }
    let file = File::create(output_path)?;
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[rows as u64, cols as u64])
        .writer(file)
        .begin_nd()?;
    for value in values {
        writer.push(value)?;
    }
    writer.finish()?;
    Ok(())
}

pub fn write_complex_spectrum_npy(path: &Path, spectrum: &[C32]) -> io::Result<PathBuf> {
    let npy_path = if path.extension().and_then(|v| v.to_str()) == Some("npy") {
        path.to_path_buf()
    } else {
        path.with_extension("npy")
    };
    let rows: Vec<ComplexRiRow> = spectrum
        .iter()
        .map(|c| ComplexRiRow {
            real: c.re,
            imag: c.im,
        })
        .collect();
    npy(&npy_path, &rows)?;
    Ok(npy_path)
}

pub fn output_header_info(
    header: &CorHeader,
    output_dir: &Path,
    basename: &str,
) -> io::Result<String> {
    let header_file_path = output_dir.join(format!("{}_header.txt", basename));
    let header_info = format!(
        "### header region information

        [Header]

        Magic Word           = {:?}
        Header Version       = {}
        Software Version     = {}
        Sampling Frequency   = {} MHz
        Observing Frequency  = {} MHz
        FFT Point            = {}
        Number of Sector     = {}
        Bandwidth            = {} MHz
        Resolution Bandwidth = {} MHz

        [Station1]
            Name     = {}
            Code     = {}
            Clock Delay = {} s
            Clock Rate  = {} s/s
            Clock Acel  = {} s/s**2
            Clock Jerk  = {} s/s**3
            Clock Snap  = {} s/s**4
            Position = ({}, {}, {}) [m], geocentric coordinate

        [Station2]
            Name     = {}
            Code     = {}
            Clock Delay = {} s
            Clock Rate  = {} s/s
            Clock Acel  = {} s/s**2
            Clock Jerk  = {} s/s**3
            Clock Snap  = {} s/s**4
            Position = ({}, {}, {}) [m], geocentric coordinate

        [Source]
            Name       = {}
            Coordinate = ({}, {}) J2000
",
        header.magic_word,
        header.header_version,
        header.software_version,
        header.sampling_speed as f32 / 1e6,
        header.observing_frequency as f32 / 1e6,
        header.fft_point,
        header.number_of_sector,
        header.sampling_speed as f32 / 2.0 / 1e6,
        (header.sampling_speed as f32 / 2.0 / 1e6) / header.fft_point as f32 * 2.0,
        header.station1_name,
        header.station1_code,
        header.station1_clock_delay,
        header.station1_clock_rate,
        header.station1_clock_acel,
        header.station1_clock_jerk,
        header.station1_clock_snap,
        header.station1_position[0] as f64,
        header.station1_position[1] as f64,
        header.station1_position[2] as f64,
        header.station2_name,
        header.station2_code,
        header.station2_clock_delay,
        header.station2_clock_rate,
        header.station2_clock_acel,
        header.station2_clock_jerk,
        header.station2_clock_snap,
        header.station2_position[0] as f64,
        header.station2_position[1] as f64,
        header.station2_position[2] as f64,
        header.source_name,
        header.source_position_ra.to_degrees() as f64,
        header.source_position_dec.to_degrees() as f64
    );
    if !header_file_path.exists() {
        std::fs::write(header_file_path, &header_info)?;
    }
    Ok(header_info)
}

pub fn generate_output_names(
    header: &CorHeader,
    obs_time: &DateTime<Utc>,
    label: &[&str],
    is_rfi_filtered: bool,
    is_frequency_mode: bool,
    is_bandpass_corrected: bool,
    length: i32,
) -> String {
    let yyyydddhhmmss2 = obs_time.format("%Y%j%H%M%S").to_string();
    let rfi_suffix = if is_rfi_filtered { "_rfi" } else { "" };
    let bp_suffix = if is_bandpass_corrected { "_bp" } else { "" };
    let _mode_suffix = if is_frequency_mode { "_freq" } else { "_time" };
    let observing_band = if (6600.0..=7112.0).contains(&(header.observing_frequency as f32 / 1e6)) {
        "c"
    } else if (8192.0..=8704.0).contains(&(header.observing_frequency as f32 / 1e6)) {
        "x"
    } else if (11923.0..=12435.0).contains(&(header.observing_frequency as f32 / 1e6)) {
        "ku"
    } else {
        "n"
    };
    let label_segment = label.get(3).copied().unwrap_or("");

    let base = format!(
        "{}_{}_{}_{}_{}_len{}s{}{}",
        header.station1_name,
        header.station2_name,
        yyyydddhhmmss2,
        label_segment,
        observing_band,
        length,
        rfi_suffix,
        bp_suffix
    );
    base
}

pub fn format_delay_output(
    results: &AnalysisResults,
    label: &[&str],
    args_length: i32,
    rfi_display: &str,
    bandpass_applied: bool,
) -> String {
    let display_length = if args_length > 0 {
        args_length as f32 * results.effective_integ_time_s
    } else {
        results.length_f32
    };
    let epoch_display = format_epoch_for_length(results, display_length);
    let length_display = format_length_for_table(display_length);
    let label_segment = label.get(3).copied().unwrap_or("");
    format!(
        " {} {:<10} {:<8} {:>8}  {:>8.6} {:>7.1} {:>+10.3} {:>10.6} {:>+11.6} {:>+11.8} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>12.5}   {:<15} {:<5}",
        epoch_display,
        label_segment,
        results.source_name,
        length_display,
        results.delay_max_amp * 100.0,
        results.delay_snr,
        results.delay_phase,
        results.delay_noise * 100.0,
        results.residual_delay,
        results.residual_rate,
        results.ant1_az,
        results.ant1_el,
        results.ant1_hgt,
        results.ant2_az,
        results.ant2_el,
        results.ant2_hgt,
        results.mjd,
        rfi_display,
        if bandpass_applied { "True" } else { "False" },
        //results.l_coord,
        //results.m_coord
    )
}

pub fn format_freq_output(
    results: &AnalysisResults,
    label: &[&str],
    args_length: i32,
    rfi_display: &str,
    bandpass_applied: bool,
) -> String {
    let display_length = if args_length > 0 {
        args_length as f32 * results.effective_integ_time_s
    } else {
        results.length_f32
    };
    let epoch_display = format_epoch_for_length(results, display_length);
    let length_display = format_length_for_table(display_length);
    let label_segment = label.get(3).copied().unwrap_or("");
    format!(
        " {} {:<10} {:<8} {:>8}  {:>8.6} {:>7.1} {:>+10.3} {:>+12.7} {:>10.6} {:>+11.6} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>12.5}   {:<15} {:<5}",
        epoch_display,
        label_segment,
        results.source_name,
        length_display,
        results.freq_max_amp * 100.0,
        results.freq_snr,
        results.freq_phase,
        results.freq_freq,
        results.freq_noise * 100.0,
        results.residual_rate,
        results.ant1_az,
        results.ant1_el,
        results.ant1_hgt,
        results.ant2_az,
        results.ant2_el,
        results.ant2_hgt,
        results.mjd,
        rfi_display,
        if bandpass_applied { "True" } else { "False" },
        //results.l_coord,
        //results.m_coord
    )
}

fn format_epoch_for_length(results: &AnalysisResults, length_s: f32) -> &str {
    if length_s < 1e-3 {
        &results.yyyydddhhmmss1_us
    } else if length_s < 1.0 {
        &results.yyyydddhhmmss1_ms
    } else {
        &results.yyyydddhhmmss1
    }
}

fn format_length_for_table(length_s: f32) -> String {
    if !length_s.is_finite() {
        return "nan".to_string();
    }
    if length_s >= 1.0 {
        format!("{}", length_s.round() as i64)
    } else {
        format!("{:.1e}", length_s)
    }
}

pub fn write_phase_corrected_spectrum_binary(
    file_path: &Path,
    file_header: &[u8],
    sector_headers: &[Vec<u8>],
    calibrated_spectra: &[Vec<C32>],
) -> io::Result<()> {
    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);

    // 1. ファイルヘッダー (256 byte) を書き込む
    writer.write_all(file_header)?;

    // 2. 各セクターのヘッダーと較正済みデータを書き込む
    for (i, spectrum) in calibrated_spectra.iter().enumerate() {
        // このセクターの生の128バイトヘッダーを書き込む
        writer.write_all(&sector_headers[i])?;

        // 較正済みの複素スペクトルの実部と虚部 (各4 byte) を交互に書き込む
        for c in spectrum {
            writer.write_f32::<LittleEndian>(c.re)?;
            writer.write_f32::<LittleEndian>(c.im)?;
        }
    }
    Ok(())
}

pub fn write_add_plot_data_to_file(
    output_dir: &Path,
    base_filename: &str,
    elapsed_times: &[f32],
    amp: &[f32],
    snr: &[f32],
    phase: &[f32],
    noise: &[f32],
    res_delay: &[f32],
    res_rate: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let output_file_path = output_dir.join(format!("{}_addplot.npy", base_filename));
    let rows: Vec<AddPlotRow> = elapsed_times
        .iter()
        .zip(amp.iter())
        .zip(snr.iter())
        .zip(phase.iter())
        .zip(noise.iter())
        .zip(res_delay.iter())
        .zip(res_rate.iter())
        .map(
            |(
                (
                    ((((&elapsed_time_s, &amplitude_pct), &snr), &phase_deg), &noise_level_pct),
                    &res_delay_samp,
                ),
                &res_rate_hz,
            )| AddPlotRow {
                elapsed_time_s,
                amplitude_pct,
                snr,
                phase_deg,
                noise_level_pct,
                res_delay_samp,
                res_rate_hz,
            },
        )
        .collect();
    npy(&output_file_path, &rows)?;
    Ok(())
}
