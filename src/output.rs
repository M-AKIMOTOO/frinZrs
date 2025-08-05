use std::io;
use std::path::{Path};
use chrono::{DateTime, Utc};

use crate::header::CorHeader;
use crate::analysis::AnalysisResults;

pub fn output_header_info(header: &CorHeader, output_dir: &Path, basename: &str) -> io::Result<String> {
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
        header.sampling_speed as f32 / 2.0,
        (header.sampling_speed as f32 / 2.0) / header.fft_point as f32 * 2.0,
        
        header.station1_name,
        header.station1_code,
        header.station1_clock_delay,
        header.station1_clock_rate,
        header.station1_clock_acel,
        header.station1_clock_jerk,
        header.station1_clock_snap,
        header.station1_position[0] as f32,
        header.station1_position[1] as f32,
        header.station1_position[2] as f32,
        header.station2_name,
        header.station2_code,
        header.station2_clock_delay,
        header.station2_clock_rate,
        header.station2_clock_acel,
        header.station2_clock_jerk,
        header.station2_clock_snap,
        header.station2_position[0] as f32,
        header.station2_position[1] as f32,
        header.station2_position[2] as f32,
        header.source_name,
        header.source_position_ra.to_degrees() as f32,
        header.source_position_dec.to_degrees() as f32
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
    length: i32,
) -> (String, String) {
    let yyyydddhhmmss2 = obs_time.format("%Y%j%H%M%S").to_string();
    let rfi_suffix = if is_rfi_filtered { "_rfi" } else { "" };
    let mode_suffix = if is_frequency_mode { "_freq" } else { "_time" };
    let observing_band = if (6600.0..=7112.0).contains(&(header.observing_frequency as f32 / 1e6)) {
        "c"
    } else if (8192.0..=8704.0).contains(&(header.observing_frequency as f32 / 1e6)) {
        "x"
    } else {
        "N"
    };

    let base = format!(
        "{}_{}_{}_{}_{}_len{}s{}",
        header.station1_name, header.station2_name, yyyydddhhmmss2, label[3], observing_band, length, rfi_suffix
    );
    (base, mode_suffix.to_string())
}

pub fn format_delay_output(results: &AnalysisResults, label: &[&str]) -> String {
    format!(
        " {}   {:<5}  {:<10} {:<8.2} {:<3.6} {:>7.1} {:>+10.3}  {:>10.6}  {:>+9.8}   {:>+4.8}   {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>12.5}\n",
        results.yyyydddhhmmss1,
        label[3],
        results.source_name,
        results.length_f32,
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
        results.mjd
    )
}


pub fn format_freq_output(results: &AnalysisResults, label: &[&str]) -> String {
    format!(
        " {}   {:<5}  {:<10} {:<8.2} {:<8.6}  {:>7.1}   {:>+10.3} {:>+10.3} {:>10.6} {:>+10.6} {:>7.3} {:>7.3} {:>7.3}  {:>7.3} {:>7.3} {:>7.3} {:>12.5}\n",
        results.yyyydddhhmmss1,
        label[3],
        results.source_name,
        results.length_f32,
        results.freq_max_amp * 100.0,
        results.freq_snr,
        results.freq_phase,
        results.freq_freq,
        results.freq_noise * 100.0,
        results.residual_rate * 1000.0,
        results.ant1_az,
        results.ant1_el,
        results.ant1_hgt,
        results.ant2_az,
        results.ant2_el,
        results.ant2_hgt,
        results.mjd
    )
}