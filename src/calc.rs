use std::path::{Path, PathBuf};

use astro::coords;
use astro::time;
use blh::{ellipsoid, GeocentricCoord, GeodeticCoord};
use chrono::{DateTime, Datelike, Timelike, Utc};
use ndarray::prelude::*;
use num_complex::Complex;

use crate::args::Args;
use crate::read::Header;

// --- Type Aliases for Clarity ---
type C32 = Complex<f32>;

// --- Analysis Results Structure ---
pub struct AnalysisResults {
    pub yyyydddhhmmss1: String,
    pub source_name: String,
    pub length_f32: f32,
    pub ant1_az: f32,
    pub ant1_el: f32,
    pub ant1_hgt: f32,
    pub ant2_az: f32,
    pub ant2_el: f32,
    pub ant2_hgt: f32,
    pub delay_max_amp: f32,
    pub delay_phase: f32,
    pub delay_snr: f32,
    pub delay_noise: f32,
    pub freq_max_amp: f32,
    pub freq_phase: f32,
    pub freq_freq: f32,
    pub freq_snr: f32,
    pub freq_noise: f32,
}

// --- Analysis and Parameter Calculation ---
pub fn analyze_results(
    freq_rate_array: &Array2<C32>,
    freq_rate_data: &Array2<f32>,
    delay_rate_array: &Array2<C32>,
    delay_rate_2d_data_comp: &Array2<C32>,
    header: &Header,
    length: i32,
    effective_integ_time: f32,
    obs_time: &DateTime<Utc>,
    padding_length: usize,
    args: &Args,
) -> AnalysisResults {
    let fft_point_usize = header.fft_point as usize;
    let fft_point_half = fft_point_usize / 2;
    let fft_point_f32 = header.fft_point as f32;
    let length_f32 = length as f32;
    let padding_length_half = padding_length / 2;

    // --- Ranges ---
    let delay_range = Array::linspace(-(fft_point_f32 / 2.0), fft_point_f32 / 2.0 - 1.0, padding_length);
    let freq_range = Array::linspace(0.0f32, (header.sampling_speed as f32 / 2.0) / 1e6, fft_point_half);
    let rate_range = rate_cal(length as f32, effective_integ_time);

    // --- Delay Analysis ---
    let delay_rate_2d_data_array = delay_rate_array.mapv(|x| x.norm());
    let delay_noise = noise_level(delay_rate_2d_data_comp.view(), delay_rate_2d_data_comp.mean().unwrap(), padding_length, fft_point_usize);

    let delay_max_amp: f32;
    let delay_phase: f32;

    if !args.delay_window.is_empty() && !args.rate_window.is_empty() {
        let delay_win_low = args.delay_window[0];
        let delay_win_high = args.delay_window[1];
        let rate_win_low = args.rate_window[0];
        let rate_win_high = args.rate_window[1];
        let mut temp_max_rate_idx_abs = 0;
        let mut temp_max_delay_idx_abs = 0;

        let mut max_val_in_window = 0.0f32;

        for r_idx in 0..rate_range.len() {
            if rate_range[r_idx] >= rate_win_low && rate_range[r_idx] <= rate_win_high {
                for d_idx in 0..delay_range.len() {
                    if delay_range[d_idx] >= delay_win_low && delay_range[d_idx] <= delay_win_high {
                        let current_val = delay_rate_2d_data_array[[r_idx, d_idx]];
                        if current_val > max_val_in_window {
                            max_val_in_window = current_val;
                            temp_max_rate_idx_abs = r_idx;
                            temp_max_delay_idx_abs = d_idx;
                        }
                    }
                }
            }
        }

        if max_val_in_window > 0.0 {
            delay_max_amp = max_val_in_window;
            delay_phase = delay_rate_2d_data_comp[[temp_max_rate_idx_abs, temp_max_delay_idx_abs]].arg().to_degrees();
        } else {
            delay_max_amp = delay_rate_2d_data_array[[padding_length_half, fft_point_half]];
            delay_phase = delay_rate_2d_data_comp[[padding_length_half, fft_point_half]].arg().to_degrees();
        }
    } else {
        delay_max_amp = delay_rate_2d_data_array[[padding_length_half, fft_point_half]];
        delay_phase = delay_rate_2d_data_comp[[padding_length_half, fft_point_half]].arg().to_degrees();
    }

    let delay_snr = delay_max_amp / delay_noise;

    // --- Frequency Analysis ---
    let cross_power = freq_rate_data.column(padding_length_half).to_owned();
    let (freq_max_ind, freq_max_amp) = cross_power.iter().enumerate().fold((0, 0.0), |(i_max, v_max), (i, &v)| {
        if v > v_max { (i, v) } else { (i_max, v_max) }
    });
    let freq_freq = freq_range[freq_max_ind];
    let freq_phase = freq_rate_array[[freq_max_ind, padding_length_half]].arg().to_degrees();
    let freq_noise = noise_level(freq_rate_array.view(), freq_rate_array.mean().unwrap(), fft_point_half, padding_length);
    let freq_snr = freq_max_amp / freq_noise;

    // --- Antenna Az/El Calculation ---
    let (ant1_az, ant1_el, ant1_hgt) = calculate_azel(
        header.yamagu32_pos,
        obs_time,
        header.source_position_ra,
        header.source_position_dec,
    );
    let (ant2_az, ant2_el, ant2_hgt) = calculate_azel(
        header.yamagu34_pos,
        obs_time,
        header.source_position_ra,
        header.source_position_dec,
    );

    AnalysisResults {
        yyyydddhhmmss1: obs_time.format("%Y/%j %H:%M:%S").to_string(),
        source_name: header.source_name.clone(),
        length_f32,
        ant1_az: ant1_az as f32,
        ant1_el: ant1_el as f32,
        ant1_hgt: ant1_hgt as f32,
        ant2_az: ant2_az as f32,
        ant2_el: ant2_el as f32,
        ant2_hgt: ant2_hgt as f32,
        delay_max_amp,
        delay_phase,
        delay_snr,
        delay_noise,
        freq_max_amp,
        freq_phase,
        freq_freq,
        freq_snr,
        freq_noise,
    }
}

fn rate_cal(n: f32, d: f32) -> Vec<f32> {
    let mut rate: Vec<f32> = Vec::with_capacity(n as usize);
    if n % 2.0 == 0.0 {
        for i in (-n as i32 / 2)..(n as i32 / 2) {
            rate.push(i as f32 / (n * d));
        }
    } else {
        for i in (-(n as i32 - 1) / 2)..((n as i32 - 1) / 2 + 1) {
            rate.push(i as f32 / (n * d));
        }
    }
    rate
}

fn noise_level(array: ArrayView2<C32>, array_mean: C32, rows: usize, cols: usize) -> f32 {
    let mut array_sum: f32 = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            array_sum += (array[[i, j]] - array_mean).norm()
        }
    }
    array_sum / (rows * cols) as f32
}

fn calculate_azel(
    pos: (f64, f64, f64),
    time: &DateTime<Utc>,
    obs_ra: f64,
    obs_dec: f64,
) -> (f64, f64, f64) {
    let ant_position = [pos.0, pos.1, pos.2];

    let obs_year = time.year() as i16;
    let obs_month = time.month() as u8;
    let obs_day = time.day() as u8;
    let obs_hour = time.hour() as u8;
    let obs_minute = (time.minute() as f32 / 60.0) as u8;
    let obs_second = (time.second() as f64) + (time.nanosecond() as f64 / 1_000_000_000.0);

    let day_of_month = time::DayOfMonth {
        day: obs_day,
        hr: obs_hour,
        min: obs_minute,
        sec: obs_second,
        time_zone: 0.0,
    };
    let date = time::Date {
        year: obs_year,
        month: obs_month,
        decimal_day: time::decimal_day(&day_of_month),
        cal_type: time::CalType::Gregorian,
    };

    let geocentric_coord = GeocentricCoord::new(ant_position[0], ant_position[1], ant_position[2]);
    let geodetic_coord: GeodeticCoord<ellipsoid::GRS80> = geocentric_coord.into();
    let longitude_radian = geodetic_coord.lon.0;
    let latitude_radian = geodetic_coord.lat.0;
    let height_meter = geodetic_coord.hgt;

    let julian_day = time::julian_day(&date);
    let mean_sidereal = time::mn_sidr(julian_day);
    let hour_angle = coords::hr_angl_frm_observer_long(mean_sidereal, -longitude_radian, obs_ra);

    let source_az = coords::az_frm_eq(hour_angle, obs_dec, latitude_radian).to_degrees() + 180.0;
    let source_el = coords::alt_frm_eq(hour_angle, obs_dec, latitude_radian).to_degrees();

    (source_az, source_el, height_meter)
}

pub fn generate_output_names(
    output_dir: &Path,
    header: &Header,
    obs_time: &DateTime<Utc>,
    label: &[&str],
    rfi_filtered: bool,
) -> (PathBuf, PathBuf) {
    let time_str = obs_time.format("%Y%j%H%M%S").to_string();
    let rfi_suffix = if rfi_filtered { "_rfi-filtered" } else { "" };

    let fringe_outputname = output_dir.join(format!(
        "{}_{}_{}_x_len{}s_delay_rate_search{}.png",
        label[0],
        label[1],
        time_str,
        header.number_of_sector,
        rfi_suffix
    ));

    let spectrum_outputname = output_dir.join(format!(
        "{}_{}_{}_x_len{}s_freq_rate_search{}.png",
        label[0],
        label[1],
        time_str,
        header.number_of_sector,
        rfi_suffix
    ));

    (fringe_outputname, spectrum_outputname)
}

pub fn format_delay_output(results: &AnalysisResults, label: &[&str]) -> String {
    let label_str = if label.len() > 3 { label[3] } else { "" };
    format!(
        " {}   {:<5}  {:<10} {:<8.2} {:<8.6}    {:.1} {:>+10.3}   {:>10.6}   {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}\n",
        results.yyyydddhhmmss1,
        label_str,
        results.source_name,
        results.length_f32,
        results.delay_max_amp * 100.0,
        results.delay_snr,
        results.delay_phase,
        results.delay_noise * 100.0,
        results.ant1_az,
        results.ant1_el,
        results.ant1_hgt,
        results.ant2_az,
        results.ant2_el,
        results.ant2_hgt
    )
}

pub fn format_freq_output(results: &AnalysisResults, label: &[&str]) -> String {
    let label_str = if label.len() > 3 { label[3] } else { "" };
    format!(
        " {}   {:<5}  {:<10} {:<8.2} {:<8.6}  {:<.1}   {:>+10.3} {:>+10.3} {:>10.6}  {:>8.3} {:>8.3} {:>8.3}  {:>8.3} {:>8.3} {:>8.3}\n",
        results.yyyydddhhmmss1,
        label_str,
        results.source_name,
        results.length_f32,
        results.freq_max_amp * 100.0,
        results.freq_snr,
        results.freq_phase,
        results.freq_freq,
        results.freq_noise * 100.0,
        results.ant1_az,
        results.ant1_el,
        results.ant1_hgt,
        results.ant2_az,
        results.ant2_el,
        results.ant2_hgt
    )
}
