use ndarray::prelude::*;
use num_complex::Complex;
use chrono::{DateTime, Utc, Datelike, Timelike};

use astro::coords;
use astro::time;
use blh::{ellipsoid, GeocentricCoord, GeodeticCoord};

type C32 = Complex<f32>;

pub fn rate_cal(n: f32, d: f32) -> Vec<f32> {
    let mut rate: Vec<f32> = Vec::with_capacity(n as usize);
    let step = 1.0 / (n * d); // 各ステップの大きさを計算
    if n % 2.0 == 0.0 {
        for i in (-n as i32 / 2)..(n as i32 / 2) {
            rate.push(i as f32 * step);
        }
    } else {
        for i in (-(n as i32 - 1) / 2)..((n as i32 - 1) / 2 + 1) {
            rate.push(i as f32 * step);
        }
    }
    rate
}

pub fn noise_level(array: ArrayView2<C32>, array_mean: C32, rows: usize, cols: usize) -> f32 {
    let mut array_sum: f32 = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            array_sum += (array[[i, j]] - array_mean).norm()
        }
    }
    array_sum / (rows * cols) as f32
}

pub fn radec2azalt(ant_position: [f32; 3], time: DateTime<Utc>, obs_ra: f32, obs_dec: f32) -> (f32, f32, f32) {
    let obs_year = time.year() as i16;
    let obs_month = time.month() as u8;
    let obs_day = time.day() as u8;
    let obs_hour = time.hour() as u8;
    let obs_minute = time.minute() as u8;
    let obs_second = time.second() as f64; // + (time.nanosecond() as f64 / 1_000_000_000.0);

    let decimal_day_calc = obs_day as f64 + obs_hour as f64 / 24.0 + obs_minute as f64 / 60.0 / 24.0 + obs_second as f64 / 24.0 / 60.0 / 60.0;

    let date = time::Date {
        year: obs_year,
        month: obs_month,
        decimal_day: decimal_day_calc,
        cal_type: time::CalType::Gregorian,
    };

    let geocentric_coord = GeocentricCoord::new(ant_position[0] as f64, ant_position[1] as f64, ant_position[2] as f64);
    let geodetic_coord: GeodeticCoord<ellipsoid::WGS84> = geocentric_coord.into();
    let longitude_radian = geodetic_coord.lon.0;
    let latitude_radian = geodetic_coord.lat.0;
    let height_meter = geodetic_coord.hgt;

    let julian_day = time::julian_day(&date);
    let mean_sidereal = time::mn_sidr(julian_day);
    let hour_angle = coords::hr_angl_frm_observer_long(mean_sidereal, -longitude_radian, obs_ra as f64);

    let source_az = coords::az_frm_eq(hour_angle, obs_dec as f64, latitude_radian).to_degrees() as f32 +180.0; 
    let source_el = coords::alt_frm_eq(hour_angle, obs_dec as f64, latitude_radian).to_degrees() as f32;

    (source_az, source_el, height_meter as f32)
}

pub fn mjd_cal(time: DateTime<Utc>) -> f64 {
    let obs_year = time.year() as i16;
    let obs_month = time.month() as u8;
    let obs_day = time.day() as u8;
    let obs_hour = time.hour() as u8;
    let obs_minute = time.minute() as u8;
    let obs_second = time.second() as f64; // + (time.nanosecond() as f64 / 1_000_000_000.0);

    let decimal_day_calc = obs_day as f64 + obs_hour as f64 / 24.0 + obs_minute as f64 / 60.0 / 24.0 + obs_second as f64 / 24.0 / 60.0 / 60.0;

    let date = time::Date {
        year: obs_year,
        month: obs_month,
        decimal_day: decimal_day_calc,
        cal_type: time::CalType::Gregorian,
    };

    let julian_day = time::julian_day(&date);
    julian_day - 2400000.5
}

pub fn unwrap_phase(phases: &mut [f32]) {
    if phases.len() < 2 {
        return;
    }
    let mut offset = 0.0;
    let mut original_prev = phases[0];

    for i in 1..phases.len() {
        let original_current = phases[i];
        let diff = original_current - original_prev;
        if diff > 180.0 {
            offset -= 360.0;
        } else if diff < -180.0 {
            offset += 360.0;
        }
        phases[i] += offset;
        original_prev = original_current;
    }
}