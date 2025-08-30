use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::io::Read;
use std::path::Path;
use std::io::Write;
use std::fs::File;

use chrono::{DateTime, Utc};
use ndarray::{Array, Array2};

use crate::args::Args;
use crate::fft::{apply_phase_correction, process_fft, process_ifft};
use num_complex::Complex;
use crate::header::parse_header;
use crate::imaging::create_map;
use crate::plot::{plot_cross_section, plot_sky_map, plot_uv_coverage};
use crate::read::read_visibility_data;
use crate::utils::{uvw_cal, rate_cal};
use std::f64::consts::PI;

type C32 = Complex<f32>;

#[allow(unused_variables)]
#[allow(unused_mut)]
pub fn run_fringe_rate_map_analysis(args: &Args, flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)]) -> Result<(), Box<dyn Error>> {
    
    println!("Starting fringe-rate map analysis...");

    let input_path = args.input.as_ref().unwrap();

    // --- File and Path Setup ---
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let frinz_dir = parent_dir.join("frinZ").join("frmap");
    fs::create_dir_all(&frinz_dir)?;
    let file_stem = input_path.file_stem().unwrap().to_str().unwrap();

    // --- Read .cor File ---
    let mut file = fs::File::open(input_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let mut cursor = Cursor::new(buffer.as_slice());

    // --- Parse Header ---
    let header = parse_header(&mut cursor)?;
        
    // --- Pre-computation for UV coverage and B_max ---
    println!("Pre-calculating UV coverage to determine optimal cell size...");
    let mut max_b = 0.0f64;
    let mut all_uv_data: Vec<(f32, f32)> = Vec::new(); // New vector for all UV data
    let mut temp_cursor = cursor.clone();
    temp_cursor.set_position(256);
    let temp_pp = header.number_of_sector;

    let mut obs_start_time: Option<DateTime<Utc>> = None;
    let mut effective_integ_time: Option<f32> = None;

    for l1 in 0..temp_pp {
        let (_, current_obs_time, current_effective_integ_time) = match read_visibility_data(&mut temp_cursor, &header, 1, l1, 0, false) {
            Ok(data) => data,
            Err(_) => break,
        };

        if l1 == 0 {
            obs_start_time = Some(current_obs_time);
            effective_integ_time = Some(current_effective_integ_time);
        }
        let (u, v, _, _, _) = uvw_cal(
            header.station1_position,
            header.station2_position,
            current_obs_time,
            header.source_position_ra,
            header.source_position_dec,
        );
        let b = (u.powi(2) + v.powi(2)).sqrt();
        if b > max_b {
            max_b = b;
        }
        all_uv_data.push((u as f32, v as f32)); // Collect all UV data
    }
    println!("Max baseline: {:.2} m", max_b);
    // println!("DEBUG: max_b = {}", max_b);

    // --- Image Parameters ---
    let desired_map_range_arcsec = 1200.0; // Set desired map full width to 20 arcmin = 1200 arcsec
    let lambda = 299792458.0 / header.observing_frequency;
    println!("DEBUG: lambda = {}", lambda);
    let rad_to_arcsec: f64 = 180.0 / PI * 3600.0;
    // Set cell size to 1/5th of the angular resolution (lambda / B_max)
    // Calculate image_size based on desired range and cell size
    let arcsec_to_rad = PI / (180.0 * 3600.0);
    let desired_map_range_rad = desired_map_range_arcsec * arcsec_to_rad;
    let image_size: usize = 1024; // Directly set desired image size
    let cell_size_rad = desired_map_range_rad / image_size as f64; // Calculate cell size based on desired image size

    println!("DEBUG: cell_size_rad = {} rad", cell_size_rad);
    println!("Angular resolution (lambda/B_max): {:.2} arcsec", (lambda / max_b).to_degrees() * 3600.0);
    println!("Calculated cell size: {:.4e} rad ({:.4} mas)", cell_size_rad, cell_size_rad.to_degrees() * 3600e3);
    println!("DEBUG: image_size = {}", image_size);

    println!("Setting map range to ~{} arcsec with image size {}x{}", desired_map_range_arcsec, image_size, image_size);

    // --- Map Accumulator ---
    let mut total_map = ndarray::Array2::<f32>::zeros((image_size, image_size));
    let mut total_beam_map = ndarray::Array2::<f32>::zeros((image_size, image_size));
    let mut uv_data: Vec<(f32, f32)> = Vec::new();
    
    let obs_start_time = obs_start_time.expect("Failed to get observation start time");
    let effective_integ_time = effective_integ_time.expect("Failed to get effective integration time");

    cursor.set_position(0); // Reset main cursor to beginning of file
    println!("Max baseline: {:.2} m", max_b);
    // println!("DEBUG: max_b = {}", max_b);

    // --- Image Parameters ---
    let desired_map_range_arcsec = 1200.0; // Set desired map full width to 20 arcmin = 1200 arcsec
    let lambda = 299792458.0 / header.observing_frequency;
    println!("DEBUG: lambda = {}", lambda);
    let rad_to_arcsec: f64 = 180.0 / PI * 3600.0;
    // Set cell size to 1/5th of the angular resolution (lambda / B_max)
    // Calculate image_size based on desired range and cell size
    let arcsec_to_rad = PI / (180.0 * 3600.0);
    let desired_map_range_rad = desired_map_range_arcsec * arcsec_to_rad;
    let image_size: usize = 1024; // Directly set desired image size
    let cell_size_rad = desired_map_range_rad / image_size as f64; // Calculate cell size based on desired image size

    println!("DEBUG: cell_size_rad = {} rad", cell_size_rad);
    println!("Angular resolution (lambda/B_max): {:.2} arcsec", (lambda / max_b).to_degrees() * 3600.0);
    println!("Calculated cell size: {:.4e} rad ({:.4} mas)", cell_size_rad, cell_size_rad.to_degrees() * 3600e3);
    println!("DEBUG: image_size = {}", image_size);

    println!("Setting map range to ~{} arcsec with image size {}x{}", desired_map_range_arcsec, image_size, image_size);

    // --- Map Accumulator ---
    let mut total_map = ndarray::Array2::<f32>::zeros((image_size, image_size));
    let mut total_beam_map = ndarray::Array2::<f32>::zeros((image_size, image_size));
    let mut uv_data: Vec<(f32, f32)> = Vec::new();
    

    // --- Loop Setup ---
    cursor.set_position(0);
    let (_, obs_start_time, effective_integ_time) = read_visibility_data(&mut cursor, &header, 1, 0, 0, false)?;
    cursor.set_position(256);

    let pp = header.number_of_sector;
    let length_in_sectors = if args.length == 0 {
        // BUG FIX: Use a fixed segment duration (e.g., 60s) if no length is specified.
        let segment_duration_sec = pp as f32;
        (segment_duration_sec / effective_integ_time).ceil().max(1.0) as i32
    } else {
        (args.length as f32 / effective_integ_time).ceil() as i32
    };
    println!("Processing in segments of {} sectors (approx. {} seconds)", length_in_sectors, length_in_sectors as f32 * effective_integ_time);

    let total_segments_available = (pp - args.skip) / length_in_sectors;
    let loop_count = if args.loop_ == 1 { // Default loop is 1, so if user doesn't specify, process all
        total_segments_available
    } else {
        total_segments_available.min(args.loop_)
    };

    // --- Main Processing Loop ---
    for l1 in 0..loop_count {
        let (mut complex_vec, current_obs_time, effective_integ_time) = match read_visibility_data(&mut cursor, &header, length_in_sectors, args.skip, l1, false) {
            Ok(data) => data,
            Err(_) => break,
        };

        if complex_vec.is_empty() {
            break;
        }

        let is_flagged = flag_ranges.iter().any(|(start, end)| current_obs_time >= *start && current_obs_time < *end);
        if is_flagged {
            continue;
        }

        // --- Apply Phase Correction ---
        if args.delay_correct != 0.0 || args.rate_correct != 0.0 || args.acel_correct != 0.0 {
            println!("Applying phase corrections: delay={}, rate={}, acel={}", args.delay_correct, args.rate_correct, args.acel_correct);

            // 1. Reshape and convert to f64 for the correction function
            let n_rows = length_in_sectors as usize;
            let n_cols = (header.fft_point / 2) as usize;
            let mut input_data_f64: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); n_cols]; n_rows];
            for r in 0..n_rows {
                for c in 0..n_cols {
                    let index = r * n_cols + c;
                    if index < complex_vec.len() {
                        input_data_f64[r][c] = Complex::new(complex_vec[index].re as f64, complex_vec[index].im as f64);
                    }
                }
            }

            // 2. Call the correction function
            let start_time_offset_sec = (current_obs_time - obs_start_time).num_seconds() as f32;

            let corrected_data_f64 = apply_phase_correction(
                &input_data_f64,
                args.rate_correct,
                args.delay_correct,
                args.acel_correct,
                effective_integ_time,
                header.sampling_speed as u32,
                header.fft_point as u32,
                start_time_offset_sec,
            );

            // 3. Flatten and convert back to f32 for process_fft
            complex_vec = corrected_data_f64.into_iter().flatten().map(|c| C32::new(c.re as f32, c.im as f32)).collect();
        }

        // 1. Get delay-rate map
        let (freq_rate_array, padding_length) = process_fft(&complex_vec, length_in_sectors, header.fft_point, header.sampling_speed, &[], args.rate_padding);
        let delay_rate_array = process_ifft(&freq_rate_array, header.fft_point, padding_length);
        
        // 2. Get axis ranges
        let rate_range_vec = rate_cal(padding_length as f32, effective_integ_time);
        let rate_range = Array::from_vec(rate_range_vec);
        let delay_range = Array::linspace(-(header.fft_point as f32 / 2.0) + 1.0, header.fft_point as f32 / 2.0, header.fft_point as usize);

        // 3. Get UVW and derivatives for the center of the segment
        let segment_center_time = current_obs_time + chrono::Duration::microseconds(((length_in_sectors as f64 * effective_integ_time as f64 * 1_000_000.0) / 2.0) as i64);
        let (u, v, _w, du_dt, dv_dt) = uvw_cal(
            header.station1_position,
            header.station2_position,
            segment_center_time,
            header.source_position_ra,
            header.source_position_dec,
        );
        if l1 == 0 {
            println!("DEBUG: seg 0: u={}, v={}, du_dt={}, dv_dt={}", u, v, du_dt, dv_dt);
        }
        uv_data.push((u as f32, v as f32));

        // 4. Create map for this segment
        let segment_map = create_map(
            &delay_rate_array,
            u, v, du_dt, dv_dt,
            &header,
            &rate_range.view(),
            &delay_range.view(),
            image_size,
            cell_size_rad,
        );
        total_map = total_map + segment_map;

        // 5. Create beam map for this segment (Point Spread Function)
        let mut beam_delay_rate_array = Array2::zeros(delay_rate_array.dim());
        // Find the index for rate=0 Hz.
        let rate_center_idx = rate_range
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(rate_range.len() / 2);
        // Find the index for delay=0 samples.
        let delay_center_idx = delay_range
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(delay_range.len() / 2);

        beam_delay_rate_array[[rate_center_idx, delay_center_idx]] = Complex::new(1.0, 0.0);

        let segment_beam_map = create_map(
            &beam_delay_rate_array,
            u, v, du_dt, dv_dt,
            &header,
            &rate_range.view(),
            &delay_range.view(),
            image_size,
            cell_size_rad,
        );
        total_beam_map = total_beam_map + segment_beam_map;

        println!("Processed segment {}/{}", l1 + 1, loop_count);
    }

    // --- Save Final Maps and Data ---
    println!("Finished processing. Saving outputs...");

    // --- Find max value and its coordinates in the final map ---
    let mut max_val = 0.0;
    let mut max_idx = (0, 0);
    for ((y, x), &val) in total_map.indexed_iter() {
        if val > max_val {
            max_val = val;
            max_idx = (y, x);
        }
    }
    let (max_y, max_x) = max_idx;

    // Save Fringe Rate Map
    let map_filename = frinz_dir.join(format!("{}_frmap.png", file_stem));
    plot_sky_map(&map_filename, &total_map, cell_size_rad, max_x, max_y)?;
    println!("Fringe rate map saved to: {:?}", map_filename);

    // Save Fringe Rate Map Data (binary)
    let map_bin_filename = frinz_dir.join(format!("{}_frmap.bin", file_stem));
    let mut map_file = File::create(&map_bin_filename)?;
    // Write dimensions (rows, cols) as f32
    map_file.write_all(&(image_size as u32).to_le_bytes())?; // rows
    map_file.write_all(&(image_size as u32).to_le_bytes())?; // cols
    for val in total_map.iter() {
        map_file.write_all(&val.to_le_bytes())?;
    }
    println!("Fringe rate map data saved to: {:?}", map_bin_filename);

    // Save Beam Map
    let beam_map_filename = frinz_dir.join(format!("{}_beam.png", file_stem));
    plot_sky_map(&beam_map_filename, &total_beam_map, cell_size_rad, max_x, max_y)?;
    println!("Beam map saved to: {:?}", beam_map_filename);

    // Save UV Coverage Plot
    let uv_coverage_filename = frinz_dir.join(format!("{}_uv.png", file_stem));
    plot_uv_coverage(&uv_coverage_filename, &all_uv_data)?;
    println!("UV coverage plot saved to: {:?}", uv_coverage_filename);

    // Save UV Coverage Data
    let uv_bin_filename = frinz_dir.join(format!("{}_uv.bin", file_stem));
    let mut uv_file = File::create(&uv_bin_filename)?;
    for (u, v) in &all_uv_data {
        uv_file.write_all(&u.to_le_bytes())?;
        uv_file.write_all(&v.to_le_bytes())?;
    }
    println!("UV coverage data saved to: {:?}", uv_bin_filename);


    // --- Find max value and its coordinates in the final map ---
    let mut max_val = 0.0;
    let mut max_idx = (0, 0);
    for ((y, x), &val) in total_map.indexed_iter() {
        if val > max_val {
            max_val = val;
            max_idx = (y, x);
        }
    }
    let (max_y, max_x) = max_idx;

    // --- Extract and Plot Cross-sections ---
    let horizontal_profile = total_map.row(max_y);
    let vertical_profile = total_map.column(max_x);
    let (height, width) = total_map.dim();
    let ra_offsets: Vec<f64> = (0..width)
        .map(|i| ((i as f64) - (width as f64 / 2.0)) * cell_size_rad * rad_to_arcsec)
        .collect();
    let dec_offsets: Vec<f64> = (0..height)
        .map(|i| ((height as f64 / 2.0) - i as f64) * cell_size_rad * rad_to_arcsec)
        .collect();
    let horizontal_data: Vec<(f64, f32)> = ra_offsets
        .iter()
        .zip(horizontal_profile.iter())
        .map(|(&x, &y)| (x, y))
        .collect();
    let vertical_data: Vec<(f64, f32)> = dec_offsets
        .iter()
        .zip(vertical_profile.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    // --- Report Estimated Position ---
    let center = (image_size / 2) as f64;
    let l_rad = ((max_x as f64) - center) * cell_size_rad;
    let m_rad = (center - (max_y as f64)) * cell_size_rad;
    let l_arcsec = l_rad * rad_to_arcsec;
    let m_arcsec = m_rad * rad_to_arcsec;

    // --- Save Cross-section Plots ---
    let cross_section_filename = frinz_dir.join(format!("{}_frmap_peak.png", file_stem));
    plot_cross_section(
        cross_section_filename.to_str().unwrap(),
        &horizontal_data,
        &vertical_data,
        max_val,
        l_arcsec,
        m_arcsec,
    )?;
    println!("Cross-section plot saved to: {:?}", cross_section_filename);

    println!("Estimated source position (relative to phase center):");
    println!("  Delta RA: {:.3} arcsec", l_arcsec); // 西方オフセットに対して正の慣例
    println!("  Delta Dec: {:.3} arcsec", m_arcsec);

    Ok(())
}