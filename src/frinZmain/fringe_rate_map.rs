use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::io::Read;
use std::path::Path;
use std::io::{self, Write};
use std::fs::File;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use ndarray::Array;

use crate::args::Args;
use crate::fft::{apply_phase_correction, process_fft, process_ifft};
use crate::C32;
use num_complex::Complex;
use crate::header::parse_header;
use crate::imaging::create_map;
use crate::plot::{plot_cross_section, plot_sky_map};
use crate::read::read_visibility_data;
use crate::utils::{uvw_cal, rate_cal};
use std::f64::consts::PI;

pub fn run_fringe_rate_map_analysis(args: &Args, flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)]) -> Result<(), Box<dyn Error>> {
    println!("Starting fringe-rate map analysis...");

    let input_path = args.input.as_ref().unwrap();

    // --- File and Path Setup ---
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let frinz_dir = parent_dir.join("frinZ").join("fringe_rate_maps");
    fs::create_dir_all(&frinz_dir)?;

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
    let mut temp_cursor = cursor.clone();
    temp_cursor.set_position(256);
    let temp_pp = header.number_of_sector;
    for l1 in 0..temp_pp {
        let (_, current_obs_time, _) = match read_visibility_data(&mut temp_cursor, &header, 1, l1, 0, false) {
            Ok(data) => data,
            Err(_) => break,
        };
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
    }
    println!("Max baseline: {:.2} m", max_b);

    // --- Image Parameters ---
    let desired_map_range_arcsec = 1200.0; // Set desired map full width to 20 arcmin = 1200 arcsec
    let lambda = 299792458.0 / header.observing_frequency;
    let rad_to_arcsec: f64 = 180.0 / PI * 3600.0;
    // Set cell size to 1/5th of the angular resolution (lambda / B_max)
    let cell_size_rad = (lambda / max_b) / 5.0;
    println!("Angular resolution (lambda/B_max): {:.2} arcsec", (lambda / max_b).to_degrees() * 3600.0);
    println!("Calculated cell size: {:.4e} rad ({:.4} mas)", cell_size_rad, cell_size_rad.to_degrees() * 3600e3);

    // Calculate image_size based on desired range and cell size
    let arcsec_to_rad = std::f64::consts::PI / (180.0 * 3600.0);
    let desired_map_range_rad = desired_map_range_arcsec * arcsec_to_rad;
    let image_size = (desired_map_range_rad / cell_size_rad).round() as usize;
    // Ensure image_size is even for simplicity
    let image_size = if image_size % 2 != 0 { image_size + 1 } else { image_size };

    println!("Setting map range to ~{} arcsec with image size {}x{}", desired_map_range_arcsec, image_size, image_size);

    // --- Map Accumulator ---
    let mut total_map = ndarray::Array2::<f32>::zeros((image_size, image_size));

    // Initialize UVW variables outside the loop
    let mut _last_u = 0.0;
    let mut _last_v = 0.0;
    let mut _last_du_dt = 0.0;
    let mut _last_dv_dt = 0.0;
    // lambda is already defined above
    

    // --- Loop Setup ---
    cursor.set_position(0);
    let (_, obs_start_time, effective_integ_time) = read_visibility_data(&mut cursor, &header, 1, 0, 0, false)?;
    cursor.set_position(256);

    let pp = header.number_of_sector;
    let length_in_sectors = if args.length == 0 {
        // If no length is specified for fringe-rate map, use the entire file
        // to get the maximum possible rate resolution.
        pp
    } else {
        (args.length as f32 / effective_integ_time).ceil() as i32
    };

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
        let (freq_rate_array, padding_length) = process_fft(&complex_vec, length_in_sectors, header.fft_point, header.sampling_speed, &[]);
        let delay_rate_array = process_ifft(&freq_rate_array, header.fft_point, padding_length);
        // 2. Get axis ranges
        let rate_range_vec = rate_cal(padding_length as f32, effective_integ_time);
        let rate_range = Array::from_vec(rate_range_vec);
        let delay_range = Array::linspace(-(header.fft_point as f32 / 2.0) + 1.0, header.fft_point as f32 / 2.0, header.fft_point as usize);

        // 3. Get UVW and derivatives
        let (u, v, _w, du_dt, dv_dt) = uvw_cal(
            header.station1_position,
            header.station2_position,
            current_obs_time, // Use the timestamp from the start of the segment
            header.source_position_ra, // Already in radians
            header.source_position_dec, // Already in radians
        );
        _last_u = u;
        _last_v = v;
        _last_du_dt = du_dt;
        _last_dv_dt = dv_dt;

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

        // 5. Add to total map
        total_map = total_map + segment_map;
        println!("Processed segment {}/{}", l1 + 1, loop_count);
    }

    // --- Save Final Map ---
    println!("Finished processing. Saving map...");
    let output_filename = frinz_dir.join(format!("{}_fringe_rate_map.png", input_path.file_stem().unwrap().to_str().unwrap()));
    plot_sky_map(&output_filename, &total_map, cell_size_rad)?;
    println!("Map saved to: {:?}", output_filename);


    // --- Find max value and its coordinates ---
    let mut max_val = 0.0;
    let mut max_idx = (0, 0);
    for ((y, x), &val) in total_map.indexed_iter() {
        if val > max_val {
            max_val = val;
            max_idx = (y, x);
        }
    }

    let (max_y, max_x) = max_idx;

    // --- Extract cross-sections ---
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

    // --- Plot cross-sections ---
    let cross_section_filename = frinz_dir.join(format!("{}_fringe_rate_map_cross_section.png", input_path.file_stem().unwrap().to_str().unwrap()));
    plot_cross_section(
        cross_section_filename.to_str().unwrap(),
        &horizontal_data,
        &vertical_data,
        max_val,
    )?;

    println!("Cross-section plot saved to: {:?}", cross_section_filename);

    // Convert peak pixel to (l, m) in radians
    let center = (image_size / 2) as f64;
    let l_rad = ((max_x as f64) - center) * cell_size_rad;
    let m_rad = (center - (max_y as f64)) * cell_size_rad;

    // Convert to arcseconds
    let l_arcsec = l_rad * rad_to_arcsec;
    let m_arcsec = m_rad * rad_to_arcsec;

    println!("Estimated source position (relative to phase center):");
    println!("  Delta RA: {:.3} arcsec", l_arcsec);
    println!("  Delta Dec: {:.3} arcsec", m_arcsec);

    // --- Extract cross-sections ---
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

    // --- Plot cross-sections ---
    let cross_section_filename = frinz_dir.join(format!("{}_fringe_rate_map_cross_section.png", input_path.file_stem().unwrap().to_str().unwrap()));
    // plot::plot_cross_section( // Commented out
    //     cross_section_filename.to_str().unwrap(),
    //     &horizontal_data,
    //     &vertical_data,
    //     max_val,
    // )?;

    println!("Cross-section plot saved to: {:?}", cross_section_filename);

    Ok(())
}