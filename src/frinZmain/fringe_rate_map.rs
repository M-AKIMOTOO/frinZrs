use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::io::Read;
use std::path::Path;

use chrono::{DateTime, Utc};
use ndarray::Array;

use crate::args::Args;
use crate::fft::{process_fft, process_ifft};
use crate::header::parse_header;
use crate::imaging::create_map;
use crate::plot;
use crate::read::read_visibility_data;
use crate::utils;


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
        let (u, v, _, _, _) = utils::uvw_cal(
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
    let image_size = 768; // pixels
    let lambda = 299792458.0 / header.observing_frequency;
    // Set cell size to 1/5th of the angular resolution (lambda / B_max)
    let cell_size_rad = (lambda / max_b) / 5.0;
    println!("Calculated cell size: {:.4e} rad ({:.4} mas)", cell_size_rad, cell_size_rad * 206265e3);

    // --- Map Accumulator ---
    let mut total_map = ndarray::Array2::<f32>::zeros((image_size, image_size));

    // --- Loop Setup ---
    cursor.set_position(0);
    let (_, _, effective_integ_time) = read_visibility_data(&mut cursor, &header, 1, 0, 0, false)?;
    cursor.set_position(256);

    let pp = header.number_of_sector;
    let length_in_sectors = if args.length == 0 {
        1 // Process one sector at a time if no length is specified
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
        let (complex_vec, current_obs_time, effective_integ_time) = match read_visibility_data(&mut cursor, &header, length_in_sectors, args.skip, l1, false) {
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

        // 1. Get delay-rate map
        let (freq_rate_array, padding_length) = process_fft(&complex_vec, length_in_sectors, header.fft_point, header.sampling_speed, &[]);
        let delay_rate_array = process_ifft(&freq_rate_array, header.fft_point, padding_length);

        // 2. Get axis ranges
        let rate_range_vec = utils::rate_cal(padding_length as f32, effective_integ_time);
        let rate_range = Array::from_vec(rate_range_vec);
        let delay_range = Array::linspace(-(header.fft_point as f32 / 2.0) + 1.0, header.fft_point as f32 / 2.0, header.fft_point as usize);

        // 3. Get UVW and derivatives
        let (u, v, _w, du_dt, dv_dt) = utils::uvw_cal(
            header.station1_position,
            header.station2_position,
            current_obs_time, // Use the timestamp from the start of the segment
            header.source_position_ra, // Already in radians
            header.source_position_dec, // Already in radians
        );

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
    plot::plot_sky_map(&output_filename, &total_map, cell_size_rad)?;
    println!("Map saved to: {:?}", output_filename);

    Ok(())
}
