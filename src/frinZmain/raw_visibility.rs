use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::io::Read;
use std::path::Path;

use crate::args::Args;
use crate::header::parse_header;
use crate::read::read_visibility_data;
use crate::plot;
use crate::C32;

/// Executes the raw visibility plotting.
pub fn run_raw_visibility_plot(args: &Args) -> Result<(), Box<dyn Error>> {
    //println!("# Starting raw visibility plotting...");

    let input_path = args.input.as_ref().unwrap();

    // --- Create Output Directory ---
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let output_dir = parent_dir.join("frinZ").join("raw_visibility");
    fs::create_dir_all(&output_dir)?;
    let base_filename = input_path.file_stem().unwrap().to_str().unwrap();

    let mut file = fs::File::open(input_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let mut cursor = Cursor::new(buffer.as_slice());

    let header = parse_header(&mut cursor)?;

    let mut all_spectra: Vec<Vec<C32>> = Vec::new();
    for l1 in 0..header.number_of_sector {
        let (complex_vec, _, _) = match read_visibility_data(
            &mut cursor,
            &header,
            1, // length in sectors
            0, // skip in sectors
            l1, // loop_idx, which acts as sector index here
            false,
        ) {
            Ok(data) => data,
            Err(_) => break, // Stop if we can't read more data
        };
        if complex_vec.is_empty() {
            eprintln!("Warning: Empty sector {} found, stopping read.", l1);
            break; 
        }
        all_spectra.push(complex_vec);
    }

    if all_spectra.is_empty() {
        eprintln!("No visibility data found in the file.");
        return Ok(());
    }

    let amp_filename = format!("{}_heatmap_amp.png", base_filename);
    let phase_filename = format!("{}_heatmap_phs.png", base_filename);
    let amp_filepath = output_dir.join(amp_filename);
    let phase_filepath = output_dir.join(phase_filename);

    // Use a default sigma of 0.0 for blurring, as in the original frinZrawvis.
    plot::plot_spectrum_heatmaps(&amp_filepath, &phase_filepath, &all_spectra, 0.0)?;

    //println!("#Raw visibility heatmaps saved to {} and {}", amp_filepath.display(), phase_filepath.display());

    Ok(())
}
