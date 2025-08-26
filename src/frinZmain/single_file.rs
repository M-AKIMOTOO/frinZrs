use std::error::Error;
use std::fs::{self, File};
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

use chrono::{DateTime, Utc};

use crate::args::Args;
use crate::output;
use crate::plot::{self, add_plot, cumulate_plot};
use crate::processing::process_cor_file;
use crate::utils;
use crate::output::generate_output_names;


pub fn run_single_file_analysis(args: &Args, flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)]) -> Result<(), Box<dyn Error>> {
    let input_path = args.input.as_ref().unwrap();
    let result = process_cor_file(input_path, args, flag_ranges)?;

    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let frinz_dir = parent_dir.join("frinZ");

    if args.cumulate != 0 {
        let path = frinz_dir.join(format!("cumulate/len{}s", args.cumulate));
        cumulate_plot(
            &result.cumulate_len,
            &result.cumulate_snr,
            &path,
            &result.header,
            &result.label.iter().map(|s| s.as_str()).collect::<Vec<&str>>(),
            &result.obs_time,
            args.cumulate,
        )?;
    }

    if args.add_plot {
        let path = frinz_dir.join("add_plot");
        let base_filename = generate_output_names(
            &result.header,
            &result.obs_time,
            &result.label.iter().map(|s| s.as_str()).collect::<Vec<&str>>(),
            !args.rfi.is_empty(),
            args.frequency,
            args.bandpass.is_some(),
            result.length_arg,
        );
        let add_plot_filename = format!("{}_{}", base_filename, result.header.source_name);
        let add_plot_filepath = path.join(add_plot_filename);

        if !result.add_plot_times.is_empty() {
            let first_time = result.add_plot_times[0];
            let elapsed_times_f32: Vec<f32> = result
                .add_plot_times
                .iter()
                .map(|dt| (*dt - first_time).num_seconds() as f32)
                .collect();

            add_plot(
                add_plot_filepath.to_str().unwrap(),
                &elapsed_times_f32, // Use elapsed time
                &result.add_plot_amp,
                &result.add_plot_snr,
                &result.add_plot_phase,
                &result.add_plot_noise,
                &result.add_plot_res_delay,
                &result.add_plot_res_rate,
                &result.header.source_name,
                result.length_arg,
                &result.obs_time, // Added this line
            )?;

            output::write_add_plot_data_to_file(
                &path,
                &base_filename,
                &elapsed_times_f32,
                &result.add_plot_amp,
                &result.add_plot_snr,
                &result.add_plot_phase,
                &result.add_plot_noise,
                &result.add_plot_res_delay,
                &result.add_plot_res_rate,
            )?;
        }
    }

    if args.allan_deviance {
        if result.add_plot_phase.len() < 3 {
            eprintln!("Warning: Not enough data points ({}) to calculate Allan deviation. Use --length and --loop to generate a time series. Skipping.", result.add_plot_phase.len());
        } else {
            println!("Calculating Allan deviation...");
            let allan_dir = frinz_dir.join("allan_deviance");
            fs::create_dir_all(&allan_dir)?;

            // tau0 is the integration time per point, which is length argument.
            let tau0 = args.length as f32;
            let adev_data = utils::calculate_allan_deviation(
                &result.add_plot_phase,
                tau0,
                result.header.observing_frequency,
            );

            let base_filename = generate_output_names(
                &result.header,
                &result.obs_time,
                &result.label.iter().map(|s| s.as_str()).collect::<Vec<&str>>(),
                !args.rfi.is_empty(),
                args.frequency,
                args.bandpass.is_some(),
                result.length_arg,
            );
            let adev_basename = format!("{}_{}_allan", base_filename, result.header.source_name);

            // Write text file
            let txt_path = allan_dir.join(format!("{}.txt", adev_basename));
            let mut writer = BufWriter::new(File::create(txt_path)?);
            writeln!(writer, "# Tau[s] Allan_Deviation")?;
            for (tau, adev) in &adev_data {
                writeln!(writer, "{} {:.6e}", tau, adev)?;
            }

            // Write plot
            let plot_path = allan_dir.join(format!("{}.png", adev_basename));
            plot::plot_allan_deviation(
                plot_path.to_str().unwrap(),
                &adev_data,
                &result.header.source_name,
            )?;
            println!("Allan deviation plot and data saved in {:?}", allan_dir);
        }
    }

    Ok(())
}
