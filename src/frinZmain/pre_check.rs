use std::error::Error;
use std::fs::File;
use std::io::{self, Cursor, Read, Write};
use std::path::Path;

use crate::args::Args;
use crate::header::parse_header;

pub fn check_memory_usage(args: &Args, input_path: &Path) -> Result<bool, Box<dyn Error>> {
    let mut file = File::open(input_path)?;
    let mut buffer = vec![0; 256];
    file.read_exact(&mut buffer)?;
    let mut cursor = Cursor::new(buffer.as_slice());
    let header = parse_header(&mut cursor)?;

    let fft_point = header.fft_point as u64;
    let pp = header.number_of_sector as u64;
    let rate_padding = args.rate_padding as u64;

    let required_memory = 4 * fft_point * pp.next_power_of_two() * rate_padding; // byte

    let mem_info = sys_info::mem_info()?;
    let total_ram = mem_info.total * 1024; // Convert KB to Bytes
    let quarter_ram = total_ram / 4;

    if required_memory > quarter_ram {

        println!(
            "Warning: The estimated memory usage ({:.2} GB) exceeds 25% of your system RAM ({:.2} GB).",
            required_memory as f64 / 1_073_741_824.0,
            total_ram as f64 / 1_073_741_824.0
        );
        print!("Do you want to continue? (y/n): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if input.trim().to_lowercase() != "y" {
            println!("Aborting.");
            return Ok(false);
        }
    }

    Ok(true)
}
