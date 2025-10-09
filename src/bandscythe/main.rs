use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use clap::Parser;

const FILE_HEADER_SIZE: usize = 256;
const SECTOR_HEADER_SIZE: usize = 128;
const BYTES_PER_COMPLEX: usize = 8; // f32 (real) + f32 (imag)

#[derive(Debug, Default)]
struct CorHeader {
    magic_word: [u8; 4],
    header_version: i32,
    software_version: i32,
    sampling_speed: i32,
    observing_frequency: f64,
    fft_point: i32,
    number_of_sector: i32,
    station1_name: String,
    station1_code: String,
    station1_position: [f64; 3],
    station2_name: String,
    station2_code: String,
    station2_position: [f64; 3],
    source_name: String,
    source_position_ra: f64,
    source_position_dec: f64,
    station1_clock_delay: f64,
    station1_clock_rate: f64,
    station1_clock_acel: f64,
    station1_clock_jerk: f64,
    station1_clock_snap: f64,
    station2_clock_delay: f64,
    station2_clock_rate: f64,
    station2_clock_acel: f64,
    station2_clock_jerk: f64,
    station2_clock_snap: f64,
}

fn parse_header(cursor: &mut Cursor<&[u8]>) -> Result<CorHeader> {
    use byteorder::ReadBytesExt;

    let mut header = CorHeader::default();
    cursor.set_position(0);

    cursor.read_exact(&mut header.magic_word)?;
    header.header_version = cursor.read_i32::<LittleEndian>()?;
    header.software_version = cursor.read_i32::<LittleEndian>()?;
    header.sampling_speed = cursor.read_i32::<LittleEndian>()?;
    header.observing_frequency = cursor.read_f64::<LittleEndian>()?;
    header.fft_point = cursor.read_i32::<LittleEndian>()?;
    header.number_of_sector = cursor.read_i32::<LittleEndian>()?;

    let mut name_buf = [0u8; 8];
    cursor.read_exact(&mut name_buf)?;
    header.station1_name = String::from_utf8_lossy(&name_buf)
        .trim_end_matches('\0')
        .to_string();
    cursor.set_position(cursor.position() + 8);

    header.station1_position[0] = cursor.read_f64::<LittleEndian>()?;
    header.station1_position[1] = cursor.read_f64::<LittleEndian>()?;
    header.station1_position[2] = cursor.read_f64::<LittleEndian>()?;

    let mut code_buf = [0u8; 1];
    cursor.read_exact(&mut code_buf)?;
    header.station1_code = String::from_utf8_lossy(&code_buf).to_string();
    cursor.set_position(cursor.position() + 7);

    cursor.read_exact(&mut name_buf)?;
    header.station2_name = String::from_utf8_lossy(&name_buf)
        .trim_end_matches('\0')
        .to_string();
    cursor.set_position(cursor.position() + 8);

    header.station2_position[0] = cursor.read_f64::<LittleEndian>()?;
    header.station2_position[1] = cursor.read_f64::<LittleEndian>()?;
    header.station2_position[2] = cursor.read_f64::<LittleEndian>()?;

    cursor.read_exact(&mut code_buf)?;
    header.station2_code = String::from_utf8_lossy(&code_buf).to_string();
    cursor.set_position(cursor.position() + 7);

    let mut source_name_buf = [0u8; 16];
    cursor.read_exact(&mut source_name_buf)?;
    header.source_name = String::from_utf8_lossy(&source_name_buf)
        .trim_end_matches('\0')
        .to_string();

    header.source_position_ra = cursor.read_f64::<LittleEndian>()?;
    header.source_position_dec = cursor.read_f64::<LittleEndian>()?;

    cursor.set_position(168);
    header.station1_clock_delay = cursor.read_f64::<LittleEndian>()?;
    header.station1_clock_rate = cursor.read_f64::<LittleEndian>()?;
    header.station1_clock_acel = cursor.read_f64::<LittleEndian>()?;
    header.station1_clock_jerk = cursor.read_f64::<LittleEndian>()?;
    header.station1_clock_snap = cursor.read_f64::<LittleEndian>()?;

    cursor.set_position(216);
    header.station2_clock_delay = cursor.read_f64::<LittleEndian>()?;
    header.station2_clock_rate = cursor.read_f64::<LittleEndian>()?;
    header.station2_clock_acel = cursor.read_f64::<LittleEndian>()?;
    header.station2_clock_jerk = cursor.read_f64::<LittleEndian>()?;
    header.station2_clock_snap = cursor.read_f64::<LittleEndian>()?;

    Ok(header)
}

#[derive(Parser, Debug)]
#[command(
    name = "BandScythe",
    about = "Cut unwanted frequency ranges from a .cor file and emit a slimmed dataset.",
    version,
    arg_required_else_help = true
)]
struct Cli {
    /// Input .cor file
    #[arg(long)]
    cor: PathBuf,

    /// Frequency window to keep, in MHz (e.g. 6664-6672)
    #[arg(long, value_name = "LOW-HIGH")]
    band: String,
}

fn parse_band(spec: &str) -> Result<(f64, f64)> {
    let mut parts = spec.split('-').map(|s| s.trim()).filter(|s| !s.is_empty());
    let low = parts
        .next()
        .ok_or_else(|| anyhow!("band specification must contain a lower bound"))?
        .parse::<f64>()
        .context("failed to parse lower bound as MHz")?;
    let high = parts
        .next()
        .ok_or_else(|| anyhow!("band specification must contain an upper bound"))?
        .parse::<f64>()
        .context("failed to parse upper bound as MHz")?;
    if parts.next().is_some() {
        return Err(anyhow!(
            "band specification must be in the form LOW-HIGH (MHz)"
        ));
    }
    if low >= high {
        return Err(anyhow!("band specification requires LOW < HIGH (MHz)"));
    }
    Ok((low, high))
}

fn default_output_path(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let ext = input.extension().and_then(|s| s.to_str()).unwrap_or("cor");
    input
        .with_file_name(format!("{stem}_bandscythe.{ext}"))
        .to_path_buf()
}

fn compute_frequency_indices(
    header: &CorHeader,
    low_mhz: f64,
    high_mhz: f64,
) -> Result<(usize, usize, f64)> {
    let base_freq_hz = header.observing_frequency;
    let sampling_hz = header.sampling_speed as f64;
    let fft_point = header.fft_point as f64;
    let total_bins = (header.fft_point / 2) as usize;

    if sampling_hz <= 0.0 || fft_point <= 0.0 {
        return Err(anyhow!("invalid sampling speed or fft_point in header"));
    }

    let freq_res_hz = sampling_hz / fft_point;
    let available_low_mhz = base_freq_hz / 1e6;
    let available_high_mhz = (base_freq_hz + freq_res_hz * total_bins as f64) / 1e6;

    if low_mhz < available_low_mhz - 1e-6 || high_mhz > available_high_mhz + 1e-6 {
        return Err(anyhow!(
            "requested band {:.6}-{:.6} MHz is outside available {:.6}-{:.6} MHz",
            low_mhz,
            high_mhz,
            available_low_mhz,
            available_high_mhz
        ));
    }

    let low_hz = (low_mhz * 1e6).max(base_freq_hz);
    let high_hz = (high_mhz * 1e6).min(base_freq_hz + freq_res_hz * total_bins as f64);

    let start_idx = ((low_hz - base_freq_hz) / freq_res_hz)
        .floor()
        .clamp(0.0, total_bins as f64) as usize;
    let end_idx_exclusive = ((high_hz - base_freq_hz) / freq_res_hz)
        .ceil()
        .clamp(0.0, total_bins as f64) as usize;

    if end_idx_exclusive <= start_idx {
        return Err(anyhow!(
            "selected band does not intersect available frequency grid"
        ));
    }

    let actual_low_mhz = (base_freq_hz + freq_res_hz * start_idx as f64) / 1e6;
    Ok((start_idx, end_idx_exclusive, actual_low_mhz))
}

fn update_header_bytes(
    header_bytes: &mut [u8; FILE_HEADER_SIZE],
    new_sampling_speed: i32,
    new_observing_freq_hz: f64,
    new_fft_point: i32,
) {
    LittleEndian::write_i32(&mut header_bytes[12..16], new_sampling_speed);
    LittleEndian::write_f64(&mut header_bytes[16..24], new_observing_freq_hz);
    LittleEndian::write_i32(&mut header_bytes[24..28], new_fft_point);
    const SIGNATURE_OFFSET: usize = 248;
    const SIGNATURE: &[u8; 8] = b"bandscy\0";
    let target = &mut header_bytes[SIGNATURE_OFFSET..SIGNATURE_OFFSET + SIGNATURE.len()];
    target.copy_from_slice(SIGNATURE);
}

fn write_cropped_cor(
    mut reader: BufReader<File>,
    mut writer: BufWriter<File>,
    mut header: CorHeader,
    mut header_bytes: [u8; FILE_HEADER_SIZE],
    start_idx: usize,
    end_idx: usize,
    freq_res_hz: f64,
) -> Result<()> {
    let bins_keep = end_idx - start_idx;
    let total_bins = (header.fft_point / 2) as usize;

    let new_fft_point = (bins_keep * 2) as i32;
    let new_sampling_speed =
        ((header.sampling_speed as f64) * (bins_keep as f64) / (total_bins as f64)).round() as i32;
    let new_observing_freq_hz = header.observing_frequency + freq_res_hz * start_idx as f64;

    header.sampling_speed = new_sampling_speed;
    header.observing_frequency = new_observing_freq_hz;
    header.fft_point = new_fft_point;

    update_header_bytes(
        &mut header_bytes,
        new_sampling_speed,
        new_observing_freq_hz,
        new_fft_point,
    );

    writer
        .write_all(&header_bytes)
        .context("failed to write updated header")?;

    let mut sector_header = vec![0u8; SECTOR_HEADER_SIZE];
    let mut sample_buf = [0u8; BYTES_PER_COMPLEX];

    for sector_idx in 0..header.number_of_sector {
        reader
            .read_exact(&mut sector_header)
            .with_context(|| format!("failed to read sector header {}", sector_idx))?;
        writer
            .write_all(&sector_header)
            .with_context(|| format!("failed to write sector header {}", sector_idx))?;

        for bin in 0..total_bins {
            reader.read_exact(&mut sample_buf).with_context(|| {
                format!("failed to read data for sector {}, bin {}", sector_idx, bin)
            })?;
            if bin >= start_idx && bin < end_idx {
                writer.write_all(&sample_buf).with_context(|| {
                    format!(
                        "failed to write data for sector {}, bin {}",
                        sector_idx, bin
                    )
                })?;
            }
        }
    }

    writer.flush().context("failed to flush output file")?;
    println!(
        "BandScythe complete: kept {} / {} channels ({:.2}%), new observing freq {:.6} MHz, sampling {:.3} MHz",
        bins_keep,
        total_bins,
        (bins_keep as f64 / total_bins as f64) * 100.0,
        new_observing_freq_hz / 1e6,
        new_sampling_speed as f64 / 1e6
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let (low_mhz, high_mhz) = parse_band(&cli.band)?;
    let input_path = cli.cor;
    let output_path = default_output_path(&input_path);

    if input_path == output_path {
        return Err(anyhow!("input and output paths must differ"));
    }

    let input_file = File::open(&input_path)
        .with_context(|| format!("failed to open input file {:?}", input_path))?;
    let mut reader = BufReader::new(input_file);

    let mut header_bytes = [0u8; FILE_HEADER_SIZE];
    reader
        .read_exact(&mut header_bytes)
        .context("failed to read .cor header")?;
    let mut cursor = Cursor::new(&header_bytes[..]);
    let header = parse_header(&mut cursor).context("failed to parse .cor header")?;

    let (start_idx, end_idx, actual_low_mhz) =
        compute_frequency_indices(&header, low_mhz, high_mhz)?;
    let freq_res_hz = header.sampling_speed as f64 / header.fft_point as f64;
    let bins_keep = end_idx - start_idx;
    if bins_keep == 0 || (bins_keep & (bins_keep - 1)) != 0 {
        return Err(anyhow!(
            "selected band keeps {} channels, which is not a power of two. Adjust --band so that the channel count is 2^n.",
            bins_keep
        ));
    }

    println!(
        "Input band {:.6}-{:.6} MHz (available {:.6}-{:.6} MHz)",
        low_mhz,
        high_mhz,
        header.observing_frequency / 1e6,
        (header.observing_frequency + freq_res_hz * (header.fft_point / 2) as f64) / 1e6
    );
    println!(
        "Keeping indices {}..{} ({} channels) starting at {:.6} MHz",
        start_idx,
        end_idx,
        end_idx - start_idx,
        actual_low_mhz
    );

    let output_file = File::create(&output_path)
        .with_context(|| format!("failed to create output file {:?}", output_path))?;
    let writer = BufWriter::new(output_file);

    write_cropped_cor(
        reader,
        writer,
        header,
        header_bytes,
        start_idx,
        end_idx,
        freq_res_hz,
    )?;

    println!("Output written to {:?}", output_path);
    Ok(())
}
