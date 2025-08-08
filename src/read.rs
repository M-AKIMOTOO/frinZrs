use std::io::{self, Cursor, BufReader};
use std::fs::File;
use chrono::{DateTime, Utc, TimeZone};
use num_complex::Complex;
use byteorder::{ReadBytesExt, LittleEndian};

use crate::header::CorHeader;

type C32 = Complex<f32>;

pub fn read_visibility_data(
    cursor: &mut Cursor<&[u8]>,
    header: &CorHeader,
    length: i32,
    skip: i32,
    loop_index: i32,
    is_cumulate: bool,
) -> io::Result<(Vec<C32>, DateTime<Utc>, f32)> {
    let sector_size = (8 + header.fft_point / 4) * 16;
    let current_cursor_pos = cursor.position();

    let (actual_length_start, actual_length_end) = if is_cumulate {
        (0, length)
    } else {
        let start = skip + length * loop_index;
        let end = start + length;
        (start, end)
    };

    let mut length_end = actual_length_end;
    if length_end > header.number_of_sector {
        length_end = header.number_of_sector;
    }

    let num_sectors_to_read = (length_end - actual_length_start) as usize;
    let fft_point_half = (header.fft_point / 2) as usize;
    let mut complex_vec = Vec::with_capacity(num_sectors_to_read * fft_point_half);
    let mut obs_time = Utc.timestamp_opt(0, 0).unwrap();
    let mut effective_integ_time = 0.0;

    for i in 0..num_sectors_to_read {
        let sector_start_pos = 256 + (actual_length_start as u64 + i as u64) * sector_size as u64;
        cursor.set_position(sector_start_pos);

        let correlation_time_sec = cursor.read_i32::<byteorder::LittleEndian>()?;
        if i == 0 {
            obs_time = Utc.timestamp_opt(correlation_time_sec as i64, 0).unwrap();
        }

        cursor.set_position(sector_start_pos + 112);
        effective_integ_time = cursor.read_f32::<byteorder::LittleEndian>()?;
        cursor.set_position(sector_start_pos + 128);

        for _ in 0..fft_point_half / 2 {
            let real1 = cursor.read_f32::<byteorder::LittleEndian>()?;
            let imag1 = cursor.read_f32::<byteorder::LittleEndian>()?;
            let real2 = cursor.read_f32::<byteorder::LittleEndian>()?;
            let imag2 = cursor.read_f32::<byteorder::LittleEndian>()?;
            complex_vec.push(C32::new(real1, imag1));
            complex_vec.push(C32::new(real2, imag2));
        }
    }
    cursor.set_position(current_cursor_pos);

    Ok((complex_vec, obs_time, effective_integ_time))
}

pub fn read_bandpass_file(path: &std::path::Path) -> io::Result<Vec<C32>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    
    // Read the FFT points header (as i32)
    let _fft_points = reader.read_i32::<LittleEndian>()?;

    let mut bandpass_data = Vec::new();
    loop {
        let real = match reader.read_f32::<LittleEndian>() {
            Ok(val) => val,
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break, // End of file
            Err(e) => return Err(e),
        };
        let imag = match reader.read_f32::<LittleEndian>() {
            Ok(val) => val,
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // This case (real part read but imaginary part is EOF) should ideally not happen
                // in a correctly formatted file, but we handle it defensively.
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Incomplete complex number at end of file"));
            },
            Err(e) => return Err(e),
        };
        bandpass_data.push(C32::new(real, imag));
    }

    Ok(bandpass_data)
}