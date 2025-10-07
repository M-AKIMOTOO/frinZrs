use byteorder::ReadBytesExt;
use chrono::{DateTime, TimeZone, Utc};
use num_complex::Complex;
use std::io::{self, Cursor, Error, ErrorKind, Read};

use crate::header::CorHeader;

type C32 = Complex<f32>;

// ファイルヘッダーのサイズ (256 バイト)
const FILE_HEADER_SIZE: u64 = 256;
// 各セクターのヘッダーサイズ (128 バイト)
const SECTOR_HEADER_SIZE: u64 = 128;
// セクターヘッダー内での有効積分時間のオフセット
const EFFECTIVE_INTEG_TIME_OFFSET: u64 = 112;

pub fn read_visibility_data(
    cursor: &mut Cursor<&[u8]>,
    header: &CorHeader,
    length: i32,
    skip: i32,
    loop_index: i32,
    is_cumulate: bool,
    pp_flag_ranges: &[(u32, u32)],
) -> io::Result<(Vec<C32>, DateTime<Utc>, f32)> {
    let sector_size = (8 + header.fft_point / 4) * 16;

    let (actual_length_start, length_end) =
        calculate_sector_range(header, length, skip, loop_index, is_cumulate);

    let num_sectors_to_read = (length_end - actual_length_start) as usize;
    let fft_point_half = (header.fft_point / 2) as usize;
    let mut complex_vec = Vec::with_capacity(num_sectors_to_read * fft_point_half);
    let mut obs_time = Utc
        .timestamp_opt(0, 0)
        .single()
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Failed to create initial timestamp"))?;
    let mut effective_integ_time = 0.0;

    for i in 0..num_sectors_to_read {
        let sector_start_pos =
            FILE_HEADER_SIZE + (actual_length_start as u64 + i as u64) * sector_size as u64;
        cursor.set_position(sector_start_pos);

        let correlation_time_sec = cursor.read_i32::<byteorder::LittleEndian>()?;
        if i == 0 {
            obs_time = Utc
                .timestamp_opt(correlation_time_sec as i64, 0)
                .single()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidData,
                        format!("Invalid timestamp seconds: {}", correlation_time_sec),
                    )
                })?;
        }

        cursor.set_position(sector_start_pos + EFFECTIVE_INTEG_TIME_OFFSET);
        effective_integ_time = cursor.read_f32::<byteorder::LittleEndian>()?;
        cursor.set_position(sector_start_pos + SECTOR_HEADER_SIZE);

        let current_pp = actual_length_start as u32 + i as u32;
        let is_pp_flagged = pp_flag_ranges
            .iter()
            .any(|(start, end)| current_pp >= *start && current_pp <= *end);

        for _ in 0..fft_point_half {
            let real = cursor.read_f32::<byteorder::LittleEndian>()?;
            let imag = cursor.read_f32::<byteorder::LittleEndian>()?;
            if is_pp_flagged {
                complex_vec.push(C32::new(0.0, 0.0));
            } else {
                complex_vec.push(C32::new(real, imag));
            }
        }
    }

    let mut a = 1.0f32;
    let mut corrected = false;

    if effective_integ_time >= 0.9 {
        // 1.0に近い場合の特別な処理
        effective_integ_time = 1.0;
        //corrected = true;
    } else {
        a /= 10.0; // 最初のaは0.1から始める
        while a >= 0.000001 && !corrected {
            // ある程度の小さい値まで繰り返す
            // effective_integ_timeがaの0.9倍からaまでの範囲にあるか
            if effective_integ_time >= a * 0.9 && effective_integ_time <= a {
                effective_integ_time = a;
                corrected = true;
            }
            a /= 10.0;
        }
    }

    Ok((complex_vec, obs_time, effective_integ_time))
}

pub fn read_sector_header(
    cursor: &mut Cursor<&[u8]>,
    header: &CorHeader,
    length: i32,
    skip: i32,
    loop_index: i32,
    is_cumulate: bool,
) -> io::Result<Vec<Vec<u8>>> {
    // 各セクターのサイズを計算します。
    let sector_size = (8 + header.fft_point / 4) * 16;

    let (actual_length_start, length_end) =
        calculate_sector_range(header, length, skip, loop_index, is_cumulate);

    let num_sectors_to_read = (length_end - actual_length_start) as usize;
    let mut sector_headers = Vec::with_capacity(num_sectors_to_read);

    for i in 0..num_sectors_to_read {
        // 各セクターのヘッダーの開始位置を計算 (ファイル先頭から256バイトのオフセット)
        let sector_start_pos =
            FILE_HEADER_SIZE + (actual_length_start as u64 + i as u64) * sector_size as u64;
        cursor.set_position(sector_start_pos);

        let mut header_buf = vec![0u8; SECTOR_HEADER_SIZE as usize];
        cursor.read_exact(&mut header_buf)?;
        sector_headers.push(header_buf);
    }

    Ok(sector_headers)
}

/// 読み込むセクターの範囲を計算するヘルパー関数
fn calculate_sector_range(
    header: &CorHeader,
    length: i32,
    skip: i32,
    loop_index: i32,
    is_cumulate: bool,
) -> (i32, i32) {
    let (start, end) = if is_cumulate {
        (0, length)
    } else {
        let start = skip + length * loop_index;
        let end = start + length;
        (start, end)
    };

    let end = end.min(header.number_of_sector);
    (start, end)
}
