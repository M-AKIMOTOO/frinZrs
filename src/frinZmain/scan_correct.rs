use chrono::{DateTime, Duration, Utc};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ScanCorrection {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub delay: f32,
    pub rate: f32,
}

pub fn parse_scan_correct_file(path: &Path) -> Result<Vec<ScanCorrection>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut corrections = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            eprintln!(
                "Warning: Skipping line with less than 5 columns in scan correct file: {}",
                line
            );
            continue;
        }

        let time_str = format!("{} {}", parts[0], parts[1]);
        let time_str_cleaned = time_str.replace('/', "").replace(' ', "").replace(':', "");
        let start_time = match crate::utils::parse_flag_time(&time_str_cleaned) {
            Some(t) => t,
            None => {
                eprintln!(
                    "Warning: Skipping invalid time format in scan correct file: {}",
                    time_str
                );
                continue;
            }
        };

        let duration_sec: f64 = match parts[2].parse() {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "Warning: Skipping invalid duration in scan correct file: {}",
                    parts[2]
                );
                continue;
            }
        };
        let delay: f32 = match parts[3].parse() {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "Warning: Skipping invalid delay in scan correct file: {}",
                    parts[3]
                );
                continue;
            }
        };
        let rate: f32 = match parts[4].parse() {
            Ok(r) => r,
            Err(_) => {
                eprintln!(
                    "Warning: Skipping invalid rate in scan correct file: {}",
                    parts[4]
                );
                continue;
            }
        };

        let end_time = start_time + Duration::seconds(duration_sec.round() as i64);

        corrections.push(ScanCorrection {
            start_time,
            end_time,
            delay,
            rate,
        });
    }
    Ok(corrections)
}

pub fn find_correction_for_time(
    corrections: &[ScanCorrection],
    time: &DateTime<Utc>,
) -> Option<(f32, f32)> {
    for corr in corrections {
        if *time >= corr.start_time && *time < corr.end_time {
            return Some((corr.delay, corr.rate));
        }
    }
    None
}
