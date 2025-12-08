use crate::args::Args;
use crate::header::{parse_header, CorHeader};
use crate::processing::{process_cor_file, ProcessResult};
use chrono::{DateTime, Utc};
use num_complex::Complex;
use plotters::coord::ranged1d::{KeyPointHint, NoDefaultFormatting, Ranged, ValueFormatter};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};

fn wrap_phase_deg(value: f32) -> f32 {
    let mut wrapped = (value + 180.0) % 360.0;
    if wrapped < 0.0 {
        wrapped += 360.0;
    }
    wrapped - 180.0
}

fn sanitize_token(token: &str) -> String {
    token
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn read_header_only(path: &Path) -> Result<CorHeader, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; 256];
    file.read_exact(&mut buffer)?;
    let mut cursor = Cursor::new(buffer.as_slice());
    let header = parse_header(&mut cursor)?;
    Ok(header)
}

fn normalize_name(name: &str) -> String {
    name.trim().to_string()
}

fn names_equal(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

#[derive(Clone)]
struct PhaseAxis {
    start: f64,
    end: f64,
    step: f64,
}

impl PhaseAxis {
    fn new(range: Range<f64>, step: f64) -> Self {
        let mut start = range.start;
        let mut end = range.end;
        if end < start {
            std::mem::swap(&mut start, &mut end);
        }
        let step = step.abs().max(1e-6);
        PhaseAxis { start, end, step }
    }
}

impl Ranged for PhaseAxis {
    type FormatOption = NoDefaultFormatting;
    type ValueType = f64;

    fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
        let coord: RangedCoordf64 = (self.start..self.end).into();
        coord.map(value, limit)
    }

    fn key_points<Hint: KeyPointHint>(&self, hint: Hint) -> Vec<f64> {
        let mut points = Vec::new();
        if self.step <= 0.0 {
            return points;
        }
        let max_points = hint.max_num_points();
        if max_points == 0 {
            return points;
        }
        let mut current = (self.start / self.step).ceil() * self.step;
        while current <= self.end + 1e-6 && points.len() < max_points {
            points.push(current.round());
            current += self.step;
        }
        points
    }

    fn range(&self) -> Range<f64> {
        self.start..self.end
    }
}

impl ValueFormatter<f64> for PhaseAxis {
    fn format(value: &f64) -> String {
        format!("{:>4.0}", value.round())
    }
}

fn normalize_effective_integration_time(value: f32) -> f32 {
    if value >= 0.9 {
        return 1.0;
    }
    let mut a = 0.1f32;
    while a >= 0.000001 {
        if value >= a * 0.9 && value <= a {
            return a;
        }
        a /= 10.0;
    }
    value.max(0.001)
}

fn peek_effective_integ_time(path: &Path, header: &CorHeader) -> Result<f32, Box<dyn Error>> {
    if header.number_of_sector <= 0 {
        return Ok(1.0);
    }
    const FILE_HEADER_SIZE: u64 = 256;
    const EFFECTIVE_INTEG_TIME_OFFSET: u64 = 112;
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(
        FILE_HEADER_SIZE + EFFECTIVE_INTEG_TIME_OFFSET,
    ))?;
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    let value = f32::from_le_bytes(buf);
    Ok(normalize_effective_integration_time(value))
}

fn estimate_sample_count(header: &CorHeader, args: &Args, eff_time: f32) -> usize {
    let mut pp = header.number_of_sector;
    if pp <= 0 {
        pp = 1;
    }
    let mut length = if args.length == 0 {
        pp
    } else {
        args.length.max(1)
    };
    let effective_time = eff_time.max(0.0001);
    if args.length != 0 {
        let total_obs_time_seconds = pp as f32 * effective_time;
        if args.length as f32 > total_obs_time_seconds {
            length = (total_obs_time_seconds / effective_time).ceil() as i32;
        } else {
            length = (args.length as f32 / effective_time).ceil() as i32;
        }
        if length <= 0 {
            length = 1;
        }
    }
    let user_loop = args.loop_.max(1);
    let skip = args.skip.clamp(0, pp);
    let remaining = (pp - skip).max(0);
    let mut loop_count = if remaining <= 0 || length <= 0 {
        1
    } else if remaining / length <= 0 {
        1
    } else if remaining / length <= user_loop {
        remaining / length
    } else {
        user_loop
    };
    if args.cumulate != 0 {
        if args.cumulate >= pp {
            loop_count = 1;
        } else {
            loop_count = (pp / args.cumulate).max(1);
        }
    }
    loop_count.max(1) as usize
}

struct BaselineInfo {
    path: PathBuf,
    header: CorHeader,
    estimated_samples: usize,
}

struct BaselineSeries {
    baseline_label: String,
    complex_map: HashMap<String, Complex<f32>>,
}

struct ClosureSample {
    timestamp: DateTime<Utc>,
    raw_complex: [Complex<f32>; 3],
    bispectrum: Complex<f32>,
    closure_phase_deg: f32,
}

fn collect_baseline_series(
    path: &Path,
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(ProcessResult, BaselineSeries), Box<dyn Error>> {
    let mut local_args = args.clone();
    local_args.plot = false;
    local_args.add_plot = false;
    local_args.output = false;
    local_args.spectrum = false;
    local_args.header = false;

    let result = process_cor_file(path, &local_args, time_flag_ranges, pp_flag_ranges, true)?;
    if result.add_plot_times.is_empty()
        || result.add_plot_phase.is_empty()
        || result.add_plot_complex.is_empty()
    {
        return Err(format!(
            "No phase samples were produced for {}. Check --length/--loop.",
            path.display()
        )
        .into());
    }
    if result.add_plot_complex.len() != result.add_plot_times.len() {
        return Err("Mismatch between complex samples and timestamps.".into());
    }

    let label = format!(
        "{}-{}",
        result.header.station1_name.trim(),
        result.header.station2_name.trim()
    );
    let mut complex_map = HashMap::new();
    let time_fmt = "%Y/%j %H:%M:%S";
    for (timestamp, complex) in result
        .add_plot_times
        .iter()
        .zip(result.add_plot_complex.iter())
    {
        complex_map.insert(timestamp.format(time_fmt).to_string(), *complex);
    }

    Ok((
        result,
        BaselineSeries {
            baseline_label: label,
            complex_map,
        },
    ))
}

fn build_closure_rows(
    key_times: &HashMap<String, DateTime<Utc>>,
    complex_maps: &[HashMap<String, Complex<f32>>],
    sign2: f32,
    sign3: f32,
) -> Result<Vec<ClosureSample>, Box<dyn Error>> {
    let mut common_keys: Vec<String> = complex_maps
        .first()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();
    if common_keys.is_empty() {
        return Err("No phase samples found in the first baseline.".into());
    }

    common_keys.retain(|k| complex_maps.iter().all(|m| m.contains_key(k)));
    if common_keys.is_empty() {
        return Err("No overlapping epochs found across the three baselines.".into());
    }

    common_keys.sort_by_key(|k| *key_times.get(k).unwrap());

    let mut rows = Vec::with_capacity(common_keys.len());
    for key in common_keys {
        let t = *key_times.get(&key).unwrap();
        let z1 = complex_maps[0][&key];
        let z2 = complex_maps[1][&key];
        let z3 = complex_maps[2][&key];
        let adj_z2 = if sign2 >= 0.0 { z2 } else { z2.conj() };
        let adj_z3 = if sign3 >= 0.0 { z3 } else { z3.conj() };
        let bispectrum = z1 * adj_z2 * adj_z3;
        let closure_phase_deg = wrap_phase_deg(bispectrum.arg().to_degrees());
        rows.push(ClosureSample {
            timestamp: t,
            raw_complex: [z1, z2, z3],
            bispectrum,
            closure_phase_deg,
        });
    }

    Ok(rows)
}

fn write_closure_tsv(
    path: &Path,
    labels: &[String],
    rows: &[ClosureSample],
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "epoch\tRe({})\tIm({})\tRe({})\tIm({})\tRe({})\tIm({})",
        labels[0], labels[0], labels[1], labels[1], labels[2], labels[2]
    )?;
    for sample in rows {
        writeln!(
            file,
            "{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            sample.timestamp.format("%Y/%j %H:%M:%S"),
            sample.raw_complex[0].re,
            sample.raw_complex[0].im,
            sample.raw_complex[1].re,
            sample.raw_complex[1].im,
            sample.raw_complex[2].re,
            sample.raw_complex[2].im
        )?;
    }
    Ok(())
}

fn plot_closure_phase(
    path: &Path,
    labels: &[String],
    rows: &[ClosureSample],
) -> Result<(), Box<dyn Error>> {
    let drawing_area = BitMapBackend::new(path.to_str().unwrap(), (900, 600)).into_drawing_area();
    drawing_area.fill(&WHITE)?;
    let (legend_area, chart_area) = drawing_area.split_vertically(50);
    legend_area.fill(&WHITE)?;

    let first_time = rows.first().map(|row| row.timestamp).unwrap();
    let y_axis = PhaseAxis::new(-180.0f64..180.0f64, 30.0);

    let x_max = rows
        .last()
        .map(|sample| (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0)
        .unwrap_or(1.0)
        .max(1.0);
    let mut chart = ChartBuilder::on(&chart_area)
        .margin(10)
        .x_label_area_size(55)
        .y_label_area_size(65)
        .build_cartesian_2d(0.0..x_max, y_axis)?;

    chart
        .configure_mesh()
        .x_desc(format!(
            "Elapsed time [s] since {} UT",
            first_time.format("%Y/%j %H:%M:%S")
        ))
        .y_desc("Phase [deg]")
        .x_labels(13)
        .x_label_formatter(&|x| format!("{:>5.0}", x))
        .y_label_formatter(&|y| format!("{:>4.0}", *y))
        .y_labels(15)
        .label_style(("sans-serif", 24))
        .axis_desc_style(("sans-serif", 28))
        .light_line_style(&WHITE)
        .draw()?;

    let colors = [
        RGBColor(0, 102, 204),
        RGBColor(204, 102, 0),
        RGBColor(34, 139, 34),
        RGBColor(160, 32, 240),
    ];

    for (idx, label) in labels.iter().enumerate() {
        let color = colors[idx];
        chart
            .draw_series(rows.iter().map(|sample| {
                let x = (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0;
                let phase = sample.raw_complex[idx].arg().to_degrees();
                Circle::new((x, wrap_phase_deg(phase) as f64), 4, color.filled())
            }))?
            .label(label.clone())
            .legend(move |(x, y)| Circle::new((x + 10, y), 5, color.filled()));
    }

    let closure_color = colors[3];
    chart
        .draw_series(rows.iter().map(|sample| {
            let x = (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0;
            Circle::new(
                (x, sample.closure_phase_deg as f64),
                4,
                closure_color.filled(),
            )
        }))?
        .label("closure (φ1+φ2-φ3)")
        .legend(move |(x, y)| Circle::new((x + 10, y), 5, closure_color.filled()));

    // Manual legend drawn above the plot area to avoid covering data points.
    let mut x_pos = 20;
    let y_pos = 15;
    let font = ("sans-serif", 22).into_font();
    let mut draw_legend_entry =
        |text: &str, color: RGBColor| -> Result<(), Box<dyn Error>> {
            legend_area.draw(&Circle::new((x_pos, y_pos), 6, color.filled()))?;
            x_pos += 10;
            legend_area.draw(&Text::new(text.to_string(), (x_pos +10, y_pos), font.clone()))?;
            // Rough spacing: marker + text width + padding
            x_pos += text.chars().count() as i32 * 12;
            Ok(())
        };

    for (label, color) in labels.iter().zip(colors.iter()) {
        draw_legend_entry(label, *color)?;
    }
    draw_legend_entry("closure (φ1+φ2-φ3)", closure_color)?;

    Ok(())
}

fn plot_bispectrum(path: &Path, rows: &[ClosureSample]) -> Result<(), Box<dyn Error>> {
    let drawing_area = BitMapBackend::new(path.to_str().unwrap(), (900, 600)).into_drawing_area();
    drawing_area.fill(&WHITE)?;

    let first_time = rows.first().map(|row| row.timestamp).unwrap();
    let x_max = rows
        .last()
        .map(|sample| (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0)
        .unwrap_or(1.0)
        .max(1.0);
    let bispec_mags: Vec<f64> = rows
        .iter()
        .map(|sample| sample.bispectrum.norm() as f64)
        .collect();
    let mut amp_max = bispec_mags.iter().cloned().fold(0.0f64, f64::max);
    if amp_max <= 0.0 {
        amp_max = 1.0;
    } else {
        amp_max *= 1.05;
    }
    let mut amp_chart = ChartBuilder::on(&drawing_area)
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..x_max, 0.0..amp_max)?;

    amp_chart
        .configure_mesh()
        .x_desc(format!(
            "Elapsed time since {} [s]",
            first_time.format("%Y/%j %H:%M:%S")
        ))
        .y_desc("|Bispectrum|")
        .x_labels(10)
        .x_label_formatter(&|x| format!("{:>6.0}", x))
        .y_label_formatter(&|y| format!("{:>9.2e}", *y))
        .label_style(("sans-serif", 24))
        .axis_desc_style(("sans-serif", 28))
        .light_line_style(&WHITE)
        .draw()?;

    let amp_color = RGBColor(160, 32, 240);
    amp_chart
        .draw_series(rows.iter().zip(bispec_mags.iter()).map(|(sample, amp)| {
            let x = (sample.timestamp - first_time).num_milliseconds() as f64 / 1000.0;
            Circle::new((x, *amp), 4, amp_color.filled())
        }))?
        .label("bispectrum amplitude")
        .legend(move |(x, y)| Circle::new((x + 10, y), 5, amp_color.filled()));

    amp_chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    Ok(())
}

pub fn run_closure_phase_analysis(
    args: &Args,
    cor_paths: &[PathBuf],
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
    refant: &str,
) -> Result<(), Box<dyn Error>> {
    if cor_paths.len() != 3 {
        return Err("closure-phase requires exactly three .cor files.".into());
    }
    if args.frequency {
        return Err(
            "--closure-phase requires time-domain fringe processing (omit --frequency).".into(),
        );
    }

    let refant_name = refant.trim();
    if refant_name.is_empty() {
        return Err("refant must not be empty.".into());
    }

    let mut baseline_infos = Vec::with_capacity(3);
    for path in cor_paths {
        let header = read_header_only(path)?;
        let eff_time = peek_effective_integ_time(path, &header)?;
        let samples = estimate_sample_count(&header, args, eff_time);
        baseline_infos.push(BaselineInfo {
            path: path.clone(),
            header,
            estimated_samples: samples,
        });
    }

    let base1_info = &baseline_infos[0];
    let base1_st1 = normalize_name(&base1_info.header.station1_name);
    if !names_equal(&base1_st1, refant_name) {
        return Err(format!(
            "Baseline 1 station1 '{}' does not match refant '{}'.",
            base1_info.header.station1_name.trim(),
            refant_name
        )
        .into());
    }
    let mid_ant = normalize_name(&base1_info.header.station2_name);

    let base2_info = &baseline_infos[1];
    let b2_st1 = normalize_name(&base2_info.header.station1_name);
    let b2_st2 = normalize_name(&base2_info.header.station2_name);
    let (sign2, third_ant) = if names_equal(&b2_st1, &mid_ant) {
        (1.0f32, b2_st2.clone())
    } else if names_equal(&b2_st2, &mid_ant) {
        (-1.0f32, b2_st1.clone())
    } else {
        return Err(format!(
            "Baseline 2 must include antenna '{}' to close the triangle.",
            mid_ant
        )
        .into());
    };
    if names_equal(&third_ant, &mid_ant) || names_equal(&third_ant, refant_name) {
        return Err("Baseline configuration does not form a valid triangle.".into());
    }

    let base3_info = &baseline_infos[2];
    let b3_st1 = normalize_name(&base3_info.header.station1_name);
    let b3_st2 = normalize_name(&base3_info.header.station2_name);
    let orientation = if names_equal(&b3_st1, refant_name) && names_equal(&b3_st2, &third_ant) {
        1.0f32
    } else if names_equal(&b3_st2, refant_name) && names_equal(&b3_st1, &third_ant) {
        -1.0f32
    } else {
        return Err(format!(
            "Baseline 3 must connect '{}' and '{}'.",
            refant_name, third_ant
        )
        .into());
    };
    let sign3 = -orientation;

    println!("#CLOSURE INPUTS");
    for (idx, info) in baseline_infos.iter().enumerate() {
        println!(
            "#  [{}] {} | baseline {}-{} | FFT {} | PP {} | samples {}",
            idx + 1,
            info.path.display(),
            info.header.station1_name.trim(),
            info.header.station2_name.trim(),
            info.header.fft_point,
            info.header.number_of_sector,
            info.estimated_samples
        );
    }
    println!(
        "#  reference antenna: {} | triangle: {} -> {} -> {}",
        refant_name, refant_name, mid_ant, third_ant
    );
    println!(
        "#  orientation: sign2={} | sign3={}",
        if sign2 >= 0.0 { "+1" } else { "-1" },
        if sign3 >= 0.0 { "+1" } else { "-1" }
    );

    let mut processed = Vec::new();
    let time_fmt = "%Y/%j %H:%M:%S";
    let mut key_times: HashMap<String, DateTime<Utc>> = HashMap::new();

    for info in &baseline_infos {
        let (result, series) =
            collect_baseline_series(&info.path, args, time_flag_ranges, pp_flag_ranges)?;
        for timestamp in &result.add_plot_times {
            let key = timestamp.format(time_fmt).to_string();
            key_times.entry(key).or_insert(*timestamp);
        }
        processed.push((info.path.clone(), result, series));
    }

    let complex_maps: Vec<_> = processed
        .iter()
        .map(|(_, _, series)| series.complex_map.clone())
        .collect();
    let labels: Vec<_> = processed
        .iter()
        .map(|(_, _, series)| series.baseline_label.clone())
        .collect();

    let rows = build_closure_rows(&key_times, &complex_maps, sign2, sign3)?;
    println!(
        "#Closure-phase samples: {} (common epochs across all baselines)",
        rows.len()
    );

    let parent_dir = cor_paths[0]
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let frinz_dir = parent_dir.join("frinZ");
    fs::create_dir_all(&frinz_dir)?;
    let closure_dir = frinz_dir.join("closure_phase");
    fs::create_dir_all(&closure_dir)?;

    let sanitized_ref = sanitize_token(refant_name);
    let sanitized_mid = sanitize_token(&mid_ant);
    let sanitized_third = sanitize_token(&third_ant);

    let suffix1 = {
        let stem = cor_paths[0]
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let prefix = format!("{}_{}_", sanitized_ref, sanitized_mid);
        stem.strip_prefix(&prefix)
            .map(|s| s.to_string())
            .unwrap_or_else(|| stem)
    };

    let expected_suffix = suffix1.clone();

    for (idx, path) in cor_paths.iter().enumerate().skip(1) {
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let prefix = if idx == 1 {
            format!("{}_{}_", sanitized_mid, sanitized_third)
        } else {
            format!("{}_{}_", sanitized_ref, sanitized_third)
        };
        if let Some(rest) = stem.strip_prefix(&prefix) {
            if rest != expected_suffix {
                eprintln!(
                    "#WARN: File suffix '{}' differs from '{}'; using arg1 suffix.",
                    rest, expected_suffix
                );
            }
        } else {
            eprintln!(
                "#WARN: File '{}' does not start with expected prefix '{}'.",
                path.display(),
                prefix
            );
        }
    }

    let suffix_token = if expected_suffix.is_empty() {
        "unknown".to_string()
    } else {
        sanitize_token(&expected_suffix)
    };

    let timestamp = rows
        .first()
        .map(|sample| sample.timestamp.format("%Y%j%H%M%S").to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let base_name = format!(
        "{}_{}_{}_{}_{}",
        sanitized_ref, sanitized_mid, sanitized_third, suffix_token, timestamp
    );
    let tsv_path = closure_dir.join(format!("{}_complex.tsv", base_name));
    let closure_png = closure_dir.join(format!("{}_closurephase.png", base_name));
    let bis_png = closure_dir.join(format!("{}_bispectrum.png", base_name));

    write_closure_tsv(&tsv_path, &labels, &rows)?;
    plot_closure_phase(&closure_png, &labels, &rows)?;
    plot_bispectrum(&bis_png, &rows)?;

    println!("#Saved TSV to {}", tsv_path.display());
    println!("#Saved closure plot to {}", closure_png.display());
    println!("#Saved bispectrum plot to {}", bis_png.display());
    Ok(())
}
