use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use chrono::{DateTime, Utc, TimeZone};
use plotters::style::colors::colormaps::ViridisRGB;
use num_complex::Complex;
use std::path::Path;
use ndarray::Array2; // Added for dynamic spectrum
use crate::utils::safe_arg;
use std::f64::consts::PI;

pub fn delay_plane(
    delay_profile: &[(f64, f64)],
    rate_profile: &[(f64, f64)],
    heatmap_func: impl Fn(f64, f64) -> f64,
    stat_keys: &[&str],
    stat_vals: &[&str],
    output_path: &str,
    rate_range: &[f32],
    length: f32,
    effective_integration_length: f32,
    delay_window: &[f32],
    rate_window: &[f32],
    max_amplitude: f64,
    //cmap_time: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let width = 1400;
    let height = 1000;
    let root = BitMapBackend::new(output_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(height / 2);
    let (upper_left, upper_right) = upper.split_horizontally(width / 2 - 10);
    let (lower_left, lower_right) = lower.split_horizontally(width / 2 - 10);
    let (heatmap_area, colorbar_area) = upper_right.split_horizontally((width / 2) - 110);

    // --- Determine axis ranges ---
    let delay_max = delay_profile.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max) * 1.1;
    let rate_max = rate_profile.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max) * 1.1;

    let (delay_line_min, delay_line_max, heatmap_delay_min, heatmap_delay_max) = if delay_window.is_empty() {
        (-32.0, 32.0, -10.0, 10.0)
    } else {
        (delay_window[0] as f64, delay_window[1] as f64, delay_window[0] as f64, delay_window[1] as f64)
    };

    let (rate_line_min, rate_line_max, heatmap_rate_min, heatmap_rate_max) = if rate_window.is_empty() {
        let rate_min_x = rate_profile.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
        let rate_max_x = rate_profile.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
        let rate_win_range_low = if (-8.0 / length as f64) < rate_range[0] as f64 {
            rate_range[0] as f64 * effective_integration_length as f64
        } else {
            -4.0 / (length as f64 * effective_integration_length as f64)
        };
        let rate_win_range_high = if (8.0 / length as f64) > *rate_range.last().unwrap() as f64 {
            *rate_range.last().unwrap() as f64 * effective_integration_length as f64
        } else {
            4.0 / (length as f64 * effective_integration_length as f64)
        };
        (rate_min_x, rate_max_x, rate_win_range_low, rate_win_range_high)
    } else {
        (rate_window[0] as f64, rate_window[1] as f64, rate_window[0] as f64, rate_window[1] as f64)
    };

    // 1. Horizontal slice (Delay Profile)
    let mut chart1 = ChartBuilder::on(&upper_left)
        .margin_top(20)
        .margin_left(15)
        .x_label_area_size(65)
        .y_label_area_size(120)
        .build_cartesian_2d(delay_line_min..delay_line_max, 0.0..delay_max)?;

    chart1.configure_mesh()
        .x_desc("Delay [Sample]")
        .y_desc("Amplitude")
        .x_labels(7)
        //.y_labels(7)
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .axis_style(BLACK.stroke_width(1))
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .label_style(("sans-serif ", 30))
        .draw()?;

    chart1.draw_series(LineSeries::new(delay_profile.iter().cloned(), GREEN))?;
    
    // Draw bounding box for chart1
    let x_spec = chart1.x_range();
    let y_spec = chart1.y_range();
    chart1.draw_series(std::iter::once(Rectangle::new([(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)], BLACK.stroke_width(1))))?;

    
    // 2. Vertical slice (Rate Profile)
    let mut chart2 = ChartBuilder::on(&lower_left)
        .margin_top(10)
        .margin_left(15)
        .x_label_area_size(65)
        .y_label_area_size(120)
        .build_cartesian_2d(rate_line_min..rate_line_max, 0.0..rate_max)?;

    chart2.configure_mesh()
        .x_desc("Rate [Hz]")
        .y_desc("Amplitude")
        .x_labels(5)
        //.y_labels(7)
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .axis_style(BLACK.stroke_width(1))
        //.x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .label_style(("sans-serif ", 30))
        .draw()?;

    chart2.draw_series(LineSeries::new(rate_profile.iter().cloned(), GREEN))?;

    // Draw bounding box for chart2
    let x_spec = chart2.x_range();
    let y_spec = chart2.y_range();
    chart2.draw_series(std::iter::once(Rectangle::new([(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)], BLACK.stroke_width(1))))?;

    // 3. 2D heatmap
    let mut chart3 = ChartBuilder::on(&heatmap_area)
        .margin(20)
        .x_label_area_size(55)
        .y_label_area_size(120)
        .build_cartesian_2d(heatmap_delay_min..heatmap_delay_max, heatmap_rate_min..heatmap_rate_max)?;

    chart3.configure_mesh()
        .x_desc("Delay [Sample]")
        .y_desc("Rate [Hz]")
        .x_labels(7)
        .y_labels(10)
        .label_style(("sans-serif ", 30))
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.2e}", v))
        .draw()?;

    let resolution = 600; // Increased resolution for a smoother heatmap
    let (delay_min, delay_max_hm) = (heatmap_delay_min, heatmap_delay_max);
    let (rate_min_hm, rate_max_hm) = (heatmap_rate_min, heatmap_rate_max);
    let mut heatmap_data = Vec::new();
    let mut heatmap_data_max_val = f64::NEG_INFINITY;

    for xi in 0..resolution {
        for yi in 0..resolution {
            let x = delay_min + (delay_max_hm - delay_min) * xi as f64 / (resolution - 1) as f64;
            let y = rate_min_hm + (rate_max_hm - rate_min_hm) * yi as f64 / (resolution - 1) as f64;
            let val = heatmap_func(x, y);
            heatmap_data.push(val);
            if val > heatmap_data_max_val { heatmap_data_max_val = val; }
        }
    }

    for (idx, val) in heatmap_data.iter().enumerate() {
        let xi = idx / resolution;
        let yi = idx % resolution;
        let x = delay_min + (delay_max_hm - delay_min) * xi as f64 / (resolution - 1) as f64;
        let y = rate_min_hm + (rate_max_hm - rate_min_hm) * yi as f64 / (resolution - 1) as f64;
        let x_step = (delay_max_hm - delay_min) / (resolution - 1) as f64;
        let y_step = (rate_max_hm - rate_min_hm) / (resolution - 1) as f64;
        let normalized_val = if heatmap_data_max_val > 0.0 { *val / heatmap_data_max_val } else { 0.0 };
        chart3.draw_series(std::iter::once(Rectangle::new(
            [(x, y), (x + x_step, y + y_step)],
            HSLColor((1.0 - normalized_val) * 0.7, 1.0, 0.5).filled(),
        )))?;
    }

    // 4. Colorbar
    let mut colorbar = ChartBuilder::on(&colorbar_area)
        .margin_top(10)
        .margin_bottom(50)
        .set_label_area_size(LabelAreaPosition::Right, 100)
        .set_label_area_size(LabelAreaPosition::Left, 0)
        .set_label_area_size(LabelAreaPosition::Top, 10)
        .set_label_area_size(LabelAreaPosition::Bottom, 25)
        .build_cartesian_2d(0.0..1.0, 0.0..max_amplitude)?;

    let steps = 100;
    for i in 0..steps {
        let value = i as f64 / (steps - 1) as f64;
        let color = HSLColor(((1.0 - value) * 0.7).into(), 1.0, 0.5);
        colorbar.draw_series(std::iter::once(Rectangle::new(
            [(0.0, value * max_amplitude), (1.0, (value + 1.0 / steps as f64) * max_amplitude)],
            color.filled(),
        )))?;
    }

    colorbar.configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .disable_x_axis()
        .y_labels(7)
        .y_label_style(("sans-serif ", 30))
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .draw()?;

    // 5. Stats
    let font = ("sans-serif ", 35).into_font();
    let left_x = 30;
    let right_x = 670;
    let mut y = 20;
    let text_style = TextStyle::from(font.clone()).color(&BLACK);
    for (k, v) in stat_keys.iter().zip(stat_vals.iter()) {
        lower_right.draw(&Text::new(k.to_string(), (left_x, y), text_style.clone()))?;
        lower_right.draw(&Text::new(v.to_string(), (right_x, y + 15), text_style.clone().pos(Pos::new(HPos::Right, VPos::Center))))?;
        y += 35;
    }

    Ok(())
}

pub fn frequency_plane(
    freq_amp_profile: &[(f64, f64)],
    freq_phase_profile: &[(f64, f64)],
    rate_profile: &[(f64, f64)],
    heatmap_func: impl Fn(f64, f64) -> f64,
    stat_keys: &[&str],
    stat_vals: &[&str],
    output_path: &str,
    bw: f64,
    max_amplitude: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let width = 1400;
    let height = 1000;
    let root = BitMapBackend::new(output_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let (left_area, right_area) = root.split_horizontally(width / 2 );
    let (top_left_area, rate_area) = left_area.split_vertically(height / 2 +50);
    let (phase_area, amp_area) = top_left_area.split_vertically(top_left_area.get_pixel_range().1.len() as u32 / 4 -10);
    let (heatmap_and_colorbar_area, stats_area) = right_area.split_vertically(height / 2 +50);
    let (heatmap_area, colorbar_area) = heatmap_and_colorbar_area.split_horizontally(heatmap_and_colorbar_area.get_pixel_range().0.len() as u32 - 120);

    // --- Determine axis ranges ---
    let amp_max_y = freq_amp_profile.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max) * 1.1;
    let rate_max_y = rate_profile.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max) * 1.1;
    let rate_min_x = rate_profile.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
    let rate_max_x = rate_profile.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);

    // --- Phase Chart ---
    let mut phase_chart = ChartBuilder::on(&phase_area)
        .margin_top(20)
        .margin_left(15)
        .x_label_area_size(0)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0..bw, -180.0..180.0)?;
    phase_chart.configure_mesh()
        //.disable_x_mesh()
        //.x_labels(0)
        .x_labels(7)
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .axis_style(BLACK.stroke_width(1))
        .y_desc("Phase")
        .y_label_formatter(&|y| format!("{:.0}", y))
        .y_labels(6)
        .label_style(("sans-serif ", 30))
        .draw()?;
    phase_chart.draw_series(LineSeries::new(freq_phase_profile.iter().cloned(), GREEN.stroke_width(1)))?;

    // Draw bounding box for phase_chart
    let x_spec = phase_chart.x_range();
    let y_spec = phase_chart.y_range();
    phase_chart.draw_series(std::iter::once(Rectangle::new([(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)], BLACK.stroke_width(1))))?;

    // --- Amplitude Chart ---
    let mut amp_chart = ChartBuilder::on(&amp_area)
        .margin_left(15)
        .x_label_area_size(65)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0..bw, 0.0..amp_max_y)?;
    amp_chart.configure_mesh()
        .x_desc("Frequency [MHz]")
        .y_desc("Amplitude")
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .x_labels(7)
        .y_labels(7)
        .axis_style(BLACK.stroke_width(1))
        .label_style(("sans-serif ", 30))
        .x_label_formatter(&|y| format!("{:.0}", y))
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .draw()?;
    amp_chart.draw_series(LineSeries::new(freq_amp_profile.iter().cloned(), GREEN.stroke_width(1)))?;

    // Draw bounding box for amp_chart
    let x_spec = amp_chart.x_range();
    let y_spec = amp_chart.y_range();
    amp_chart.draw_series(std::iter::once(Rectangle::new([(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)], BLACK.stroke_width(1))))?;

    // --- Rate Chart ---
    let mut rate_chart = ChartBuilder::on(&rate_area)
        .margin_top(10)
        .margin_left(15)
        .x_label_area_size(75)
        .y_label_area_size(120)
        .build_cartesian_2d(rate_min_x..rate_max_x, 0.0..rate_max_y)?;
    rate_chart.configure_mesh()
        .x_desc("Rate [Hz]")
        .y_desc("Amplitude")
        .x_max_light_lines(0)
        .y_max_light_lines(0)
        .x_labels(7)
        //.y_labels(7)
        .axis_style(BLACK.stroke_width(1))
        .label_style(("sans-serif ", 30))
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .draw()?;
    rate_chart.draw_series(LineSeries::new(rate_profile.iter().cloned(), GREEN.stroke_width(1)))?;

    // Draw bounding box for rate_chart
    let x_spec = rate_chart.x_range();
    let y_spec = rate_chart.y_range();
    rate_chart.draw_series(std::iter::once(Rectangle::new([(x_spec.start, y_spec.start), (x_spec.end, y_spec.end)], BLACK.stroke_width(1))))?;

    // --- Heatmap Chart ---
    let mut heatmap_chart = ChartBuilder::on(&heatmap_area)
        .margin(20)
        .x_label_area_size(55)
        .y_label_area_size(85)
        .build_cartesian_2d(0.0..bw, rate_min_x..rate_max_x)?;
    heatmap_chart.configure_mesh()
        .x_desc("Frequency [MHz]")
        .y_desc("Rate [Hz]")
        .x_labels(7)
        .y_labels(7)
        .x_label_formatter(&|y| format!("{:.0}", y))
        .y_label_formatter(&|y| format!("{:.1}", y))
        .label_style(("sans-serif ", 30))
        .draw()?;

    let resolution = 400;
    let mut heatmap_values = Vec::new();
    let mut heatmap_data_max_val = f64::NEG_INFINITY;
    for i in 0..resolution {
        for j in 0..resolution {
            let y = rate_min_x + (rate_max_x - rate_min_x) * i as f64 / (resolution - 1) as f64;
            let x = 0.0 + bw * j as f64 / (resolution - 1) as f64;
            let val = heatmap_func(x, y);
            heatmap_values.push(val);
            if val > heatmap_data_max_val { heatmap_data_max_val = val; }
        }
    }

    for (idx, val) in heatmap_values.iter().enumerate() {
        let i = idx / resolution;
        let j = idx % resolution;
        let y = rate_min_x + (rate_max_x - rate_min_x) * i as f64 / (resolution - 1) as f64;
        let x = 0.0 + bw * j as f64 / (resolution - 1) as f64;
        let x_step = bw / (resolution - 1) as f64;
        let y_step = (rate_max_x - rate_min_x) / (resolution - 1) as f64;
        let normalized_val = if heatmap_data_max_val > 0.0 { *val / heatmap_data_max_val } else { 0.0 };
        heatmap_chart.draw_series(std::iter::once(Rectangle::new(
            [(x, y), (x + x_step, y + y_step)],
            HSLColor((1.0 - normalized_val) * 0.7, 1.0, 0.5).filled(),
        )))?;
    }

    // 4. Colorbar
    let mut colorbar = ChartBuilder::on(&colorbar_area)
        .margin_top(10)
        .margin_bottom(50)
        .set_label_area_size(LabelAreaPosition::Right, 100)
        .set_label_area_size(LabelAreaPosition::Left, 0)
        .set_label_area_size(LabelAreaPosition::Top, 10)
        .set_label_area_size(LabelAreaPosition::Bottom, 25)
        .build_cartesian_2d(0.0..1.0, 0.0..max_amplitude)?;

    let steps = 100;
    for i in 0..steps {
        let value = i as f64 / (steps - 1) as f64;
        let color = HSLColor(((1.0 - value) * 0.7).into(), 1.0, 0.5);
        colorbar.draw_series(std::iter::once(Rectangle::new(
            [(0.0, value * max_amplitude), (1.0, (value + 1.0 / steps as f64) * max_amplitude)],
            color.filled(),
        )))?;
    }

    colorbar.configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .disable_x_axis()
        .y_labels(7)
        .y_label_style(("sans-serif ", 30))
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .draw()?;

    // 5. Stats
    let font = ("sans-serif ", 35).into_font();
    let left_x = 30;
    let right_x = stats_area.get_pixel_range().0.len() as i32 - 30;
    let mut y = 20;
    let text_style = TextStyle::from(font.clone()).color(&BLACK);
    for (k, v) in stat_keys.iter().zip(stat_vals.iter()) {
        stats_area.draw(&Text::new(k.to_string(), (left_x, y), text_style.clone()))?;
        stats_area.draw(&Text::new(v.to_string(), (right_x, y), text_style.clone().pos(Pos::new(HPos::Right, VPos::Top))))?;
        y += 35;
    }

    root.present()?;
    Ok(())
}

pub fn add_plot(
    output_path: &str,
    length: &[f32],
    amp: &[f32],
    snr: &[f32],
    phase: &[f32],
    noise: &[f32],
    res_delay: &[f32],
    res_rate: &[f32],
    source_name: &str,
    len_val: i32,
    obs_start_time: &DateTime<Utc>,
) -> Result<(), Box<dyn std::error::Error>> {
    let plots = vec![
        (amp, "Amplitude [%]", "amp"),
        (snr, "SNR", "snr"),
        (phase, "Phase [deg]", "phase"),
        (noise, "Noise Level [%]", "noise"),
        (res_delay, "Residual Delay [sample]", "res_delay"),
        (res_rate, "Residual Rate [Hz]", "res_rate"),
    ];

    for (data, y_label, filename_suffix) in plots {
        if data.is_empty() {
            continue;
        }
        let file_path = format!("{}_{}{}", output_path, filename_suffix, ".png");
        let root = BitMapBackend::new(&file_path, (900, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut y_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let mut y_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if filename_suffix == "amp" {
            y_min = 0.0;
        } else if filename_suffix == "snr" {
            y_min = 0.0;
        } else if filename_suffix == "phase" {
            y_min = -180.0;
            y_max = 180.0;
        }

        let x_range = if length.len() > 1 {
            *length.first().unwrap()..*length.last().unwrap()
        } else {
            // Handle case with a single data point
            let center = length.first().unwrap_or(&0.0);
            (center - 1.0)..(center + 1.0)
        };

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("{}, length: {} s", source_name, len_val), ("sans-serif ", 25).into_font())
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(100)
            .build_cartesian_2d(x_range, y_min..y_max)?;

        chart.configure_mesh()
            .x_desc(&format!("The elapsed time since {} UT", obs_start_time.format("%Y/%j %H:%M:%S")))
            .y_desc(y_label)
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| {
                if filename_suffix == "phase" {
                    format!("{:.0}", v)
                } else if filename_suffix == "snr" {
                    format!("{:.0}", v)
                } else if filename_suffix == "res_delay" {
                    format!("{:.3}", v)
                } else if filename_suffix == "res_rate" {
                    format!("{:.2e}", v)
                } else {
                    format!("{:.1e}", v)
                }
            })
            .y_labels(if filename_suffix == "phase" { 7 } else { 5 })
            .label_style(("sans-serif ", 25).into_font())
            .draw()?;

        chart.draw_series(PointSeries::of_element(
            length.iter().zip(data.iter()).map(|(x, y)| (*x, *y)),
            5,
            GREEN,
            &|c, s, st| {
                return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
                    + Circle::new((0,0),s,st.filled()) // At point center, draw a circle
            },
        ))?;

        root.present()?;
    }

    Ok(())
}

pub fn cumulate_plot(
    cumulate_len: &[f32],
    cumulate_snr: &[f32],
    cumulate_path: &std::path::PathBuf,
    header: &crate::header::CorHeader,
    label: &[&str],
    obs_time: &chrono::DateTime<chrono::Utc>,
    cumulate_arg: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let base_filename = crate::output::generate_output_names(header, obs_time, label, false, false, false, cumulate_arg);
    let cumulate_filename = format!("{}_{}_cumulate{}.png", base_filename, header.source_name, cumulate_arg);
    let cumulate_filepath = cumulate_path.join(cumulate_filename);
    
    let root = BitMapBackend::new(&cumulate_filepath, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Determine the Y-axis range dynamically
    let data_min_snr = cumulate_snr.iter().cloned().filter(|&x| x > 0.0).fold(f32::INFINITY, f32::min);
    let data_max_snr = *cumulate_snr.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&100.0);

    let y_axis_start = if data_min_snr.is_finite() && data_min_snr > 0.0 {
        // Calculate the nearest power of 10 below the minimum data point
        10.0_f32.powf(data_min_snr.log10().floor())
    } else {
        // Fallback if data is empty or contains no positive values
        1.0
    };
    let y_axis_end = data_max_snr * 1.05;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(100)
        .build_cartesian_2d((*cumulate_len.first().unwrap()..*cumulate_len.last().unwrap()).log_scale(), (y_axis_start..y_axis_end).log_scale())?;

    chart.configure_mesh()
        .x_desc("Integration Time [s]")
        .y_desc("S/N ")
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.0}", v))
        .x_labels(10)
        .y_labels(10)
        .label_style(("sans-serif ", 25).into_font())
        .max_light_lines(9)
        .light_line_style(&BLACK.mix(0.15))
        .draw()?;

    chart.draw_series(LineSeries::new(cumulate_len.iter().zip(cumulate_snr.iter()).map(|(x, y)| (*x, *y)), GREEN.filled()).point_size(5)).unwrap();

    root.present()?;
    Ok(())
}

pub fn phase_reference_plot(
    cal_times: &[DateTime<Utc>],
    original_cal_phases: &[f32],
    fitted_cal_phases: &[f32],
    target_times: &[DateTime<Utc>],
    original_target_phases: &[f32],
    residual_target_phases: &[f32],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let all_times: Vec<&DateTime<Utc>> = cal_times.iter().chain(target_times.iter()).collect();
    if all_times.is_empty() {
        println!("Warning: No data to plot in phase_reference_plot.");
        return Ok(());
    }
    let mut x_min = **all_times.iter().min().unwrap();
    let mut x_max = **all_times.iter().max().unwrap();

    if x_max == x_min {
        let duration = chrono::Duration::seconds(1);
        x_min = x_min - duration;
        x_max = x_max + duration;
    }

    // Determine y-axis range from all relevant phase data
    let all_phases: Vec<f32> = original_cal_phases
        .iter()
        .chain(fitted_cal_phases.iter())
        .chain(original_target_phases.iter())
        .chain(residual_target_phases.iter())
        .cloned()
        .collect();

    let y_min = all_phases.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = all_phases.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y_range_padding = (y_max - y_min) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        //.caption("Phase Reference Plot", ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(100)
        .build_cartesian_2d(x_min.timestamp() as f32..x_max.timestamp() as f32, (y_min - y_range_padding)..(y_max + y_range_padding))?;

    chart.configure_mesh()
        .x_desc("Time [UTC]")
        .y_desc("Phase [deg]")
        .x_label_formatter(&|ts| {
            chrono::Utc.timestamp_opt(*ts as i64, 0).unwrap().format("%H:%M:%S").to_string()
        })
        .y_label_formatter(&|v| format!("{:.0}", v))
        .y_labels(7)
        .label_style(("sans-serif", 25).into_font())
        .draw()?;

    // 1. Original Calibrator Phase (Red Points)
    chart.draw_series(PointSeries::of_element(
        cal_times.iter().zip(original_cal_phases.iter()).map(|(x, y)| (x.timestamp() as f32, *y)),
        5,
        &RED,
        &|c, s, st| Circle::new(c, s, st.filled()),
    ))?
    .label("Calibrator (Original)")
    .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));

    // 2. Fitted Calibrator Phase (Blue Line)
    if !fitted_cal_phases.is_empty() {
        chart.draw_series(LineSeries::new(
            cal_times.iter().zip(fitted_cal_phases.iter()).map(|(x, y)| (x.timestamp() as f32, *y)),
            &BLUE,
        ))?
        .label("Calibrator (Fitted)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));
    }

    // 3. Original Target Phase (Green Circles)
    chart.draw_series(PointSeries::of_element(
        target_times.iter().zip(original_target_phases.iter()).map(|(x, y)| (x.timestamp() as f32, *y)),
        5,
        &GREEN,
        &|c, s, st| Circle::new(c, s, st.filled()),
    ))?
    .label("Target (Original)")
    .legend(|(x, y)| Circle::new((x, y), 5, GREEN.filled()));

    // 4. Residual Phases from Calibrator Fit (Black Circles)
    chart.draw_series(PointSeries::of_element(
        target_times.iter().zip(residual_target_phases.iter()).map(|(x, y)| (x.timestamp() as f32, *y)),
        5,
        &BLACK,
        &|c, s, st| Circle::new(c, s, st.filled()),
    ))?
    .label("Residual (Calibrator Fit)")
    .legend(|(x, y)| Circle::new((x, y), 5, BLACK.filled()));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn plot_allan_deviation(
    output_path: &str,
    data: &[(f32, f32)],
    source_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    if data.is_empty() {
        return Ok(());
    }

    let (min_tau, max_tau) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), (t, _)| (min.min(*t), max.max(*t)));
    let (min_adev, max_adev) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), (_, a)| (min.min(*a), max.max(*a)));

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Allan Deviation for {}", source_name), ("sans-serif", 25).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(100)
        .build_cartesian_2d((min_tau..max_tau).log_scale(), (min_adev..max_adev*1.1).log_scale())?;

    chart.configure_mesh()
        .x_desc("Averaging Time (τ) [s]")
        .y_desc("Allan Deviation (σ_y(τ))")
        .x_labels(10)
        .y_labels(10)
        .label_style(("sans-serif", 25).into_font())
        .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|(t, a)| (*t, *a)),
        &RED,
    ))?;
    chart.draw_series(PointSeries::of_element(
        data.iter().map(|(t, a)| (*t, *a)),
        5,
        &RED,
        &|c, s, st| Circle::new(c, s, st.filled()),
    ))?;

    root.present()?;
    Ok(())
}

pub fn plot_acel_search_result(
    input_file_path: &str,
    output_file_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::{BufReader, BufRead};

    let file = File::open(input_file_path)?;
    let reader = BufReader::new(file);

    let mut data = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            let time: f64 = parts[0].parse()?;
            let phase: f64 = parts[1].parse()?;
            data.push((time, phase));
        }
    }

    if data.is_empty() {
        println!("Warning: No data to plot in {}.", input_file_path);
        return Ok(());
    }

    // Determine min/max for axes
    let (min_time, max_time) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _)| (min_t.min(*t), max_t.max(*t)));
    let (min_phase, max_phase) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_p, max_p), (_, p)| (min_p.min(*p), max_p.max(*p)));

    let root = BitMapBackend::new(output_file_path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Acel Search Result: Time vs Phase", ("sans-serif", 25).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(100)
        .build_cartesian_2d(min_time..max_time, min_phase..max_phase)?;

    chart.configure_mesh()
        .x_desc("Time [s]")
        .y_desc("Unwarnpped Phase [rad]")
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.2}", v))
        .x_labels(10)
        .y_labels(10)
        .label_style(("sans-serif", 20).into_font())
        .draw()?;

    chart.draw_series(PointSeries::of_element(
        data.iter().map(|(t, p)| (*t, *p)),
        3,
        &BLUE,
        &|c, s, st| Circle::new(c, s, st.filled()),
    ))?;

    root.present()?;
    Ok(())
}

pub fn plot_sky_map<P: AsRef<Path>>(
    output_path: P,
    map_data: &ndarray::Array2<f32>,
    cell_size_rad: f64,
    max_x_idx: usize,
    max_y_idx: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let (height, width) = map_data.dim();
    // Define a fixed output image size for consistent plot appearance
    let backend_width = 1024;
    let backend_height = 800;
    let root = BitMapBackend::new(output_path.as_ref(), (backend_width, backend_height)).into_drawing_area();
    root.fill(&WHITE)?;

    let (main_area, colorbar_area) = root.split_horizontally(backend_width - 120);

    let max_val = map_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = map_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    let rad_to_arcsec = 180.0 / PI * 3600.0;
    // Define coordinate ranges in arcseconds for the chart
    let l_arcsec_max = (width as f64 / 2.0) * cell_size_rad * rad_to_arcsec;
    let m_arcsec_max = (height as f64 / 2.0) * cell_size_rad * rad_to_arcsec;
    let l_range = l_arcsec_max..-l_arcsec_max; // Invert RA axis
    let m_range = -m_arcsec_max..m_arcsec_max;

    let mut chart = ChartBuilder::on(&main_area)
        .x_label_area_size(65)
        .y_label_area_size(80)
        .margin(10)
        .caption("Fringe Rate Map", ("sans-serif", 25))
        .build_cartesian_2d(l_range.clone(), m_range.clone())?;

    chart.configure_mesh()
        .x_desc("ΔRA (arcsec)")
        .y_desc("ΔDec (arcsec)")
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .label_style(("sans-serif", 30))
        .draw()?;

    // Draw the heatmap
    chart.draw_series(
        map_data.indexed_iter().map(|((y, x), &val)| {
            let l_arcsec = ((x as f64) - (width as f64 / 2.0)) * cell_size_rad * rad_to_arcsec;
            let m_arcsec = (((height as f64 / 2.0) - y as f64)) * cell_size_rad * rad_to_arcsec;
            let cell_l_arcsec = cell_size_rad * rad_to_arcsec;
            let cell_m_arcsec = cell_size_rad * rad_to_arcsec;

            let mut norm_val = if max_val > min_val { (val - min_val) / (max_val - min_val) } else { 0.0 };
            if norm_val.is_nan() { norm_val = 0.0; }
            norm_val = norm_val.clamp(0.0, 1.0);
            let color = ViridisRGB.get_color(norm_val as f64);
            
            Rectangle::new([(l_arcsec, m_arcsec), (l_arcsec + cell_l_arcsec, m_arcsec + cell_m_arcsec)], color.filled())
        })
    )?;

    // Add a white 'X' mark at (0,0)
    chart.draw_series(PointSeries::of_element(
        vec![(0.0, 0.0)], // Center of the 'X'
        10, // Size of the 'X'
        &WHITE, // Color of the 'X'
        &|c, s, st| Cross::new(c, s, st.stroke_width(2)), // Draw a cross
    ))?;

    // Add a red 'X' mark at the maximum value position
    let (height, width) = map_data.dim(); // Get dimensions for calculation
    let rad_to_arcsec: f64 = 180.0 / PI * 3600.0; // Re-define rad_to_arcsec for local use
    let l_arcsec_max_val = ((max_x_idx as f64) - (width as f64 / 2.0)) * cell_size_rad * rad_to_arcsec;
    let m_arcsec_max_val = (((height as f64 / 2.0) - max_y_idx as f64)) * cell_size_rad * rad_to_arcsec;

    chart.draw_series(PointSeries::of_element(
        vec![(l_arcsec_max_val, m_arcsec_max_val)], // Center of the 'X'
        10, // Size of the 'X'
        &RED, // Color of the 'X'
        &|c, s, st| Cross::new(c, s, st.stroke_width(2)), // Draw a cross
    ))?;

    // Draw color bar
    let mut colorbar_chart = ChartBuilder::on(&colorbar_area)
        .margin(20)
        .set_label_area_size(LabelAreaPosition::Right, 40)
        .build_cartesian_2d(0f32..1f32, min_val..max_val)?;
    
    colorbar_chart.configure_mesh()
        .disable_x_mesh().disable_x_axis()
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .y_label_style(("sans-serif", 20))
        .draw()?;

    let color_map = ViridisRGB;
    colorbar_chart.draw_series((0..200).map(|y| {
        let y_val = min_val + (max_val - min_val) * y as f32 / 199.0;
        let mut norm_val = if max_val > min_val { (y_val - min_val) / (max_val - min_val) } else { 0.0 };
        if norm_val.is_nan() { norm_val = 0.0; }
        norm_val = norm_val.clamp(0.0, 1.0);
        let color = color_map.get_color(norm_val as f64);
        Rectangle::new([(0.0, y_val), (1.0, y_val + (max_val-min_val)/199.0)], color.filled())
    }))?;

    root.present()?;
    Ok(())
}

pub fn plot_dynamic_spectrum_freq(
    output_path: &str,
    spectrum_array: &Array2<Complex<f32>>,
    header: &crate::header::CorHeader,
    obs_time: &DateTime<Utc>,
    _length: i32,
    _effective_integration_time: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let (time_samples, freq_channels) = spectrum_array.dim();

    let width = 1200;
    let height = 1000;
    let root = BitMapBackend::new(output_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(height / 2);

    let freq_range = 0..freq_channels;
    let time_range = 0..time_samples;

    // --- Amplitude Heatmap ---
    let amplitudes: Vec<f32> = spectrum_array.iter().map(|c| c.norm()).collect();
    let max_amp = amplitudes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_amp = amplitudes.iter().cloned().fold(f32::INFINITY, f32::min);

    let mut amp_chart = ChartBuilder::on(&upper)
        .caption(
            format!(
                "Dynamic Spectrum (Amplitude) - {} - {}",
                header.source_name,
                obs_time.format("%Y-%m-%d %H:%M:%S")
            ),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(5)
        .y_label_area_size(100)
        .build_cartesian_2d(freq_range.clone(), time_range.clone())?;

    amp_chart
        .configure_mesh()
        .y_desc("Time [PP]")
        //.x_desc("Frequency [channels]")
        .label_style(("sans-serif", 25).into_font())
        .draw()?;

    amp_chart.draw_series(spectrum_array.indexed_iter().map(|((t, f), c)| {
        let norm_val = if max_amp > min_amp {
            (c.norm() - min_amp) / (max_amp - min_amp)
        } else {
            0.0
        };
        let color = ViridisRGB.get_color(norm_val as f64);
        Rectangle::new([(f, t), (f + 1, t + 1)], color.filled())
    }))?;

    // --- Phase Heatmap ---
    let mut phase_chart = ChartBuilder::on(&lower)
        .caption(
            format!(
                "Dynamic Spectrum (Phase) - {} - {}",
                header.source_name,
                obs_time.format("%Y-%m-%d %H:%M:%S")
            ),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(55)
        .y_label_area_size(100)
        .build_cartesian_2d(freq_range.clone(), time_range.clone())?;

    phase_chart
        .configure_mesh()
        .y_desc("Time [PP]")
        .x_desc("Frequency [channels]")
        .label_style(("sans-serif", 25).into_font())
        .draw()?;

    phase_chart.draw_series(spectrum_array.indexed_iter().map(|((t, f), c)| {
        let norm_val = (safe_arg(c).to_degrees() + 180.0) / 360.0;
        let color = ViridisRGB.get_color(norm_val as f64);
        Rectangle::new([(f, t), (f + 1, t + 1)], color.filled())
    }))?;

    root.present()?;
    Ok(())
}

pub fn plot_dynamic_spectrum_lag(
    output_path: &str,
    lag_data: &Array2<f32>,
    header: &crate::header::CorHeader,
    obs_time: &DateTime<Utc>,
    _length: i32,
    _effective_integration_time: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let (time_samples, lag_samples) = lag_data.dim();

    let width = 1200;
    let height = 600;
    let root = BitMapBackend::new(output_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let lag_range_min = -(lag_samples as i32 / 2);
    let lag_range_max = lag_samples as i32 / 2;
    let lag_range = lag_range_min..lag_range_max;
    let time_range = 0..time_samples;

    let max_val = lag_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = lag_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "Dynamic Spectrum (Time Lag) - {} - {}",
                header.source_name,
                obs_time.format("%Y-%m-%d %H:%M:%S")
            ),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(55)
        .y_label_area_size(100)
        .build_cartesian_2d(lag_range.clone(), time_range.clone())?;

    chart
        .configure_mesh()
        .y_desc("Time [PP]")
        .x_desc("Lag [samples]")
        .label_style(("sans-serif", 25).into_font())
        .draw()?;

    chart.draw_series(lag_data.indexed_iter().map(|((t, l), &val)| {
        let x = lag_range_min + l as i32;
        let norm_val = if max_val > min_val {
            (val - min_val) / (max_val - min_val)
        } else {
            0.0
        };
        let color = ViridisRGB.get_color(norm_val as f64);
        Rectangle::new([(x, t), (x + 1, t + 1)], color.filled())
    }))?;

    root.present()?;
    Ok(())
}

fn gaussian_blur_2d(data: &Vec<Vec<f32>>, sigma: f32) -> Vec<Vec<f32>> {
    if sigma <= 0.0 {
        return data.clone();
    }

    let rows = data.len();
    if rows == 0 { return Vec::new(); }
    let cols = data[0].len();
    if cols == 0 { return data.clone(); }

    let kernel_radius = (sigma * 3.0).ceil() as usize;
    let kernel_size = 2 * kernel_radius + 1;
    let mut kernel = vec![0.0; kernel_size];
    let mut kernel_sum = 0.0;

    for i in 0..kernel_size {
        let x = i as f32 - kernel_radius as f32;
        kernel[i] = (-0.5 * (x / sigma).powi(2)).exp();
        kernel_sum += kernel[i];
    }

    for i in 0..kernel_size {
        kernel[i] /= kernel_sum;
    }

    let mut blurred_data = vec![vec![0.0; cols]; rows];
    let mut temp_data = vec![vec![0.0; cols]; rows];

    // Apply horizontal blur
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0;
            for k_idx in 0..kernel_size {
                let col_idx = (c as isize + k_idx as isize - kernel_radius as isize).clamp(0, cols as isize - 1) as usize;
                sum += data[r][col_idx] * kernel[k_idx];
            }
            temp_data[r][c] = sum;
        }
    }

    // Apply vertical blur
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0;
            for k_idx in 0..kernel_size {
                let row_idx = (r as isize + k_idx as isize - kernel_radius as isize).clamp(0, rows as isize - 1) as usize;
                sum += temp_data[row_idx][c] * kernel[k_idx];
            }
            blurred_data[r][c] = sum;
        }
    }

    blurred_data
}

fn plot_single_heatmap_with_colorbar(
    output_path: &Path,
    data: &Vec<Vec<f32>>,
    title: &str,
    x_desc: &str,
    y_desc: &str,
    color_bar_title: &str,
    min_val: f32,
    max_val: f32,
    num_color_bar_labels: usize,
    color_value_normalizer: impl Fn(f32) -> f64,
    color_bar_label_formatter: impl Fn(f32) -> String,
) -> Result<(), Box<dyn std::error::Error>> {
    let (rows, cols) = (data.len(), data[0].len());

    let main_chart_width = 500;
    let color_bar_area_width = 110;
    let total_width = main_chart_width + color_bar_area_width;
    let total_height = 384;

    let root = BitMapBackend::new(output_path, (total_width, total_height)).into_drawing_area();
    root.fill(&WHITE)?;

    let (chart_area, color_bar_area) = root.split_horizontally(main_chart_width);

    let top_margin = 10;
    let bottom_margin = 10;
    let x_label_area_size = 35;
    let y_label_area_size = 45;

    let mut chart = ChartBuilder::on(&chart_area)
        .caption(title, ("sans-serif", 20).into_font())
        .margin_top(top_margin)
        .margin_bottom(bottom_margin)
        .margin_left(10)
        .margin_right(10)
        .x_label_area_size(x_label_area_size)
        .y_label_area_size(y_label_area_size)
        .build_cartesian_2d(0..cols, 0..rows)?;

    chart.configure_mesh()
        .x_desc(x_desc)
        .y_desc(y_desc)
        .x_label_style(("sans-serif", 18).into_font())
        .y_label_style(("sans-serif", 18).into_font())
        .draw()?;

    chart.draw_series(
        (0..cols).flat_map(|x| (0..rows).map(move |y| (x, y)))
        .map(|(x, y)| {
            let val = data[y][x];
            let color_value = color_value_normalizer(val);
            let color = ViridisRGB.get_color(color_value);
            Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())
        })
    )?;

    // Draw color bar
    let color_bar_width = 25;
    let color_bar_x_offset = 0;
    let color_bar_y_offset = top_margin as i32;
    let color_bar_height = total_height as i32 - (top_margin + bottom_margin + x_label_area_size) as i32;

    for i in 0..color_bar_height {
        let color_value = i as f64 / (color_bar_height - 1) as f64;
        let color = ViridisRGB.get_color(color_value);
        color_bar_area.draw(&Rectangle::new(
            [(color_bar_x_offset, color_bar_y_offset + i), (color_bar_x_offset + color_bar_width, color_bar_y_offset + i + 1)],
            color.filled(),
        ))?;
    }

    // Add labels to the color bar
    color_bar_area.draw_text(
        color_bar_title,
        &TextStyle::from(("sans-serif", 18).into_font()).color(&BLACK).transform(FontTransform::Rotate270),
        (color_bar_area_width as i32 - 25, total_height as i32 / 2),
    )?;

    for i in 0..num_color_bar_labels {
        let val_fraction = i as f32 / (num_color_bar_labels - 1) as f32;
        let value = min_val + (max_val - min_val) * val_fraction;
        let y_pos = color_bar_y_offset + color_bar_height - (val_fraction * color_bar_height as f32) as i32;
        color_bar_area.draw_text(
            &color_bar_label_formatter(value),
            &TextStyle::from(("sans-serif", 18).into_font()).color(&BLACK),
            (color_bar_x_offset + color_bar_width + 5, y_pos - 7),
        )?;
    }

    root.present()?;
    Ok(())
}

pub fn plot_spectrum_heatmaps<P: AsRef<Path>>(
    output_path_amplitude: P,
    output_path_phase: P,
    spectrum_data: &Vec<Vec<Complex<f32>>>,
    sigma: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if spectrum_data.is_empty() || spectrum_data[0].is_empty() {
        return Err("Spectrum data for heatmap is empty".into());
    }

    // --- Amplitude Heatmap ---
    let amplitudes_2d: Vec<Vec<f32>> = spectrum_data.iter().map(|row| row.iter().map(|c| c.norm()).collect()).collect();
    let blurred_amplitudes = gaussian_blur_2d(&amplitudes_2d, sigma);
    let max_amp = blurred_amplitudes.iter().flatten().cloned().fold(0.0, f32::max);
    let min_amp = blurred_amplitudes.iter().flatten().cloned().fold(f32::MAX, f32::min);

    plot_single_heatmap_with_colorbar(
        output_path_amplitude.as_ref(),
        &blurred_amplitudes,
        "Amplitude Spectrum",
        "Channels",
        "PP",
        "Amplitude (a.u.)",
        min_amp,
        max_amp,
        5,
        |v| if max_amp > min_amp { ((v - min_amp) / (max_amp - min_amp)) as f64 } else { 0.0 },
        |v| format!("{:.1e}", v),
    )?;

    // --- Phase Heatmap ---
    let phases_2d: Vec<Vec<f32>> = spectrum_data.iter().map(|row| row.iter().map(|c| safe_arg(c).to_degrees()).collect()).collect();
    let blurred_phases = gaussian_blur_2d(&phases_2d, sigma);

    plot_single_heatmap_with_colorbar(
        output_path_phase.as_ref(),
        &blurred_phases,
        "Phase Spectrum",
        "Channels",
        "PP",
        "Phase (deg)",
        -180.0,
        180.0,
        9,
        |v| ((v + 180.0) / 360.0) as f64,
        |v| format!("{:.0}", v),
    )?;

    Ok(())
}

pub fn plot_cross_section(
    output_path: &str,
    horizontal_data: &[(f64, f32)], // (RA offset, Intensity)
    vertical_data: &[(f64, f32)],   // (Dec offset, Intensity)
    max_intensity: f32,
    delta_ra_arcsec: f64,
    delta_dec_arcsec: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1000, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let (h_min, h_max) = horizontal_data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| (min.min(*x), max.max(*x)));
    let (v_min, v_max) = vertical_data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| (min.min(*x), max.max(*x)));
    let x_min = h_min.min(v_min);
    let x_max = h_max.max(v_max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Fringe Rate Map Cross-section at Peak", ("sans-serif", 25).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(x_max..x_min, 0.0..1.1f64)? // Y-axis is normalized intensity
        ;

    chart.configure_mesh()
        .x_desc("Offset (arcsec)")
        .y_desc("Normalized Intensity")
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .label_style(("sans-serif", 30).into_font())
        .draw()?;

    // Draw horizontal cross-section (RA)
    chart.draw_series(LineSeries::new(
        horizontal_data.iter().map(|(x, y)| (*x, (*y / max_intensity) as f64)),
        &BLUE,
    ))?
    .label("RA Cross-section")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));

    // Draw vertical cross-section (Dec)
    chart.draw_series(LineSeries::new(
        vertical_data.iter().map(|(x, y)| (*x, (*y / max_intensity) as f64)),
        &RED,
    ))?
    .label("Dec Cross-section")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // Add Delta RA/Dec to legend
    let font = ("sans-serif", 35).into_font();
    let text_style = TextStyle::from(font.clone()).color(&BLACK);
    let x_pos = root.get_pixel_range().0.end as i32 - 300;
    let mut y_pos = root.get_pixel_range().1.start as i32 + 20;

    root.draw(&Text::new(format!("ΔRA: {:.3} arcsec", delta_ra_arcsec), (x_pos, y_pos), text_style.clone()))?;
    y_pos += 25;
    root.draw(&Text::new(format!("ΔDec: {:.3} arcsec", delta_dec_arcsec), (x_pos, y_pos), text_style.clone()))?;

    root.present()?;
    Ok(())
}

pub fn plot_uv_coverage<P: AsRef<Path>>(
    output_path: P,
    uv_data: &[(f32, f32)], // Data is (u, v) in meters
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path.as_ref(), (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    if uv_data.is_empty() {
        return Ok(());
    }

    // Find range for u and v, then make the plot symmetrical
    let u_max = uv_data.iter().map(|(u, _)| u.abs()).fold(0.0, f32::max);
    let v_max = uv_data.iter().map(|(_, v)| v.abs()).fold(0.0, f32::max);
    let max_abs = u_max.max(v_max) * 1.1; // Add 10% padding

    let mut chart = ChartBuilder::on(&root)
        .caption("UV Coverage", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(-max_abs..max_abs, -max_abs..max_abs)?;

    chart.configure_mesh()
        .x_desc("U (meters)")
        .y_desc("V (meters)")
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .label_style(("sans-serif", 25))
        .draw()?;

    chart.draw_series(
        uv_data.iter().map(|(u, v)| Circle::new((*u, *v), 2, BLUE.filled()))
    )?;
    
    // Also plot the symmetrical points
    chart.draw_series(
        uv_data.iter().map(|(u, v)| Circle::new((-*u, -*v), 2, RED.filled()))
    )?;

    root.present()?;
    Ok(())
}
