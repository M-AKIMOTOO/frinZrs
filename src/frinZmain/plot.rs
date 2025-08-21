use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use chrono::{DateTime, Utc, TimeZone};

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
    let (heatmap_and_colorbar_area, stats_area) = right_area.split_vertically(height / 2);
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
        .y_label_area_size(120)
        .build_cartesian_2d(0.0..bw, rate_min_x..rate_max_x)?;
    heatmap_chart.configure_mesh()
        .x_desc("Frequency [MHz]")
        .y_desc("Rate [Hz]")
        .x_labels(7)
        .y_labels(7)
        .x_label_formatter(&|y| format!("{:.0}", y))
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
    source_name: &str,
    len_val: i32,
    obs_start_time: &DateTime<Utc>,
) -> Result<(), Box<dyn std::error::Error>> {
    let plots = vec![
        (amp, "Amplitude [%]", "amp"),
        (snr, "SNR", "snr"),
        (phase, "Phase [deg]", "phase"),
        (noise, "Noise Level [%]", "noise"),
    ];

    for (data, y_label, filename_suffix) in plots {
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

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("{}, length: {} s", source_name, len_val), ("sans-serif ", 25).into_font())
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(100)
            .build_cartesian_2d(*length.first().unwrap()..*length.last().unwrap(), y_min..y_max)?;

        chart.configure_mesh()
            .x_desc(&format!("The elapsed time since {} UT", obs_start_time.format("%Y/%j %H:%M:%S")))
            .y_desc(y_label)
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| {
                if filename_suffix == "phase" {
                    format!("{:.0}", v)
                } else if filename_suffix == "snr" {
                    format!("{:.0}", v)
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
    let y_label = "Phase [deg]"; // Label is now generic

    let mut chart = ChartBuilder::on(&root)
        //.caption("Phase Reference Plot", ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(100)
        .build_cartesian_2d(x_min.timestamp() as f32..x_max.timestamp() as f32, (y_min - y_range_padding)..(y_max + y_range_padding))?;

    chart.configure_mesh()
        .x_desc("Time [UTC]")
        .y_desc(y_label)
        .x_label_formatter(&|ts| {
            chrono::Utc.timestamp_opt(*ts as i64, 0).unwrap().format("%H:%M:%S").to_string()
        })
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