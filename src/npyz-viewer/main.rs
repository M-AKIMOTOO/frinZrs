use eframe::egui;
use ndarray::IxDyn;
use ndarray_npy::{NpzReader, read_npy};
use std::fmt::Display;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const CELL_WIDTH: f32 = 160.0;
const ROW_NUM_WIDTH: f32 = 56.0;
const ROW_HEIGHT: f32 = 24.0;
const FORCE_EXIT_ON_VIEWPORT_X: bool = true;

fn main() -> eframe::Result<()> {
    match parse_launch_mode() {
        LaunchMode::Browser { initial_path } => {
            let native_options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_title("npyz-viewer")
                    .with_inner_size(egui::vec2(680.0, 520.0)),
                renderer: eframe::Renderer::Glow,
                ..Default::default()
            };
            eframe::run_native(
                "npyz-viewer",
                native_options,
                Box::new(move |_cc| {
                    Ok(Box::new(NpzBrowserApp::from_initial_path(
                        initial_path.clone(),
                    )))
                }),
            )
        }
        LaunchMode::ViewNpy { path } => {
            let app = NpyViewerApp::from_npy_path(path);
            let title = app.title.clone();
            let native_options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_title(&title)
                    .with_inner_size(egui::vec2(1180.0, 760.0)),
                renderer: eframe::Renderer::Glow,
                ..Default::default()
            };
            eframe::run_native(
                "NPY Viewer",
                native_options,
                Box::new(move |_cc| Ok(Box::new(app))),
            )
        }
        LaunchMode::ViewNpzEntry { npz_path, entry } => {
            let app = NpyViewerApp::from_npz_entry(npz_path, entry);
            let title = app.title.clone();
            let native_options = eframe::NativeOptions {
                viewport: egui::ViewportBuilder::default()
                    .with_title(&title)
                    .with_inner_size(egui::vec2(1180.0, 760.0)),
                renderer: eframe::Renderer::Glow,
                ..Default::default()
            };
            eframe::run_native(
                "NPY Viewer",
                native_options,
                Box::new(move |_cc| Ok(Box::new(app))),
            )
        }
    }
}

enum LaunchMode {
    Browser { initial_path: Option<PathBuf> },
    ViewNpy { path: PathBuf },
    ViewNpzEntry { npz_path: PathBuf, entry: String },
}

fn parse_launch_mode() -> LaunchMode {
    let mut args = std::env::args_os();
    let _exe = args.next();
    let Some(first) = args.next() else {
        return LaunchMode::Browser { initial_path: None };
    };

    if first == "--view-npy" {
        if let Some(path) = args.next() {
            return LaunchMode::ViewNpy {
                path: PathBuf::from(path),
            };
        }
        return LaunchMode::Browser { initial_path: None };
    }

    if first == "--view-npz-entry" {
        if let (Some(npz_path), Some(entry)) = (args.next(), args.next()) {
            return LaunchMode::ViewNpzEntry {
                npz_path: PathBuf::from(npz_path),
                entry: entry.to_string_lossy().into_owned(),
            };
        }
        return LaunchMode::Browser { initial_path: None };
    }

    let path = PathBuf::from(first);
    let ext = path
        .extension()
        .and_then(|v| v.to_str())
        .map(|v| v.to_ascii_lowercase());
    if ext.as_deref() == Some("npy") {
        LaunchMode::ViewNpy { path }
    } else {
        LaunchMode::Browser {
            initial_path: Some(path),
        }
    }
}

struct NpzBrowserApp {
    selected_path: Option<PathBuf>,
    selected_kind: Option<SelectedKind>,
    entries: Vec<String>,
    status: Option<String>,
}

impl Default for NpzBrowserApp {
    fn default() -> Self {
        Self {
            selected_path: None,
            selected_kind: None,
            entries: Vec::new(),
            status: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SelectedKind {
    Npz,
    SingleNpy,
}

#[derive(Clone)]
struct LoadedNpy {
    dtype: String,
    shape: Vec<usize>,
    total_elements: usize,
    preview_values: Vec<String>,
    complex_values: Option<Vec<ComplexCell>>,
    field_names: Option<Vec<String>>,
}

#[derive(Clone)]
struct ComplexCell {
    re: String,
    im: String,
}

#[derive(Clone)]
struct NpyWindowUiState {
    search_query: String,
    search_requested: bool,
    search_jump_target: Option<(usize, usize)>,
    search_jump_frames: u8,
    search_status: Option<String>,
    calc_expr: String,
    calc_name: String,
    trig_in_degrees: bool,
    plot_kind: PlotKind,
    plot_selected_cols: Vec<usize>,
    hist_bins: usize,
    derived_columns: Vec<DerivedColumn>,
    calc_status: Option<String>,
}

impl Default for NpyWindowUiState {
    fn default() -> Self {
        Self {
            search_query: String::new(),
            search_requested: false,
            search_jump_target: None,
            search_jump_frames: 0,
            search_status: None,
            calc_expr: String::new(),
            calc_name: String::new(),
            trig_in_degrees: false,
            plot_kind: PlotKind::Linear,
            plot_selected_cols: Vec::new(),
            hist_bins: 30,
            derived_columns: Vec::new(),
            calc_status: None,
        }
    }
}

#[derive(Clone)]
struct DerivedColumn {
    name: String,
    values: Vec<String>,
}

#[derive(Clone)]
enum Expr {
    Number(f64),
    Column(usize),
    UnaryMinus(Box<Expr>),
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Func {
        func: FuncOp,
        arg: Box<Expr>,
    },
}

#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Atan2,
}

#[derive(Clone, Copy)]
enum FuncOp {
    Sin,
    Cos,
    Tan,
    Exp,
    Abs,
    Sqrt,
    Asin,
    Acos,
    Atan,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlotKind {
    Linear,
    Scatter,
    Hist,
}

impl PlotKind {
    fn label(self) -> &'static str {
        match self {
            PlotKind::Linear => "linear",
            PlotKind::Scatter => "scatter",
            PlotKind::Hist => "hist",
        }
    }
}

impl LoadedNpy {
    fn from_array<T: Display>(dtype: impl Into<String>, array: ndarray::ArrayD<T>) -> Self {
        Self {
            dtype: dtype.into(),
            shape: array.shape().to_vec(),
            total_elements: array.len(),
            preview_values: array.iter().map(ToString::to_string).collect(),
            complex_values: None,
            field_names: None,
        }
    }

    fn from_complex_array<T: Display>(
        dtype: impl Into<String>,
        array: ndarray::ArrayD<num_complex::Complex<T>>,
    ) -> Self {
        Self {
            dtype: dtype.into(),
            shape: array.shape().to_vec(),
            total_elements: array.len(),
            preview_values: Vec::new(),
            complex_values: Some(
                array
                    .iter()
                    .map(|v| ComplexCell {
                        re: v.re.to_string(),
                        im: v.im.to_string(),
                    })
                    .collect(),
            ),
            field_names: None,
        }
    }
}

impl NpzBrowserApp {
    fn from_initial_path(initial_path: Option<PathBuf>) -> Self {
        let mut app = Self::default();
        if let Some(path) = initial_path {
            app.open_path(path);
        }
        app
    }

    fn open_path(&mut self, path: PathBuf) {
        if !path.exists() {
            self.status = Some(format!("Failed: file does not exist: {}", path.display()));
            self.entries.clear();
            self.selected_path = None;
            self.selected_kind = None;
            return;
        }

        let ext = path
            .extension()
            .and_then(|v| v.to_str())
            .map(|v| v.to_ascii_lowercase());

        match ext.as_deref() {
            Some("npz") => match list_npy_entries(&path) {
                Ok(entries) => {
                    self.status = Some(format!("Loaded {} entries", entries.len()));
                    self.entries = entries;
                    self.selected_path = Some(path);
                    self.selected_kind = Some(SelectedKind::Npz);
                }
                Err(err) => {
                    self.status = Some(format!("Failed: {}", err));
                    self.entries.clear();
                    self.selected_path = None;
                    self.selected_kind = None;
                }
            },
            Some("npy") => {
                let name = path
                    .file_name()
                    .and_then(|v| v.to_str())
                    .map(ToString::to_string)
                    .unwrap_or_else(|| path.display().to_string());
                self.status = Some("Loaded single .npy file".to_string());
                self.entries = vec![name];
                self.selected_path = Some(path);
                self.selected_kind = Some(SelectedKind::SingleNpy);
            }
            _ => {
                self.status = Some("Failed: select a .npz or .npy file".to_string());
                self.entries.clear();
                self.selected_path = None;
                self.selected_kind = None;
            }
        }
    }

    fn open_npz_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("NumPy data", &["npz", "npy"])
            .pick_file()
        {
            self.open_path(path);
        }
    }

    fn open_npy_window(&mut self, name: &str) {
        let Some(path) = self.selected_path.as_ref() else {
            self.status = Some("No file is selected".to_string());
            return;
        };
        let Some(kind) = self.selected_kind else {
            self.status = Some("No file is selected".to_string());
            return;
        };

        match spawn_viewer_process(path, kind, name) {
            Ok(()) => {
                self.status = Some(format!("Opened {}", name));
            }
            Err(err) => {
                self.status = Some(format!("Failed to open {}: {}", name, err));
            }
        }
    }
}

impl eframe::App for NpzBrowserApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("npyz-viewer");
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Open .npz/.npy").clicked() {
                    self.open_npz_dialog();
                }

                if let Some(path) = &self.selected_path {
                    ui.label(path.display().to_string());
                } else {
                    ui.label("No file selected");
                }
            });

            if let Some(status) = &self.status {
                ui.separator();
                ui.label(status);
            }

            ui.separator();
            ui.label("Entries (.npy):");

            egui::ScrollArea::vertical().show(ui, |ui| {
                let names = self.entries.clone();
                for name in names {
                    if ui.button(&name).clicked() {
                        self.open_npy_window(&name);
                    }
                }
            });
        });
    }
}

struct NpyViewerApp {
    title: String,
    key: String,
    result: Result<LoadedNpy, String>,
    ui_state: Arc<Mutex<NpyWindowUiState>>,
}

impl NpyViewerApp {
    fn from_npy_path(path: PathBuf) -> Self {
        let key = path.display().to_string();
        let title = path
            .file_name()
            .and_then(|v| v.to_str())
            .map(|name| format!("NPY: {}", name))
            .unwrap_or_else(|| format!("NPY: {}", path.display()));
        Self {
            title,
            key,
            result: load_npy_from_file(&path),
            ui_state: Arc::new(Mutex::new(NpyWindowUiState::default())),
        }
    }

    fn from_npz_entry(npz_path: PathBuf, entry: String) -> Self {
        let title = format!("NPY: {}", entry);
        let key = format!("{}::{}", npz_path.display(), entry);
        Self {
            title,
            key,
            result: load_npy_from_npz(&npz_path, &entry),
            ui_state: Arc::new(Mutex::new(NpyWindowUiState::default())),
        }
    }
}

impl eframe::App for NpyViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if FORCE_EXIT_ON_VIEWPORT_X && ctx.input(|i| i.viewport().close_requested()) {
            std::process::exit(0);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let should_close =
                draw_npy_window_content(ui, &self.title, &self.key, &self.result, &self.ui_state);
            if should_close {
                std::process::exit(0);
            }
        });
    }
}

fn spawn_viewer_process(path: &Path, kind: SelectedKind, name: &str) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|err| format!("current_exe failed: {}", err))?;
    let mut cmd = std::process::Command::new(exe);
    match kind {
        SelectedKind::Npz => {
            cmd.arg("--view-npz-entry").arg(path).arg(name);
        }
        SelectedKind::SingleNpy => {
            cmd.arg("--view-npy").arg(path);
        }
    }
    cmd.spawn()
        .map(|_| ())
        .map_err(|err| format!("spawn failed: {}", err))
}

fn list_npy_entries(path: &Path) -> Result<Vec<String>, String> {
    let file =
        File::open(path).map_err(|err| format!("Cannot open {}: {}", path.display(), err))?;
    let mut reader = NpzReader::new(BufReader::new(file)).map_err(|err| {
        format!(
            "Invalid npz {}: {}. Hint: .npy is not a zip archive.",
            path.display(),
            err
        )
    })?;
    let mut names = reader
        .names()
        .map_err(|err| format!("Cannot read npz entries: {}", err))?;
    names.sort();
    Ok(names)
}

fn load_npy_from_npz(npz_path: &Path, npy_name: &str) -> Result<LoadedNpy, String> {
    let file = File::open(npz_path)
        .map_err(|err| format!("Cannot open {}: {}", npz_path.display(), err))?;
    let mut npz = NpzReader::new(BufReader::new(file))
        .map_err(|err| format!("Invalid npz {}: {}", npz_path.display(), err))?;

    macro_rules! try_load {
        ($ty:ty, $label:literal) => {
            if let Ok(array) = npz.by_name::<ndarray::OwnedRepr<$ty>, IxDyn>(npy_name) {
                return Ok(LoadedNpy::from_array($label, array));
            }
        };
    }

    try_load!(f64, "f64");
    try_load!(f32, "f32");
    try_load!(i64, "i64");
    try_load!(i32, "i32");
    try_load!(i16, "i16");
    try_load!(i8, "i8");
    try_load!(u64, "u64");
    try_load!(u32, "u32");
    try_load!(u16, "u16");
    try_load!(u8, "u8");
    try_load!(bool, "bool");
    if let Ok(array) = npz.by_name::<ndarray::OwnedRepr<num_complex::Complex32>, IxDyn>(npy_name) {
        return Ok(LoadedNpy::from_complex_array("complex64", array));
    }
    if let Ok(array) = npz.by_name::<ndarray::OwnedRepr<num_complex::Complex64>, IxDyn>(npy_name) {
        return Ok(LoadedNpy::from_complex_array("complex128", array));
    }

    Err(format!(
        "Unsupported dtype for '{}'. Supported: bool, i8/i16/i32/i64, u8/u16/u32/u64, f32/f64, complex64/complex128",
        npy_name
    ))
}

fn load_npy_from_file(npy_path: &Path) -> Result<LoadedNpy, String> {
    if let Some(loaded) = load_string_npy_from_file(npy_path)? {
        return Ok(loaded);
    }

    macro_rules! try_load {
        ($ty:ty, $label:literal) => {
            if let Ok(array) = read_npy::<_, ndarray::ArrayD<$ty>>(npy_path) {
                return Ok(LoadedNpy::from_array($label, array));
            }
        };
    }

    try_load!(f64, "f64");
    try_load!(f32, "f32");
    try_load!(i64, "i64");
    try_load!(i32, "i32");
    try_load!(i16, "i16");
    try_load!(i8, "i8");
    try_load!(u64, "u64");
    try_load!(u32, "u32");
    try_load!(u16, "u16");
    try_load!(u8, "u8");
    try_load!(bool, "bool");
    if let Ok(array) = read_npy::<_, ndarray::ArrayD<num_complex::Complex32>>(npy_path) {
        return Ok(LoadedNpy::from_complex_array("complex64", array));
    }
    if let Ok(array) = read_npy::<_, ndarray::ArrayD<num_complex::Complex64>>(npy_path) {
        return Ok(LoadedNpy::from_complex_array("complex128", array));
    }

    Err(format!(
        "Unsupported dtype for '{}'. Supported: bool, i8/i16/i32/i64, u8/u16/u32/u64, f32/f64, complex64/complex128",
        npy_path.display()
    ))
}

fn load_string_npy_from_file(npy_path: &Path) -> Result<Option<LoadedNpy>, String> {
    let bytes = std::fs::read(npy_path)
        .map_err(|err| format!("Cannot open {}: {}", npy_path.display(), err))?;
    load_string_npy_from_bytes(&bytes)
}

fn load_string_npy_from_bytes(bytes: &[u8]) -> Result<Option<LoadedNpy>, String> {
    let meta = parse_npy_metadata(bytes)?;
    let elem_count = element_count_from_shape(&meta.shape)?;
    let value_count = elem_count;

    if let Some(layout) = parse_real_imag_layout(&meta.descr)? {
        let values = decode_real_imag_values(bytes, meta.data_offset, value_count, &layout)?;
        return Ok(Some(LoadedNpy {
            dtype: meta.descr,
            shape: meta.shape,
            total_elements: elem_count,
            preview_values: Vec::new(),
            complex_values: Some(values),
            field_names: None,
        }));
    }

    if let Some(fields) = parse_structured_field_descriptors(&meta.descr)? {
        let (values, field_names, field_count) =
            decode_structured_values(bytes, meta.data_offset, value_count, &fields)?;
        let mut shape = meta.shape;
        if shape.is_empty() {
            shape.push(1);
        }
        shape.push(field_count);
        return Ok(Some(LoadedNpy {
            dtype: meta.descr,
            shape,
            total_elements: value_count
                .checked_mul(field_count)
                .ok_or_else(|| "dtype size overflow".to_string())?,
            preview_values: values,
            complex_values: None,
            field_names: Some(field_names),
        }));
    }

    let descriptor = parse_descriptor(&meta.descr)
        .ok_or_else(|| format!("Invalid dtype descriptor '{}'", meta.descr))?;

    match descriptor.kind {
        'U' => {
            let item_bytes = descriptor
                .size
                .checked_mul(4)
                .ok_or_else(|| "dtype size overflow".to_string())?;
            let values = decode_unicode_preview(
                bytes,
                meta.data_offset,
                value_count,
                item_bytes,
                descriptor.endian,
            )?;
            Ok(Some(LoadedNpy {
                dtype: meta.descr,
                shape: meta.shape,
                total_elements: elem_count,
                preview_values: values,
                complex_values: None,
                field_names: None,
            }))
        }
        'S' => {
            let item_bytes = descriptor.size;
            let values = decode_bytes_preview(bytes, meta.data_offset, value_count, item_bytes)?;
            Ok(Some(LoadedNpy {
                dtype: meta.descr,
                shape: meta.shape,
                total_elements: elem_count,
                preview_values: values,
                complex_values: None,
                field_names: None,
            }))
        }
        _ => Ok(None),
    }
}

#[derive(Clone, Debug)]
struct NpyMeta {
    descr: String,
    shape: Vec<usize>,
    data_offset: usize,
}

#[derive(Clone, Copy, Debug)]
struct ParsedDescriptor {
    endian: EndianMarker,
    kind: char,
    size: usize,
}

#[derive(Clone, Copy, Debug)]
struct RealImagLayout {
    record_bytes: usize,
    real_desc: ParsedDescriptor,
    real_offset: usize,
    imag_desc: ParsedDescriptor,
    imag_offset: usize,
}

#[derive(Clone, Copy, Debug)]
enum EndianMarker {
    Little,
    Big,
    Native,
    NotApplicable,
}

fn parse_real_imag_layout(descr: &str) -> Result<Option<RealImagLayout>, String> {
    let fields = match parse_structured_field_descriptors(descr)? {
        Some(fields) => fields,
        None => return Ok(None),
    };

    let mut record_bytes = 0usize;
    let mut real: Option<(ParsedDescriptor, usize)> = None;
    let mut imag: Option<(ParsedDescriptor, usize)> = None;

    for (name, field_descr) in fields {
        let parsed = parse_descriptor(&field_descr)
            .ok_or_else(|| format!("Invalid field dtype descriptor '{}'", field_descr))?;
        let field_bytes = descriptor_item_bytes(parsed)?;
        let offset = record_bytes;

        if name == "real" && real.is_none() {
            real = Some((parsed, offset));
        }
        if name == "imag" && imag.is_none() {
            imag = Some((parsed, offset));
        }

        record_bytes = record_bytes
            .checked_add(field_bytes)
            .ok_or_else(|| "dtype size overflow".to_string())?;
    }

    let (real_desc, real_offset) = match real {
        Some(v) => v,
        None => return Ok(None),
    };
    let (imag_desc, imag_offset) = match imag {
        Some(v) => v,
        None => return Ok(None),
    };

    Ok(Some(RealImagLayout {
        record_bytes,
        real_desc,
        real_offset,
        imag_desc,
        imag_offset,
    }))
}

fn parse_structured_field_descriptors(
    descr: &str,
) -> Result<Option<Vec<(String, String)>>, String> {
    let mut i = skip_ws(descr, 0);
    if descr[i..].chars().next() != Some('[') {
        return Ok(None);
    }
    i += 1;

    let mut fields = Vec::new();
    loop {
        i = skip_ws(descr, i);
        match descr[i..].chars().next() {
            Some(']') => break,
            Some('(') => {}
            Some(other) => {
                return Err(format!(
                    "Invalid structured dtype: expected '(' but found '{}'",
                    other
                ));
            }
            None => return Err("Invalid structured dtype: unexpected end".to_string()),
        }
        i += 1;

        i = skip_ws(descr, i);
        let (name, next_i) = parse_single_quoted(descr, i)?;
        i = skip_ws(descr, next_i);
        expect_char(descr, i, ',')?;
        i += 1;

        i = skip_ws(descr, i);
        let (field_descr, next_i) = parse_single_quoted(descr, i)?;
        i = next_i;

        loop {
            i = skip_ws(descr, i);
            match descr[i..].chars().next() {
                Some(')') => {
                    i += 1;
                    break;
                }
                Some(_) => i += 1,
                None => return Err("Invalid structured dtype: missing ')'".to_string()),
            }
        }

        fields.push((name, field_descr));
        i = skip_ws(descr, i);
        match descr[i..].chars().next() {
            Some(',') => {
                i += 1;
            }
            Some(']') => break,
            Some(other) => {
                return Err(format!(
                    "Invalid structured dtype: expected ',' or ']' but found '{}'",
                    other
                ));
            }
            None => return Err("Invalid structured dtype: unexpected end".to_string()),
        }
    }

    Ok(Some(fields))
}

fn parse_single_quoted(s: &str, i: usize) -> Result<(String, usize), String> {
    expect_char(s, i, '\'')?;
    let start = i + 1;
    let end_rel = s[start..]
        .find('\'')
        .ok_or_else(|| "Invalid structured dtype: missing quote".to_string())?;
    let end = start + end_rel;
    Ok((s[start..end].to_string(), end + 1))
}

fn expect_char(s: &str, i: usize, expected: char) -> Result<(), String> {
    let found = s[i..].chars().next();
    if found == Some(expected) {
        Ok(())
    } else {
        Err(format!(
            "Invalid structured dtype: expected '{}' but found '{}'",
            expected,
            found.unwrap_or('\0')
        ))
    }
}

fn skip_ws(s: &str, mut i: usize) -> usize {
    while let Some(ch) = s[i..].chars().next() {
        if ch.is_whitespace() {
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    i
}

fn descriptor_item_bytes(desc: ParsedDescriptor) -> Result<usize, String> {
    match desc.kind {
        'U' => desc
            .size
            .checked_mul(4)
            .ok_or_else(|| "dtype size overflow".to_string()),
        _ => Ok(desc.size),
    }
}

fn decode_real_imag_values(
    bytes: &[u8],
    data_offset: usize,
    value_count: usize,
    layout: &RealImagLayout,
) -> Result<Vec<ComplexCell>, String> {
    let needed = value_count
        .checked_mul(layout.record_bytes)
        .ok_or_else(|| "dtype size overflow".to_string())?;
    if bytes.len() < data_offset + needed {
        return Err("NPY data is shorter than expected".to_string());
    }

    let real_width = descriptor_item_bytes(layout.real_desc)?;
    let imag_width = descriptor_item_bytes(layout.imag_desc)?;

    let mut out = Vec::with_capacity(value_count);
    for i in 0..value_count {
        let row_start = data_offset + i * layout.record_bytes;
        let real_start = row_start + layout.real_offset;
        let imag_start = row_start + layout.imag_offset;
        let real_slice = &bytes[real_start..real_start + real_width];
        let imag_slice = &bytes[imag_start..imag_start + imag_width];

        out.push(ComplexCell {
            re: decode_scalar_to_string(real_slice, layout.real_desc)?,
            im: decode_scalar_to_string(imag_slice, layout.imag_desc)?,
        });
    }
    Ok(out)
}

fn decode_structured_values(
    bytes: &[u8],
    data_offset: usize,
    value_count: usize,
    fields: &[(String, String)],
) -> Result<(Vec<String>, Vec<String>, usize), String> {
    #[derive(Clone, Copy)]
    struct FieldInfo {
        desc: ParsedDescriptor,
        offset: usize,
        width: usize,
    }

    let mut infos = Vec::with_capacity(fields.len());
    let mut names = Vec::with_capacity(fields.len());
    let mut record_bytes = 0usize;

    for (name, field_descr) in fields {
        let desc = parse_descriptor(field_descr)
            .ok_or_else(|| format!("Invalid field dtype descriptor '{}'", field_descr))?;
        let width = descriptor_item_bytes(desc)?;
        let offset = record_bytes;
        record_bytes = record_bytes
            .checked_add(width)
            .ok_or_else(|| "dtype size overflow".to_string())?;
        infos.push(FieldInfo {
            desc,
            offset,
            width,
        });
        names.push(name.clone());
    }

    if infos.is_empty() {
        return Err("Invalid structured dtype: no fields".to_string());
    }

    let needed = value_count
        .checked_mul(record_bytes)
        .ok_or_else(|| "dtype size overflow".to_string())?;
    if bytes.len() < data_offset + needed {
        return Err("NPY data is shorter than expected".to_string());
    }

    let field_count = infos.len();
    let mut out = Vec::with_capacity(value_count.saturating_mul(field_count));
    for i in 0..value_count {
        let row_start = data_offset + i * record_bytes;
        for info in &infos {
            let start = row_start + info.offset;
            let end = start + info.width;
            out.push(decode_scalar_to_string(&bytes[start..end], info.desc)?);
        }
    }

    Ok((out, names, field_count))
}

fn decode_scalar_to_string(bytes: &[u8], desc: ParsedDescriptor) -> Result<String, String> {
    match (desc.kind, desc.size) {
        ('f', 4) => Ok(decode_f32(bytes, desc.endian)?.to_string()),
        ('f', 8) => Ok(decode_f64(bytes, desc.endian)?.to_string()),
        ('i', 1) => Ok((bytes.first().copied().unwrap_or(0) as i8).to_string()),
        ('i', 2) => Ok(decode_i16(bytes, desc.endian)?.to_string()),
        ('i', 4) => Ok(decode_i32(bytes, desc.endian)?.to_string()),
        ('i', 8) => Ok(decode_i64(bytes, desc.endian)?.to_string()),
        ('u', 1) => Ok(bytes.first().copied().unwrap_or(0).to_string()),
        ('u', 2) => Ok(decode_u16(bytes, desc.endian)?.to_string()),
        ('u', 4) => Ok(decode_u32(bytes, desc.endian)?.to_string()),
        ('u', 8) => Ok(decode_u64(bytes, desc.endian)?.to_string()),
        ('b', 1) => Ok((bytes.first().copied().unwrap_or(0) != 0).to_string()),
        ('S', _) => {
            let last_non_zero = bytes
                .iter()
                .rposition(|b| *b != 0)
                .map(|idx| idx + 1)
                .unwrap_or(0);
            Ok(String::from_utf8_lossy(&bytes[..last_non_zero]).to_string())
        }
        ('U', _) => decode_unicode_scalar(bytes, desc.endian),
        _ => Err(format!(
            "Unsupported structured field dtype '{}{}'",
            desc.kind, desc.size
        )),
    }
}

fn decode_unicode_scalar(bytes: &[u8], endian: EndianMarker) -> Result<String, String> {
    if bytes.len() % 4 != 0 {
        return Err("Unexpected field byte length for unicode scalar".to_string());
    }
    let mut s = String::new();
    for ch in bytes.chunks_exact(4) {
        let code = match endian {
            EndianMarker::Big => u32::from_be_bytes([ch[0], ch[1], ch[2], ch[3]]),
            _ => u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]),
        };
        if code == 0 {
            break;
        }
        if let Some(c) = char::from_u32(code) {
            s.push(c);
        } else {
            s.push('\u{FFFD}');
        }
    }
    Ok(s)
}

fn decode_f32(bytes: &[u8], endian: EndianMarker) -> Result<f32, String> {
    let arr: [u8; 4] = bytes
        .try_into()
        .map_err(|_| "Unexpected field byte length for f32".to_string())?;
    Ok(match endian {
        EndianMarker::Big => f32::from_bits(u32::from_be_bytes(arr)),
        _ => f32::from_bits(u32::from_le_bytes(arr)),
    })
}

fn decode_f64(bytes: &[u8], endian: EndianMarker) -> Result<f64, String> {
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| "Unexpected field byte length for f64".to_string())?;
    Ok(match endian {
        EndianMarker::Big => f64::from_bits(u64::from_be_bytes(arr)),
        _ => f64::from_bits(u64::from_le_bytes(arr)),
    })
}

fn decode_i16(bytes: &[u8], endian: EndianMarker) -> Result<i16, String> {
    let arr: [u8; 2] = bytes
        .try_into()
        .map_err(|_| "Unexpected field byte length for i16".to_string())?;
    Ok(match endian {
        EndianMarker::Big => i16::from_be_bytes(arr),
        _ => i16::from_le_bytes(arr),
    })
}

fn decode_i32(bytes: &[u8], endian: EndianMarker) -> Result<i32, String> {
    let arr: [u8; 4] = bytes
        .try_into()
        .map_err(|_| "Unexpected field byte length for i32".to_string())?;
    Ok(match endian {
        EndianMarker::Big => i32::from_be_bytes(arr),
        _ => i32::from_le_bytes(arr),
    })
}

fn decode_i64(bytes: &[u8], endian: EndianMarker) -> Result<i64, String> {
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| "Unexpected field byte length for i64".to_string())?;
    Ok(match endian {
        EndianMarker::Big => i64::from_be_bytes(arr),
        _ => i64::from_le_bytes(arr),
    })
}

fn decode_u16(bytes: &[u8], endian: EndianMarker) -> Result<u16, String> {
    let arr: [u8; 2] = bytes
        .try_into()
        .map_err(|_| "Unexpected field byte length for u16".to_string())?;
    Ok(match endian {
        EndianMarker::Big => u16::from_be_bytes(arr),
        _ => u16::from_le_bytes(arr),
    })
}

fn decode_u32(bytes: &[u8], endian: EndianMarker) -> Result<u32, String> {
    let arr: [u8; 4] = bytes
        .try_into()
        .map_err(|_| "Unexpected field byte length for u32".to_string())?;
    Ok(match endian {
        EndianMarker::Big => u32::from_be_bytes(arr),
        _ => u32::from_le_bytes(arr),
    })
}

fn decode_u64(bytes: &[u8], endian: EndianMarker) -> Result<u64, String> {
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| "Unexpected field byte length for u64".to_string())?;
    Ok(match endian {
        EndianMarker::Big => u64::from_be_bytes(arr),
        _ => u64::from_le_bytes(arr),
    })
}

fn parse_npy_metadata(bytes: &[u8]) -> Result<NpyMeta, String> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err("Invalid NPY header: bad magic".to_string());
    }

    let major = bytes[6];
    let header_len_size = match major {
        1 => 2usize,
        2 | 3 => 4usize,
        _ => {
            return Err(format!(
                "Unsupported NPY version: {}.{}",
                bytes[6], bytes[7]
            ));
        }
    };

    let header_len_start = 8usize;
    let header_len_end = header_len_start + header_len_size;
    if bytes.len() < header_len_end {
        return Err("Invalid NPY header: truncated length field".to_string());
    }

    let header_len = if header_len_size == 2 {
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize
    } else {
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
    };

    let header_start = header_len_end;
    let data_offset = header_start
        .checked_add(header_len)
        .ok_or_else(|| "Invalid NPY header: offset overflow".to_string())?;
    if bytes.len() < data_offset {
        return Err("Invalid NPY header: truncated metadata".to_string());
    }

    let header = std::str::from_utf8(&bytes[header_start..data_offset])
        .map_err(|err| format!("Invalid NPY header UTF-8: {}", err))?;
    let descr = extract_descr_field(header)
        .ok_or_else(|| "Invalid NPY header: missing 'descr'".to_string())?;
    let shape_raw = extract_parenthesized_field(header, "shape")
        .ok_or_else(|| "Invalid NPY header: missing 'shape'".to_string())?;
    let shape = parse_shape_tuple(&shape_raw)?;

    Ok(NpyMeta {
        descr,
        shape,
        data_offset,
    })
}

fn extract_descr_field(header: &str) -> Option<String> {
    let key_pat = "'descr':";
    let pos = header.find(key_pat)?;
    let mut i = pos + key_pat.len();
    i = skip_ws(header, i);

    let first = header[i..].chars().next()?;
    if first == '\'' {
        let (value, _) = parse_single_quoted(header, i).ok()?;
        return Some(value);
    }
    if first == '[' {
        let mut depth = 0usize;
        let mut in_quote = false;
        let start = i;
        for (off, ch) in header[i..].char_indices() {
            if ch == '\'' {
                in_quote = !in_quote;
                continue;
            }
            if in_quote {
                continue;
            }
            if ch == '[' {
                depth += 1;
            } else if ch == ']' {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = i + off + ch.len_utf8();
                    return Some(header[start..end].to_string());
                }
            }
        }
        return None;
    }

    let rest = &header[i..];
    let end = rest.find(',').unwrap_or(rest.len());
    Some(rest[..end].trim().to_string())
}

fn extract_parenthesized_field(header: &str, key: &str) -> Option<String> {
    let key_pat = format!("'{}':", key);
    let pos = header.find(&key_pat)?;
    let rest = &header[pos + key_pat.len()..];
    let open = rest.find('(')?;
    let after_open = &rest[open + 1..];
    let close = after_open.find(')')?;
    Some(after_open[..close].to_string())
}

fn parse_shape_tuple(raw: &str) -> Result<Vec<usize>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    let mut shape = Vec::new();
    for part in trimmed.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        let value = t
            .parse::<usize>()
            .map_err(|_| format!("Invalid shape component '{}'", t))?;
        shape.push(value);
    }
    Ok(shape)
}

fn parse_descriptor(descr: &str) -> Option<ParsedDescriptor> {
    let mut chars = descr.chars();
    let first = chars.next()?;
    let (endian, kind) = match first {
        '<' => (EndianMarker::Little, chars.next()?),
        '>' => (EndianMarker::Big, chars.next()?),
        '=' => (EndianMarker::Native, chars.next()?),
        '|' => (EndianMarker::NotApplicable, chars.next()?),
        _ => (EndianMarker::Native, first),
    };
    let size_str: String = chars.collect();
    let size = size_str.parse::<usize>().ok()?;
    Some(ParsedDescriptor { endian, kind, size })
}

fn element_count_from_shape(shape: &[usize]) -> Result<usize, String> {
    if shape.is_empty() {
        return Ok(1);
    }
    let mut total = 1usize;
    for &d in shape {
        total = total
            .checked_mul(d)
            .ok_or_else(|| "shape product overflow".to_string())?;
    }
    Ok(total)
}

fn decode_unicode_preview(
    bytes: &[u8],
    data_offset: usize,
    value_count: usize,
    item_bytes: usize,
    endian: EndianMarker,
) -> Result<Vec<String>, String> {
    let needed = value_count
        .checked_mul(item_bytes)
        .ok_or_else(|| "dtype size overflow".to_string())?;
    if bytes.len() < data_offset + needed {
        return Err("NPY data is shorter than expected".to_string());
    }

    let mut out = Vec::with_capacity(value_count);
    for i in 0..value_count {
        let start = data_offset + i * item_bytes;
        let end = start + item_bytes;
        let cell = &bytes[start..end];
        let mut s = String::new();
        for ch in cell.chunks_exact(4) {
            let code = match endian {
                EndianMarker::Little | EndianMarker::Native | EndianMarker::NotApplicable => {
                    u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]])
                }
                EndianMarker::Big => u32::from_be_bytes([ch[0], ch[1], ch[2], ch[3]]),
            };
            if code == 0 {
                break;
            }
            s.push(std::char::from_u32(code).unwrap_or('\u{FFFD}'));
        }
        out.push(s);
    }
    Ok(out)
}

fn decode_bytes_preview(
    bytes: &[u8],
    data_offset: usize,
    value_count: usize,
    item_bytes: usize,
) -> Result<Vec<String>, String> {
    let needed = value_count
        .checked_mul(item_bytes)
        .ok_or_else(|| "dtype size overflow".to_string())?;
    if bytes.len() < data_offset + needed {
        return Err("NPY data is shorter than expected".to_string());
    }

    let mut out = Vec::with_capacity(value_count);
    for i in 0..value_count {
        let start = data_offset + i * item_bytes;
        let end = start + item_bytes;
        let cell = &bytes[start..end];
        let last_non_zero = cell
            .iter()
            .rposition(|b| *b != 0)
            .map(|idx| idx + 1)
            .unwrap_or(0);
        out.push(String::from_utf8_lossy(&cell[..last_non_zero]).to_string());
    }
    Ok(out)
}

fn draw_npy_window_content(
    ui: &mut egui::Ui,
    title: &str,
    key: &str,
    result: &Result<LoadedNpy, String>,
    ui_state: &Arc<Mutex<NpyWindowUiState>>,
) -> bool {
    let mut close_requested = false;
    ui.horizontal(|ui| {
        if ui.button("Close").clicked() {
            close_requested = true;
        }
        ui.label(title);
    });
    ui.separator();
    if close_requested {
        return true;
    }

    match result {
        Ok(loaded) => {
            ui.label(format!("dtype: {}", loaded.dtype));
            ui.label(format!("shape: {:?}", loaded.shape));
            ui.label(format!("total elements: {}", loaded.total_elements));
            ui.separator();

            let Some(layout) = compute_table_layout(loaded) else {
                ui.label("No values");
                return close_requested;
            };

            let Ok(mut state) = ui_state.lock() else {
                ui.colored_label(egui::Color32::RED, "UI state lock failed");
                return close_requested;
            };

            ui.group(|ui| {
                ui.label("Parameters");
                draw_search_controls(ui, &mut state);
                ui.horizontal_top(|ui| {
                    let total_width = ui.available_width().max(2.0);
                    let left_width = (total_width * 0.5).max(1.0);
                    let right_width = (total_width - left_width).max(1.0);

                    ui.vertical(|ui| {
                        ui.set_min_width(left_width);
                        ui.set_max_width(left_width);
                        draw_calc_controls(ui, key, loaded, layout, &mut state);
                    });

                    ui.vertical(|ui| {
                        ui.set_min_width(right_width);
                        ui.set_max_width(right_width);
                        draw_plot_controls(ui, key, loaded, layout, &mut state);
                    });
                });
                if let Some(status) = &state.calc_status {
                    let is_error = status.starts_with("Calc error:");
                    let color = if is_error {
                        egui::Color32::from_rgb(230, 120, 120)
                    } else {
                        egui::Color32::from_rgb(140, 220, 140)
                    };
                    ui.colored_label(color, status);
                }
                if let Some(status) = &state.search_status {
                    ui.label(status);
                }
            });
            ui.separator();

            ui.horizontal_top(|ui| {
                let total_width = ui.available_width().max(2.0);
                let left_width = (total_width * 0.6).max(1.0); // 3:2
                let right_width = (total_width - left_width).max(1.0); // 3:2

                ui.vertical(|ui| {
                    ui.set_min_width(left_width);
                    ui.set_max_width(left_width);
                    ui.group(|ui| {
                        ui.label("Data");
                        ui.separator();
                        draw_calc_like_preview(ui, key, loaded, layout, &mut state);
                    });
                });

                ui.vertical(|ui| {
                    ui.set_min_width(right_width);
                    ui.set_max_width(right_width);
                    ui.group(|ui| {
                        ui.label("Plot");
                        ui.separator();
                        draw_plot_panel(ui, loaded, layout, &state);
                    });
                });
            });
        }
        Err(err) => {
            ui.colored_label(egui::Color32::RED, err);
        }
    }

    close_requested
}

#[derive(Clone, Copy)]
struct TableLayout {
    rows: usize,
    base_cols: usize,
    display_base_cols: usize,
    is_complex: bool,
}

fn compute_table_layout(loaded: &LoadedNpy) -> Option<TableLayout> {
    if let Some(complex_values) = &loaded.complex_values {
        let value_len = complex_values.len();
        if value_len == 0 {
            return None;
        }
        let base_cols = preview_table_columns(&loaded.shape, value_len);
        let rows = value_len.div_ceil(base_cols);
        return Some(TableLayout {
            rows,
            base_cols,
            display_base_cols: base_cols * 2,
            is_complex: true,
        });
    }

    let value_len = loaded.preview_values.len();
    if value_len == 0 {
        return None;
    }
    let base_cols = preview_table_columns(&loaded.shape, value_len);
    let rows = value_len.div_ceil(base_cols);
    Some(TableLayout {
        rows,
        base_cols,
        display_base_cols: base_cols,
        is_complex: false,
    })
}

fn draw_search_controls(ui: &mut egui::Ui, state: &mut NpyWindowUiState) {
    ui.horizontal(|ui| {
        ui.label("Search:");
        let resp = ui.text_edit_singleline(&mut state.search_query);
        let enter_pressed = resp.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if enter_pressed {
            state.search_requested = true;
        }
        if ui.button("Find").clicked() {
            state.search_requested = true;
        }
        if ui.button("Clear").clicked() {
            state.search_query.clear();
            state.search_requested = false;
            state.search_jump_target = None;
            state.search_jump_frames = 0;
            state.search_status = None;
        }
    });
}

fn draw_calc_controls(
    ui: &mut egui::Ui,
    _key: &str,
    loaded: &LoadedNpy,
    layout: TableLayout,
    state: &mut NpyWindowUiState,
) {
    let column_labels = display_column_labels(loaded, layout, &state.derived_columns);
    if column_labels.is_empty() {
        return;
    }

    ui.group(|ui| {
        ui.label("Calc (expression)");
        ui.horizontal_wrapped(|ui| {
            ui.label("Columns:");
            for (i, name) in column_labels.iter().enumerate() {
                ui.monospace(format!("${}={}", i + 1, name));
            }
        });

        ui.horizontal(|ui| {
            ui.label("Expr:");
            ui.add(
                egui::TextEdit::singleline(&mut state.calc_expr)
                    .hint_text("sin($1), atan2($2,$1), sqrt(abs($1)), pow($1,2), abs($3)"),
            );
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut state.trig_in_degrees, "Trig uses degrees");
            ui.label("Output name:");
            ui.text_edit_singleline(&mut state.calc_name);

            if ui.button("Add Column").clicked() {
                match apply_calc_column(loaded, layout, state) {
                    Ok(name) => state.calc_status = Some(format!("Added column '{}'", name)),
                    Err(err) => state.calc_status = Some(format!("Calc error: {}", err)),
                }
            }
            if !state.derived_columns.is_empty() && ui.button("Clear Calc Columns").clicked() {
                state.derived_columns.clear();
                state.calc_status = Some("Cleared calculated columns".to_string());
            }
        });
    });
}

fn draw_plot_controls(
    ui: &mut egui::Ui,
    key: &str,
    loaded: &LoadedNpy,
    layout: TableLayout,
    state: &mut NpyWindowUiState,
) {
    let column_labels = display_column_labels(loaded, layout, &state.derived_columns);
    state
        .plot_selected_cols
        .retain(|idx| *idx < column_labels.len());
    if state.plot_selected_cols.len() > 2 {
        state.plot_selected_cols.truncate(2);
    }
    if state.plot_selected_cols.is_empty() && !column_labels.is_empty() {
        state.plot_selected_cols.push(0);
    }

    ui.group(|ui| {
        ui.label("Plot");
        ui.horizontal(|ui| {
            ui.label("Type:");
            egui::ComboBox::from_id_salt(format!("plot_kind_{key}"))
                .selected_text(state.plot_kind.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.plot_kind,
                        PlotKind::Linear,
                        PlotKind::Linear.label(),
                    );
                    ui.selectable_value(
                        &mut state.plot_kind,
                        PlotKind::Scatter,
                        PlotKind::Scatter.label(),
                    );
                    ui.selectable_value(
                        &mut state.plot_kind,
                        PlotKind::Hist,
                        PlotKind::Hist.label(),
                    );
                });

            if state.plot_kind == PlotKind::Hist {
                ui.label("Bins:");
                ui.add(egui::Slider::new(&mut state.hist_bins, 2..=200));
            }
        });

        ui.label("Select 1 or 2 columns for plot:");
        ui.horizontal_wrapped(|ui| {
            for (idx, name) in column_labels.iter().enumerate() {
                let selected = state.plot_selected_cols.contains(&idx);
                let can_enable = selected || state.plot_selected_cols.len() < 2;
                ui.add_enabled_ui(can_enable, |ui| {
                    if ui
                        .selectable_label(selected, format!("${} {}", idx + 1, name))
                        .clicked()
                    {
                        toggle_plot_column(&mut state.plot_selected_cols, idx);
                    }
                });
            }
        });

        let chosen = state
            .plot_selected_cols
            .iter()
            .map(|idx| format!("${}", idx + 1))
            .collect::<Vec<_>>()
            .join(", ");
        ui.label(format!("Selected: {}", chosen));
    });
}

fn toggle_plot_column(selected: &mut Vec<usize>, idx: usize) {
    if let Some(pos) = selected.iter().position(|x| *x == idx) {
        selected.remove(pos);
    } else if selected.len() < 2 {
        selected.push(idx);
    }
}

fn draw_plot_panel(
    ui: &mut egui::Ui,
    loaded: &LoadedNpy,
    layout: TableLayout,
    state: &NpyWindowUiState,
) {
    let selected = &state.plot_selected_cols;
    if selected.is_empty() {
        ui.label("Plot: choose at least 1 column.");
        return;
    }

    match state.plot_kind {
        PlotKind::Linear | PlotKind::Scatter => {
            let points = if selected.len() == 1 {
                collect_plot_points_one_col(loaded, layout, &state.derived_columns, selected[0])
            } else {
                collect_plot_points_two_cols(
                    loaded,
                    layout,
                    &state.derived_columns,
                    selected[0],
                    selected[1],
                )
            };

            let title = if selected.len() == 1 {
                format!(
                    "{} plot: x=index, y=${}",
                    state.plot_kind.label(),
                    selected[0] + 1
                )
            } else {
                format!(
                    "{} plot: x=${}, y=${}",
                    state.plot_kind.label(),
                    selected[0] + 1,
                    selected[1] + 1
                )
            };
            draw_xy_plot(ui, &title, &points, state.plot_kind);
        }
        PlotKind::Hist => {
            let target_col = if selected.len() == 1 {
                selected[0]
            } else {
                selected[1]
            };
            let values = collect_hist_values(loaded, layout, &state.derived_columns, target_col);
            let title = format!("hist: column ${}", target_col + 1);
            draw_hist_plot(ui, &title, &values, state.hist_bins);
        }
    }
}

fn collect_plot_points_one_col(
    loaded: &LoadedNpy,
    layout: TableLayout,
    derived: &[DerivedColumn],
    y_col: usize,
) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(layout.rows);
    for row in 0..layout.rows {
        if let Some(y) = parse_numeric_cell(loaded, layout, derived, row, y_col) {
            if y.is_finite() {
                out.push([row as f64, y]);
            }
        }
    }
    out
}

fn collect_plot_points_two_cols(
    loaded: &LoadedNpy,
    layout: TableLayout,
    derived: &[DerivedColumn],
    x_col: usize,
    y_col: usize,
) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(layout.rows);
    for row in 0..layout.rows {
        let x = parse_numeric_cell(loaded, layout, derived, row, x_col);
        let y = parse_numeric_cell(loaded, layout, derived, row, y_col);
        if let (Some(xv), Some(yv)) = (x, y) {
            if xv.is_finite() && yv.is_finite() {
                out.push([xv, yv]);
            }
        }
    }
    out
}

fn collect_hist_values(
    loaded: &LoadedNpy,
    layout: TableLayout,
    derived: &[DerivedColumn],
    col: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(layout.rows);
    for row in 0..layout.rows {
        if let Some(v) = parse_numeric_cell(loaded, layout, derived, row, col) {
            if v.is_finite() {
                out.push(v);
            }
        }
    }
    out
}

fn draw_xy_plot(ui: &mut egui::Ui, title: &str, points: &[[f64; 2]], kind: PlotKind) {
    let size = plot_canvas_size(ui);
    let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
    let rect = resp.rect;
    let plot_rect = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 52.0, rect.top() + 20.0),
        egui::pos2(rect.right() - 14.0, rect.bottom() - 30.0),
    );

    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(18));
    painter.rect_filled(plot_rect, 2.0, egui::Color32::from_gray(10));
    painter.line_segment(
        [
            egui::pos2(plot_rect.left(), plot_rect.bottom()),
            plot_rect.left_top(),
        ],
        egui::Stroke::new(1.0, egui::Color32::GRAY),
    );
    painter.line_segment(
        [
            egui::pos2(plot_rect.left(), plot_rect.bottom()),
            plot_rect.right_bottom(),
        ],
        egui::Stroke::new(1.0, egui::Color32::GRAY),
    );
    painter.text(
        egui::pos2(rect.left() + 8.0, rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        title,
        egui::FontId::proportional(13.0),
        egui::Color32::LIGHT_GRAY,
    );

    if points.is_empty() {
        painter.text(
            plot_rect.center(),
            egui::Align2::CENTER_CENTER,
            "No numeric points",
            egui::FontId::proportional(13.0),
            egui::Color32::LIGHT_GRAY,
        );
        return;
    }

    let (mut x_min, mut x_max) = (points[0][0], points[0][0]);
    let (mut y_min, mut y_max) = (points[0][1], points[0][1]);
    for p in points {
        x_min = x_min.min(p[0]);
        x_max = x_max.max(p[0]);
        y_min = y_min.min(p[1]);
        y_max = y_max.max(p[1]);
    }
    let (x_min, x_max) = normalize_range(x_min, x_max);
    let (y_min, y_max) = normalize_range(y_min, y_max);

    let to_screen = |x: f64, y: f64| -> egui::Pos2 {
        let tx = ((x - x_min) / (x_max - x_min)).clamp(0.0, 1.0) as f32;
        let ty = ((y - y_min) / (y_max - y_min)).clamp(0.0, 1.0) as f32;
        egui::pos2(
            plot_rect.left() + tx * plot_rect.width(),
            plot_rect.bottom() - ty * plot_rect.height(),
        )
    };

    if kind == PlotKind::Linear {
        let pts = points
            .iter()
            .map(|p| to_screen(p[0], p[1]))
            .collect::<Vec<_>>();
        if pts.len() >= 2 {
            painter.add(egui::Shape::line(
                pts,
                egui::Stroke::new(1.6, egui::Color32::LIGHT_BLUE),
            ));
        }
    }

    for p in points {
        let pos = to_screen(p[0], p[1]);
        let r = if kind == PlotKind::Scatter { 2.6 } else { 1.8 };
        painter.circle_filled(pos, r, egui::Color32::from_rgb(120, 220, 170));
    }

    painter.text(
        egui::pos2(plot_rect.left(), plot_rect.bottom() + 4.0),
        egui::Align2::LEFT_TOP,
        format!("x:[{} .. {}]", format_number(x_min), format_number(x_max)),
        egui::FontId::proportional(11.0),
        egui::Color32::GRAY,
    );
    painter.text(
        egui::pos2(plot_rect.right(), plot_rect.top() - 2.0),
        egui::Align2::RIGHT_BOTTOM,
        format!("y:[{} .. {}]", format_number(y_min), format_number(y_max)),
        egui::FontId::proportional(11.0),
        egui::Color32::GRAY,
    );
}

fn draw_hist_plot(ui: &mut egui::Ui, title: &str, values: &[f64], bins: usize) {
    let size = plot_canvas_size(ui);
    let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
    let rect = resp.rect;
    let plot_rect = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 52.0, rect.top() + 20.0),
        egui::pos2(rect.right() - 14.0, rect.bottom() - 30.0),
    );

    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(18));
    painter.rect_filled(plot_rect, 2.0, egui::Color32::from_gray(10));
    painter.line_segment(
        [
            egui::pos2(plot_rect.left(), plot_rect.bottom()),
            plot_rect.left_top(),
        ],
        egui::Stroke::new(1.0, egui::Color32::GRAY),
    );
    painter.line_segment(
        [
            egui::pos2(plot_rect.left(), plot_rect.bottom()),
            plot_rect.right_bottom(),
        ],
        egui::Stroke::new(1.0, egui::Color32::GRAY),
    );
    painter.text(
        egui::pos2(rect.left() + 8.0, rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        title,
        egui::FontId::proportional(13.0),
        egui::Color32::LIGHT_GRAY,
    );

    if values.is_empty() {
        painter.text(
            plot_rect.center(),
            egui::Align2::CENTER_CENTER,
            "No numeric values",
            egui::FontId::proportional(13.0),
            egui::Color32::LIGHT_GRAY,
        );
        return;
    }

    let mut min_v = values[0];
    let mut max_v = values[0];
    for v in values {
        min_v = min_v.min(*v);
        max_v = max_v.max(*v);
    }
    let (min_v, max_v) = normalize_range(min_v, max_v);

    let bins = bins.max(2).min(200);
    let mut counts = vec![0usize; bins];
    let width = (max_v - min_v) / bins as f64;
    for v in values {
        let mut idx = ((v - min_v) / width).floor() as isize;
        if idx < 0 {
            idx = 0;
        }
        if idx as usize >= bins {
            idx = bins as isize - 1;
        }
        counts[idx as usize] += 1;
    }
    let y_max = counts.iter().copied().max().unwrap_or(1).max(1) as f64;

    for (i, c) in counts.iter().enumerate() {
        let x0 = min_v + (i as f64) * width;
        let x1 = x0 + width;
        let y = *c as f64;
        let p0 = egui::pos2(
            plot_rect.left() + (((x0 - min_v) / (max_v - min_v)) as f32) * plot_rect.width(),
            plot_rect.bottom(),
        );
        let p1 = egui::pos2(
            plot_rect.left() + (((x1 - min_v) / (max_v - min_v)) as f32) * plot_rect.width(),
            plot_rect.bottom() - ((y / y_max) as f32) * plot_rect.height(),
        );
        let r = egui::Rect::from_min_max(egui::pos2(p0.x, p1.y), egui::pos2(p1.x, p0.y));
        painter.rect_filled(
            r.shrink2(egui::vec2(1.0, 0.0)),
            0.0,
            egui::Color32::from_rgb(90, 150, 230),
        );
    }

    painter.text(
        egui::pos2(plot_rect.left(), plot_rect.bottom() + 4.0),
        egui::Align2::LEFT_TOP,
        format!("x:[{} .. {}]", format_number(min_v), format_number(max_v)),
        egui::FontId::proportional(11.0),
        egui::Color32::GRAY,
    );
    painter.text(
        egui::pos2(plot_rect.right(), plot_rect.top() - 2.0),
        egui::Align2::RIGHT_BOTTOM,
        format!("max count: {}", y_max as usize),
        egui::FontId::proportional(11.0),
        egui::Color32::GRAY,
    );
}

fn normalize_range(min_v: f64, max_v: f64) -> (f64, f64) {
    if !min_v.is_finite() || !max_v.is_finite() {
        return (0.0, 1.0);
    }
    if (max_v - min_v).abs() < f64::EPSILON {
        let pad = if min_v == 0.0 {
            1.0
        } else {
            min_v.abs() * 0.05
        };
        (min_v - pad, max_v + pad)
    } else {
        (min_v, max_v)
    }
}

fn plot_canvas_size(ui: &egui::Ui) -> egui::Vec2 {
    let w = ui.available_width();
    let h = ui.available_height();
    let width = if w.is_finite() { w.max(240.0) } else { 360.0 };
    let height = if h.is_finite() { h.max(240.0) } else { 280.0 };
    egui::vec2(width, height)
}

fn apply_calc_column(
    loaded: &LoadedNpy,
    layout: TableLayout,
    state: &mut NpyWindowUiState,
) -> Result<String, String> {
    let total_cols = layout.display_base_cols + state.derived_columns.len();
    if total_cols == 0 {
        return Err("No columns available".to_string());
    }

    let expr_text = state.calc_expr.trim();
    if expr_text.is_empty() {
        return Err("Expression is empty".to_string());
    }
    let expr = parse_calc_expression(expr_text)?;
    if let Some(max_col) = max_column_index(&expr) {
        if max_col >= total_cols {
            return Err(format!(
                "Column reference ${} is out of range (1..={})",
                max_col + 1,
                total_cols
            ));
        }
    }

    let mut out = Vec::with_capacity(layout.rows);
    let mut valid_count = 0usize;
    for row in 0..layout.rows {
        let value = eval_expression_row(
            &expr,
            loaded,
            layout,
            &state.derived_columns,
            row,
            state.trig_in_degrees,
        );
        let text = if let Some(v) = value {
            valid_count += 1;
            format_number(v)
        } else {
            "0".to_string()
        };
        out.push(text);
    }

    if valid_count == 0 {
        return Err(
            "No numeric rows were produced. Check referenced columns and expression.".to_string(),
        );
    }

    let name = if state.calc_name.trim().is_empty() {
        format!("Calc{}", state.derived_columns.len() + 1)
    } else {
        state.calc_name.trim().to_string()
    };

    state.derived_columns.push(DerivedColumn {
        name: name.clone(),
        values: out,
    });
    state.calc_name.clear();
    Ok(name)
}

fn parse_numeric_cell(
    loaded: &LoadedNpy,
    layout: TableLayout,
    derived: &[DerivedColumn],
    row: usize,
    col: usize,
) -> Option<f64> {
    let text = cell_value(loaded, layout, derived, row, col)?;
    parse_numeric_string(text)
}

fn parse_numeric_string(text: &str) -> Option<f64> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok().filter(|v| v.is_finite())
}

fn parse_calc_expression(input: &str) -> Result<Expr, String> {
    let mut p = ExprParser { input, pos: 0 };
    let expr = p.parse_expression()?;
    p.skip_ws();
    if !p.is_eof() {
        return Err(format!("Unexpected token at position {}", p.pos + 1));
    }
    Ok(expr)
}

struct ExprParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> ExprParser<'a> {
    fn parse_expression(&mut self) -> Result<Expr, String> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<Expr, String> {
        let mut node = self.parse_mul_div()?;
        loop {
            self.skip_ws();
            if self.consume_char('+') {
                let rhs = self.parse_mul_div()?;
                node = Expr::Binary {
                    op: BinaryOp::Add,
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                };
            } else if self.consume_char('-') {
                let rhs = self.parse_mul_div()?;
                node = Expr::Binary {
                    op: BinaryOp::Sub,
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                };
            } else {
                break;
            }
        }
        Ok(node)
    }

    fn parse_mul_div(&mut self) -> Result<Expr, String> {
        let mut node = self.parse_power()?;
        loop {
            self.skip_ws();
            if self.consume_char('*') {
                let rhs = self.parse_power()?;
                node = Expr::Binary {
                    op: BinaryOp::Mul,
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                };
            } else if self.consume_char('/') {
                let rhs = self.parse_power()?;
                node = Expr::Binary {
                    op: BinaryOp::Div,
                    lhs: Box::new(node),
                    rhs: Box::new(rhs),
                };
            } else {
                break;
            }
        }
        Ok(node)
    }

    fn parse_power(&mut self) -> Result<Expr, String> {
        let node = self.parse_unary()?;
        self.skip_ws();
        if self.consume_char('^') {
            let rhs = self.parse_power()?;
            Ok(Expr::Binary {
                op: BinaryOp::Pow,
                lhs: Box::new(node),
                rhs: Box::new(rhs),
            })
        } else {
            Ok(node)
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        self.skip_ws();
        if self.consume_char('+') {
            self.parse_unary()
        } else if self.consume_char('-') {
            Ok(Expr::UnaryMinus(Box::new(self.parse_unary()?)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        self.skip_ws();
        if self.consume_char('(') {
            let expr = self.parse_expression()?;
            self.skip_ws();
            if !self.consume_char(')') {
                return Err(format!("Expected ')' at position {}", self.pos + 1));
            }
            return Ok(expr);
        }

        if self.consume_char('$') {
            let idx = self.parse_usize()?;
            if idx == 0 {
                return Err("Column index must be >= 1".to_string());
            }
            return Ok(Expr::Column(idx - 1));
        }

        if let Some(name) = self.parse_identifier() {
            return self.parse_function_call(&name);
        }

        if let Some(num) = self.parse_number()? {
            return Ok(Expr::Number(num));
        }

        Err(format!("Unexpected token at position {}", self.pos + 1))
    }

    fn parse_function_call(&mut self, name: &str) -> Result<Expr, String> {
        self.skip_ws();
        if !self.consume_char('(') {
            return Err(format!(
                "Expected '(' after function '{}' at position {}",
                name,
                self.pos + 1
            ));
        }
        if name.eq_ignore_ascii_case("pow") {
            let lhs = self.parse_expression()?;
            self.skip_ws();
            if !self.consume_char(',') {
                return Err(format!(
                    "Expected ',' in pow(...) at position {}",
                    self.pos + 1
                ));
            }
            let rhs = self.parse_expression()?;
            self.skip_ws();
            if !self.consume_char(')') {
                return Err(format!(
                    "Expected ')' after pow at position {}",
                    self.pos + 1
                ));
            }
            return Ok(Expr::Binary {
                op: BinaryOp::Pow,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            });
        }
        if name.eq_ignore_ascii_case("atan2") {
            let lhs = self.parse_expression()?;
            self.skip_ws();
            if !self.consume_char(',') {
                return Err(format!(
                    "Expected ',' in atan2(...) at position {}",
                    self.pos + 1
                ));
            }
            let rhs = self.parse_expression()?;
            self.skip_ws();
            if !self.consume_char(')') {
                return Err(format!(
                    "Expected ')' after atan2 at position {}",
                    self.pos + 1
                ));
            }
            return Ok(Expr::Binary {
                op: BinaryOp::Atan2,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            });
        }

        let arg = self.parse_expression()?;
        self.skip_ws();
        if !self.consume_char(')') {
            return Err(format!(
                "Expected ')' after function '{}' at position {}",
                name,
                self.pos + 1
            ));
        }
        let func = match name.to_ascii_lowercase().as_str() {
            "sin" => FuncOp::Sin,
            "cos" => FuncOp::Cos,
            "tan" => FuncOp::Tan,
            "exp" => FuncOp::Exp,
            "abs" => FuncOp::Abs,
            "sqrt" => FuncOp::Sqrt,
            "asin" => FuncOp::Asin,
            "acos" => FuncOp::Acos,
            "atan" => FuncOp::Atan,
            _ => return Err(format!("Unknown function '{}'", name)),
        };
        Ok(Expr::Func {
            func,
            arg: Box::new(arg),
        })
    }

    fn parse_usize(&mut self) -> Result<usize, String> {
        self.skip_ws();
        let start = self.pos;
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_digit() {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(format!("Expected number at position {}", self.pos + 1));
        }
        self.input[start..self.pos]
            .parse::<usize>()
            .map_err(|_| format!("Invalid integer at position {}", start + 1))
    }

    fn parse_identifier(&mut self) -> Option<String> {
        self.skip_ws();
        let start = self.pos;
        if let Some(ch) = self.peek_char() {
            if ch.is_ascii_alphabetic() || ch == '_' {
                self.pos += ch.len_utf8();
            } else {
                return None;
            }
        } else {
            return None;
        }
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }
        if self.pos == start {
            None
        } else {
            Some(self.input[start..self.pos].to_string())
        }
    }

    fn parse_number(&mut self) -> Result<Option<f64>, String> {
        self.skip_ws();
        let start = self.pos;
        let mut has_digit = false;
        let mut seen_dot = false;

        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_digit() {
                has_digit = true;
                self.pos += 1;
            } else if ch == '.' && !seen_dot {
                seen_dot = true;
                self.pos += 1;
            } else {
                break;
            }
        }

        if let Some(ch) = self.peek_char() {
            if (ch == 'e' || ch == 'E') && has_digit {
                self.pos += 1;
                if let Some(sign) = self.peek_char() {
                    if sign == '+' || sign == '-' {
                        self.pos += 1;
                    }
                }
                let exp_start = self.pos;
                while let Some(d) = self.peek_char() {
                    if d.is_ascii_digit() {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                if self.pos == exp_start {
                    return Err(format!("Invalid exponent at position {}", self.pos + 1));
                }
            }
        }

        if self.pos == start || (!has_digit && !seen_dot) {
            self.pos = start;
            return Ok(None);
        }

        let s = &self.input[start..self.pos];
        let n = s
            .parse::<f64>()
            .map_err(|_| format!("Invalid number '{}' at position {}", s, start + 1))?;
        Ok(Some(n))
    }

    fn consume_char(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.pos += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }
}

fn max_column_index(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Column(c) => Some(*c),
        Expr::Number(_) => None,
        Expr::UnaryMinus(inner) => max_column_index(inner),
        Expr::Binary { lhs, rhs, .. } => match (max_column_index(lhs), max_column_index(rhs)) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        },
        Expr::Func { arg, .. } => max_column_index(arg),
    }
}

fn eval_expression_row(
    expr: &Expr,
    loaded: &LoadedNpy,
    layout: TableLayout,
    derived: &[DerivedColumn],
    row: usize,
    trig_in_degrees: bool,
) -> Option<f64> {
    fn finite_or_zero(v: f64) -> f64 {
        if v.is_finite() { v } else { 0.0 }
    }

    match expr {
        Expr::Number(n) => Some(*n),
        Expr::Column(c) => parse_numeric_cell(loaded, layout, derived, row, *c),
        Expr::UnaryMinus(inner) => {
            eval_expression_row(inner, loaded, layout, derived, row, trig_in_degrees)
                .map(|v| finite_or_zero(-v))
        }
        Expr::Binary { op, lhs, rhs } => {
            let l = eval_expression_row(lhs, loaded, layout, derived, row, trig_in_degrees)?;
            let r = eval_expression_row(rhs, loaded, layout, derived, row, trig_in_degrees)?;
            match op {
                BinaryOp::Add => Some(finite_or_zero(l + r)),
                BinaryOp::Sub => Some(finite_or_zero(l - r)),
                BinaryOp::Mul => Some(finite_or_zero(l * r)),
                BinaryOp::Div => {
                    if r == 0.0 {
                        Some(0.0)
                    } else {
                        Some(finite_or_zero(l / r))
                    }
                }
                BinaryOp::Pow => Some(finite_or_zero(l.powf(r))),
                BinaryOp::Atan2 => {
                    if l == 0.0 && r == 0.0 {
                        Some(0.0)
                    } else {
                        let a = l.atan2(r);
                        let out = if trig_in_degrees { a.to_degrees() } else { a };
                        Some(finite_or_zero(out))
                    }
                }
            }
        }
        Expr::Func { func, arg } => {
            let x = eval_expression_row(arg, loaded, layout, derived, row, trig_in_degrees)?;
            match func {
                FuncOp::Sin => {
                    let a = if trig_in_degrees { x.to_radians() } else { x };
                    Some(finite_or_zero(a.sin()))
                }
                FuncOp::Cos => {
                    let a = if trig_in_degrees { x.to_radians() } else { x };
                    Some(finite_or_zero(a.cos()))
                }
                FuncOp::Tan => {
                    let a = if trig_in_degrees { x.to_radians() } else { x };
                    Some(finite_or_zero(a.tan()))
                }
                FuncOp::Exp => Some(finite_or_zero(x.exp())),
                FuncOp::Abs => Some(finite_or_zero(x.abs())),
                FuncOp::Sqrt => {
                    if x < 0.0 {
                        Some(0.0)
                    } else {
                        Some(finite_or_zero(x.sqrt()))
                    }
                }
                FuncOp::Asin => {
                    if (-1.0..=1.0).contains(&x) {
                        let a = x.asin();
                        let out = if trig_in_degrees { a.to_degrees() } else { a };
                        Some(finite_or_zero(out))
                    } else {
                        Some(0.0)
                    }
                }
                FuncOp::Acos => {
                    if (-1.0..=1.0).contains(&x) {
                        let a = x.acos();
                        let out = if trig_in_degrees { a.to_degrees() } else { a };
                        Some(finite_or_zero(out))
                    } else {
                        Some(0.0)
                    }
                }
                FuncOp::Atan => {
                    let a = x.atan();
                    let out = if trig_in_degrees { a.to_degrees() } else { a };
                    Some(finite_or_zero(out))
                }
            }
        }
    }
}

fn format_number(value: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    if value.abs() >= 1e8 || value.abs() < 1e-6 {
        return format!("{value:.6e}");
    }
    let s = format!("{value:.12}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn draw_calc_like_preview(
    ui: &mut egui::Ui,
    key: &str,
    loaded: &LoadedNpy,
    layout: TableLayout,
    state: &mut NpyWindowUiState,
) {
    let labels = display_column_labels(loaded, layout, &state.derived_columns);
    let total_cols = labels.len();
    if total_cols == 0 {
        ui.label("No values");
        return;
    }

    if layout.is_complex {
        ui.label(format!(
            "all complex values as cells: {} rows x {} columns (Re/Im)",
            layout.rows, layout.base_cols
        ));
    } else {
        ui.label(format!(
            "all values as cells: {} rows x {} columns",
            layout.rows, layout.base_cols
        ));
    }
    ui.label(format!(
        "showing {} / {} values",
        layout.rows * layout.base_cols,
        loaded.total_elements
    ));

    let total_rows = layout.rows + 1;
    let table_width = ROW_NUM_WIDTH + (total_cols as f32) * CELL_WIDTH;
    let search_query = normalize_query(&state.search_query);
    if state.search_requested {
        state.search_requested = false;
        state.search_jump_target = search_query.as_ref().and_then(|q| {
            find_first_match_cell(loaded, layout, &state.derived_columns, total_cols, q)
        });
        if let Some((row, col)) = state.search_jump_target {
            state.search_jump_frames = 3;
            state.search_status = Some(format!("Found at row {}, col ${}", row + 1, col + 1));
            ui.ctx().request_repaint();
        } else if search_query.is_some() {
            state.search_jump_frames = 0;
            state.search_status = Some("No match".to_string());
        } else {
            state.search_jump_frames = 0;
            state.search_status = None;
        }
    }

    let mut scroll = egui::ScrollArea::both()
        .id_salt(format!("data_scroll_{key}"))
        .auto_shrink([false, false]);
    if state.search_jump_frames > 0 {
        if let Some((row, col)) = state.search_jump_target {
            let y = (((row + 1) as f32) * ROW_HEIGHT - ROW_HEIGHT * 2.0).max(0.0);
            let x = (ROW_NUM_WIDTH + (col as f32) * CELL_WIDTH - CELL_WIDTH).max(0.0);
            scroll = scroll.vertical_scroll_offset(y).horizontal_scroll_offset(x);
        }
    }

    scroll.show_rows(ui, ROW_HEIGHT, total_rows, |ui, row_range| {
        ui.set_min_width(table_width);
        for visual_row in row_range {
            if visual_row == 0 {
                draw_header_row(ui, key, &labels);
            } else {
                let row = visual_row - 1;
                draw_data_row(
                    ui,
                    row,
                    loaded,
                    layout,
                    &state.derived_columns,
                    &labels,
                    &search_query,
                );
            }
        }
    });
    if state.search_jump_frames > 0 {
        state.search_jump_frames -= 1;
        if state.search_jump_frames == 0 {
            state.search_jump_target = None;
        } else {
            ui.ctx().request_repaint();
        }
    }
}

fn preview_table_columns(shape: &[usize], preview_len: usize) -> usize {
    if preview_len == 0 {
        return 1;
    }

    let raw_cols = if shape.len() <= 1 {
        1
    } else {
        shape.last().copied().unwrap_or(1).max(1)
    };

    raw_cols.min(preview_len).max(1)
}

fn draw_header_row(ui: &mut egui::Ui, key: &str, labels: &[String]) {
    ui.push_id(format!("header_{key}"), |ui| {
        ui.horizontal(|ui| {
            ui.add_sized([ROW_NUM_WIDTH, ROW_HEIGHT], egui::Label::new("#"));
            for label in labels {
                ui.add_sized(
                    [CELL_WIDTH, ROW_HEIGHT],
                    egui::Label::new(egui::RichText::new(label).strong()),
                );
            }
        });
    });
}

fn draw_data_row(
    ui: &mut egui::Ui,
    row: usize,
    loaded: &LoadedNpy,
    layout: TableLayout,
    derived: &[DerivedColumn],
    labels: &[String],
    search_query: &Option<String>,
) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [ROW_NUM_WIDTH, ROW_HEIGHT],
            egui::Label::new((row + 1).to_string()),
        );
        for col in 0..labels.len() {
            let value = cell_value(loaded, layout, derived, row, col).unwrap_or("");
            let highlight = search_query
                .as_ref()
                .map(|q| cell_matches_query(value, q))
                .unwrap_or(false);
            draw_readonly_cell(ui, value, highlight);
        }
    });
}

fn display_column_labels(
    loaded: &LoadedNpy,
    layout: TableLayout,
    derived: &[DerivedColumn],
) -> Vec<String> {
    let mut out = Vec::with_capacity(layout.display_base_cols + derived.len());
    if layout.is_complex {
        for col in 0..layout.base_cols {
            let name = excel_column_name(col);
            out.push(format!("{name}.Re"));
            out.push(format!("{name}.Im"));
        }
    } else {
        if let Some(field_names) = &loaded.field_names {
            for col in 0..layout.base_cols {
                if let Some(name) = field_names.get(col) {
                    out.push(name.clone());
                } else {
                    out.push(excel_column_name(col));
                }
            }
        } else {
            for col in 0..layout.base_cols {
                out.push(excel_column_name(col));
            }
        }
    }
    for d in derived {
        out.push(d.name.clone());
    }
    out
}

fn cell_value<'a>(
    loaded: &'a LoadedNpy,
    layout: TableLayout,
    derived: &'a [DerivedColumn],
    row: usize,
    col: usize,
) -> Option<&'a str> {
    if col < layout.display_base_cols {
        if layout.is_complex {
            let base_col = col / 2;
            let idx = row.checked_mul(layout.base_cols)?.checked_add(base_col)?;
            let cell = loaded.complex_values.as_ref()?.get(idx)?;
            if col % 2 == 0 {
                Some(cell.re.as_str())
            } else {
                Some(cell.im.as_str())
            }
        } else {
            let idx = row.checked_mul(layout.base_cols)?.checked_add(col)?;
            loaded.preview_values.get(idx).map(String::as_str)
        }
    } else {
        let dcol = col - layout.display_base_cols;
        derived.get(dcol)?.values.get(row).map(String::as_str)
    }
}

fn normalize_query(query: &str) -> Option<String> {
    let t = query.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

fn find_first_match_cell(
    loaded: &LoadedNpy,
    layout: TableLayout,
    derived: &[DerivedColumn],
    total_cols: usize,
    query: &str,
) -> Option<(usize, usize)> {
    for row in 0..layout.rows {
        for col in 0..total_cols {
            if let Some(value) = cell_value(loaded, layout, derived, row, col) {
                if cell_matches_query(value, query) {
                    return Some((row, col));
                }
            }
        }
    }
    None
}

fn cell_matches_query(value: &str, query: &str) -> bool {
    if query.is_empty() {
        return false;
    }
    let normalized = value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .replace('\0', "");
    if normalized.contains(query) {
        return true;
    }
    normalized.to_lowercase().contains(&query.to_lowercase())
}

fn draw_readonly_cell(ui: &mut egui::Ui, value: &str, highlight: bool) {
    let text = if highlight {
        egui::RichText::new(value).background_color(egui::Color32::from_rgb(70, 60, 0))
    } else {
        egui::RichText::new(value)
    };
    ui.add_sized([CELL_WIDTH, ROW_HEIGHT], egui::Label::new(text));
}

fn excel_column_name(mut index: usize) -> String {
    let mut name = String::new();
    loop {
        let rem = index % 26;
        name.insert(0, (b'A' + rem as u8) as char);
        if index < 26 {
            break;
        }
        index = (index / 26) - 1;
    }
    name
}
