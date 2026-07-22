#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod context;
mod cookbook;
mod data;
mod map;
mod picking;
mod sql;
mod theme;

use eframe::egui;

/// Window/taskbar icon, embedded at build time (macOS Dock needs an .app
/// bundle with assets/geopq-workbench.icns instead).
fn app_icon() -> egui::IconData {
    let img = image::load_from_memory(include_bytes!("../assets/icon-256.png"))
        .expect("builtin icon")
        .into_rgba8();
    let (width, height) = img.dimensions();
    egui::IconData {
        rgba: img.into_raw(),
        width,
        height,
    }
}

fn main() -> eframe::Result {
    env_logger::init();
    // Args: local paths or http(s) URLs.
    let files: Vec<data::source::Source> = std::env::args()
        .skip(1)
        .map(|a| {
            if a.starts_with("http://") || a.starts_with("https://") {
                data::source::Source::Remote { url: a, len: 0 }
            } else if a.starts_with("s3://") {
                data::source::Source::S3 {
                    uri: a,
                    profile: std::env::var("AWS_PROFILE").ok(),
                    endpoint: None,
                    url: String::new(),
                    len: 0,
                }
            } else if std::path::Path::new(&a).is_dir() {
                data::source::Source::Dir(a.into())
            } else {
                data::source::Source::Local(a.into())
            }
        })
        .collect();

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        multisampling: crate::map::renderer::MSAA_SAMPLES as u16,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1480.0, 940.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("GeoPQ Workbench")
            .with_icon(app_icon()),
        ..Default::default()
    };

    eframe::run_native(
        "geopq-workbench",
        native_options,
        Box::new(move |cc| Ok(Box::new(app::ViewerApp::new(cc, files)))),
    )
}
