#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod data;
mod map;
mod picking;

use eframe::egui;

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
            .with_title("GeoParquet Viewer"),
        ..Default::default()
    };

    eframe::run_native(
        "geopq-viewer",
        native_options,
        Box::new(move |cc| Ok(Box::new(app::ViewerApp::new(cc, files)))),
    )
}
