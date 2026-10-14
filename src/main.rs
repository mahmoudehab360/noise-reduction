#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod gpu;
mod media;
mod noise;
mod params;
mod worker;

use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 860.0])
            .with_min_inner_size([1000.0, 620.0])
            .with_title("LowLight DN - Temporal + Spatial denoiser"),
        ..Default::default()
    };
    eframe::run_native(
        "LowLight DN",
        options,
        Box::new(|cc| Box::new(app::App::new(cc))),
    )
}
