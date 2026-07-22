//! The workbench design system: embedded Inter typography, Phosphor
//! icons, a type/spacing scale, and curated dark + light palettes with
//! a single teal accent. Applied once at startup; both themes are
//! styled, so the OS preference (or the egui theme toggle) picks
//! between two intentional looks rather than one styled and one stock.

use eframe::egui;
use eframe::egui::style::{Selection, WidgetVisuals, Widgets};
use eframe::egui::{
    Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Shadow, Stroke,
    TextStyle, Theme, Visuals,
};

/// Brand accent (Geomermaids teal).
pub const ACCENT: Color32 = Color32::from_rgb(38, 166, 154);
/// Accent variant readable on light backgrounds.
pub const ACCENT_DARK: Color32 = Color32::from_rgb(13, 128, 118);

/// Family name for medium-weight text (headings, emphasized labels).
pub const MEDIUM: &str = "inter-medium";

pub fn apply(ctx: &egui::Context) {
    ctx.set_fonts(fonts());
    ctx.all_styles_mut(|style| {
        // Type scale.
        style.text_styles = [
            (TextStyle::Small, FontId::new(11.0, FontFamily::Proportional)),
            (TextStyle::Body, FontId::new(13.0, FontFamily::Proportional)),
            (TextStyle::Button, FontId::new(13.0, FontFamily::Proportional)),
            (TextStyle::Heading, FontId::new(17.0, FontFamily::Name(MEDIUM.into()))),
            (TextStyle::Monospace, FontId::new(12.0, FontFamily::Monospace)),
        ]
        .into();
        // Spacing scale: roomier than egui's default, tighter than a
        // consumer app — this is a workbench.
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(10.0, 5.0);
        style.spacing.menu_margin = egui::Margin::same(8);
        style.spacing.window_margin = egui::Margin::same(12);
        style.spacing.scroll.bar_width = 8.0;
        style.spacing.scroll.floating_allocated_width = 8.0;
        style.spacing.slider_width = 110.0;
        style.spacing.combo_width = 110.0;
        style.spacing.interact_size.y = 22.0;
    });
    ctx.set_visuals_of(Theme::Dark, dark());
    ctx.set_visuals_of(Theme::Light, light());
}

fn fonts() -> FontDefinitions {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "inter".into(),
        FontData::from_static(include_bytes!("../assets/fonts/Inter-Regular.otf")).into(),
    );
    fonts.font_data.insert(
        MEDIUM.into(),
        FontData::from_static(include_bytes!("../assets/fonts/Inter-Medium.otf")).into(),
    );
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
    // Order: Inter for text, Phosphor for icons, egui defaults last.
    // This only works because our Inter is SUBSETTED to U+0020-D7FF —
    // stock Inter ships stylistic alternates in the Private Use Area
    // (where Phosphor's icons live), and Phosphor maps icon ligature
    // codepoints over Latin letters, so with full fonts either order
    // shadows the other. Re-subset with pyftsubset if updating Inter.
    let prop = fonts.families.get_mut(&FontFamily::Proportional).unwrap();
    prop.retain(|n| n != "phosphor" && n != "inter");
    prop.insert(0, "inter".into());
    prop.insert(1, "phosphor".into());
    let medium_chain: Vec<String> = std::iter::once(MEDIUM.to_string())
        .chain(prop.iter().cloned())
        .collect();
    fonts.families.insert(FontFamily::Name(MEDIUM.into()), medium_chain);
    fonts
}

const RADIUS: CornerRadius = CornerRadius::same(6);

/// Card container for list items (layer rows and friends): subtle fill,
/// hairline border, roomier rounding than widgets.
pub fn card(style: &egui::Style) -> egui::Frame {
    egui::Frame::new()
        .fill(style.visuals.faint_bg_color)
        .stroke(style.visuals.widgets.noninteractive.bg_stroke)
        .corner_radius(CornerRadius::same(8))
        .inner_margin(egui::Margin::symmetric(7, 5))
}

/// Compact control density for busy side panels: tighter rhythm and
/// smaller interactive rows than the global scale.
pub fn compact(ui: &mut egui::Ui) {
    let s = ui.spacing_mut();
    s.item_spacing = egui::vec2(6.0, 3.0);
    s.interact_size.y = 18.0;
    s.button_padding = egui::vec2(6.0, 2.0);
    s.slider_width = 88.0;
}

fn widgets_dark() -> Widgets {
    let base = |bg: Color32, fg: Color32, stroke: Color32| WidgetVisuals {
        bg_fill: bg,
        weak_bg_fill: bg,
        bg_stroke: Stroke::new(1.0, stroke),
        corner_radius: RADIUS,
        fg_stroke: Stroke::new(1.0, fg),
        expansion: 0.0,
    };
    let text = Color32::from_rgb(214, 218, 224);
    let text_strong = Color32::from_rgb(238, 240, 244);
    Widgets {
        noninteractive: base(
            Color32::from_rgb(27, 29, 34),
            Color32::from_rgb(170, 176, 184),
            Color32::from_rgb(46, 49, 56),
        ),
        inactive: base(Color32::from_rgb(44, 47, 54), text, Color32::TRANSPARENT),
        hovered: base(
            Color32::from_rgb(54, 58, 66),
            text_strong,
            Color32::from_rgb(72, 77, 86),
        ),
        active: base(Color32::from_rgb(38, 41, 48), text_strong, ACCENT),
        open: base(Color32::from_rgb(38, 41, 48), text, Color32::from_rgb(72, 77, 86)),
    }
}

fn dark() -> Visuals {
    let mut v = Visuals::dark();
    v.override_text_color = None;
    v.panel_fill = Color32::from_rgb(22, 24, 28);
    v.window_fill = Color32::from_rgb(27, 29, 34);
    v.extreme_bg_color = Color32::from_rgb(15, 16, 19);
    v.faint_bg_color = Color32::from_rgb(31, 33, 39);
    v.code_bg_color = Color32::from_rgb(15, 16, 19);
    v.widgets = widgets_dark();
    v.selection = Selection {
        bg_fill: ACCENT.gamma_multiply(0.35),
        stroke: Stroke::new(1.0, ACCENT),
    };
    v.hyperlink_color = ACCENT;
    v.warn_fg_color = Color32::from_rgb(242, 170, 66);
    v.error_fg_color = Color32::from_rgb(235, 90, 85);
    v.window_corner_radius = CornerRadius::same(10);
    v.menu_corner_radius = CornerRadius::same(8);
    v.window_stroke = Stroke::new(1.0, Color32::from_rgb(50, 53, 61));
    v.window_shadow = Shadow {
        offset: [0, 6],
        blur: 24,
        spread: 0,
        color: Color32::from_black_alpha(96),
    };
    v.popup_shadow = Shadow {
        offset: [0, 4],
        blur: 12,
        spread: 0,
        color: Color32::from_black_alpha(80),
    };
    v.slider_trailing_fill = true;
    v
}

fn widgets_light() -> Widgets {
    let base = |bg: Color32, fg: Color32, stroke: Color32| WidgetVisuals {
        bg_fill: bg,
        weak_bg_fill: bg,
        bg_stroke: Stroke::new(1.0, stroke),
        corner_radius: RADIUS,
        fg_stroke: Stroke::new(1.0, fg),
        expansion: 0.0,
    };
    let text = Color32::from_rgb(50, 55, 62);
    let text_strong = Color32::from_rgb(24, 27, 32);
    Widgets {
        noninteractive: base(
            Color32::from_rgb(247, 247, 249),
            Color32::from_rgb(95, 101, 110),
            Color32::from_rgb(222, 224, 229),
        ),
        inactive: base(Color32::from_rgb(232, 233, 237), text, Color32::TRANSPARENT),
        hovered: base(
            Color32::from_rgb(222, 224, 230),
            text_strong,
            Color32::from_rgb(196, 200, 208),
        ),
        active: base(Color32::from_rgb(214, 217, 224), text_strong, ACCENT_DARK),
        open: base(Color32::from_rgb(224, 226, 232), text, Color32::from_rgb(196, 200, 208)),
    }
}

fn light() -> Visuals {
    let mut v = Visuals::light();
    v.panel_fill = Color32::from_rgb(247, 247, 249);
    v.window_fill = Color32::from_rgb(252, 252, 253);
    v.extreme_bg_color = Color32::WHITE;
    v.faint_bg_color = Color32::from_rgb(240, 241, 244);
    v.widgets = widgets_light();
    v.selection = Selection {
        bg_fill: ACCENT.gamma_multiply(0.25),
        stroke: Stroke::new(1.0, ACCENT_DARK),
    };
    v.hyperlink_color = ACCENT_DARK;
    v.warn_fg_color = Color32::from_rgb(178, 112, 8);
    v.error_fg_color = Color32::from_rgb(190, 42, 38);
    v.window_corner_radius = CornerRadius::same(10);
    v.menu_corner_radius = CornerRadius::same(8);
    v.window_stroke = Stroke::new(1.0, Color32::from_rgb(216, 219, 226));
    v.window_shadow = Shadow {
        offset: [0, 6],
        blur: 24,
        spread: 0,
        color: Color32::from_black_alpha(40),
    };
    v.popup_shadow = Shadow {
        offset: [0, 4],
        blur: 12,
        spread: 0,
        color: Color32::from_black_alpha(32),
    };
    v.slider_trailing_fill = true;
    v
}
