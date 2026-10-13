//! The look of vearl.org: the same colours, the same three fonts (Instrument Sans, Instrument Serif, IBM Plex Mono), sharp corners,
//! mono capital labels with wide letter spacing, hairlines, and the mint -> lavender -> rose "nacre" gradient.
use eframe::egui::{
    self, epaint::Mesh, vec2, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Margin, Pos2, Rect, Response, RichText, Sense, Shape, Stroke,
    StrokeKind, Ui,
};
use std::sync::Arc;

// colours of site.css (:root)
pub const BG: Color32 = Color32::from_rgb(0x07, 0x08, 0x0b);
pub const BG2: Color32 = Color32::from_rgb(0x0b, 0x0d, 0x12);
pub const PANEL: Color32 = Color32::from_rgb(0x10, 0x13, 0x1a);
pub const LINE: Color32 = Color32::from_rgb(0x1b, 0x20, 0x29);
pub const LINE2: Color32 = Color32::from_rgb(0x28, 0x30, 0x40);
pub const INK: Color32 = Color32::from_rgb(0xec, 0xeb, 0xe6);
pub const INK2: Color32 = Color32::from_rgb(0xb7, 0xbc, 0xc6);
pub const MUTE: Color32 = Color32::from_rgb(0x8a, 0x91, 0xa0);
pub const DIM: Color32 = Color32::from_rgb(0x5a, 0x62, 0x72);
pub const MINT: Color32 = Color32::from_rgb(0x9f, 0xf0, 0xe2);
pub const MINT_HOVER: Color32 = Color32::from_rgb(0xc8, 0xff, 0xf5);
pub const LAV: Color32 = Color32::from_rgb(0xb9, 0xa6, 0xff);
pub const ROSE: Color32 = Color32::from_rgb(0xf4, 0xbc, 0xd6);
pub const ACCENT_INK: Color32 = Color32::from_rgb(0x06, 0x11, 0x0f);
pub const OK: Color32 = Color32::from_rgb(0x7e, 0xe0, 0xa2);
pub const WARN: Color32 = Color32::from_rgb(0xf0, 0xc6, 0x74);
pub const BAD: Color32 = Color32::from_rgb(0xff, 0x8f, 0x8f);
pub const SERIF_INK: Color32 = Color32::from_rgb(0xcd, 0xc3, 0xff);

pub fn alpha(c: Color32, a: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

fn named(n: &str) -> FontFamily {
    FontFamily::Name(n.into())
}
pub fn sans(size: f32) -> FontId {
    FontId::new(size, named("sans"))
}
pub fn sans_medium(size: f32) -> FontId {
    FontId::new(size, named("sans-medium"))
}
pub fn sans_semibold(size: f32) -> FontId {
    FontId::new(size, named("sans-semibold"))
}
pub fn serif(size: f32) -> FontId {
    FontId::new(size, named("serif"))
}
pub fn serif_italic(size: f32) -> FontId {
    FontId::new(size, named("serif-italic"))
}
pub fn mono(size: f32) -> FontId {
    FontId::new(size, named("mono"))
}
pub fn mono_medium(size: f32) -> FontId {
    FontId::new(size, named("mono-medium"))
}
pub fn mono_semibold(size: f32) -> FontId {
    FontId::new(size, named("mono-semibold"))
}

pub fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::empty();
    let list: [(&str, &[u8]); 8] = [
        ("sans", include_bytes!("../assets/fonts/InstrumentSans-regular.ttf")),
        ("sans-medium", include_bytes!("../assets/fonts/InstrumentSans-medium.ttf")),
        ("sans-semibold", include_bytes!("../assets/fonts/InstrumentSans-semibold.ttf")),
        ("serif", include_bytes!("../assets/fonts/InstrumentSerif-regular.ttf")),
        ("serif-italic", include_bytes!("../assets/fonts/InstrumentSerif-italic.ttf")),
        ("mono", include_bytes!("../assets/fonts/IBMPlexMono-regular.ttf")),
        ("mono-medium", include_bytes!("../assets/fonts/IBMPlexMono-medium.ttf")),
        ("mono-semibold", include_bytes!("../assets/fonts/IBMPlexMono-semibold.ttf")),
    ];
    for (name, bytes) in list {
        fonts.font_data.insert(name.to_string(), Arc::new(FontData::from_static(bytes)));
        // every family falls back to the regular sans for the few characters a face does not have
        fonts.families.insert(named(name), if name == "sans" { vec![name.to_string()] } else { vec![name.to_string(), "sans".to_string()] });
    }
    fonts.families.insert(FontFamily::Proportional, vec!["sans".into()]);
    fonts.families.insert(FontFamily::Monospace, vec!["mono".into(), "sans".into()]);
    ctx.set_fonts(fonts);

    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.all_styles_mut(|s| {
        use egui::TextStyle::*;
        s.text_styles.insert(Heading, sans_semibold(28.0));
        s.text_styles.insert(Body, sans(15.0));
        s.text_styles.insert(Button, mono_medium(12.5));
        s.text_styles.insert(Small, sans(12.5));
        s.text_styles.insert(Monospace, mono(13.5));
        let v = &mut s.visuals;
        v.dark_mode = true;
        v.override_text_color = None;
        v.panel_fill = BG;
        v.window_fill = PANEL;
        v.extreme_bg_color = BG2; // fields
        v.faint_bg_color = BG2;
        v.code_bg_color = BG2;
        v.window_stroke = Stroke::new(1.0, LINE2);
        v.window_corner_radius = CornerRadius::ZERO;
        v.menu_corner_radius = CornerRadius::ZERO;
        v.window_shadow = egui::epaint::Shadow { offset: [0, 18], blur: 48, spread: 0, color: Color32::from_black_alpha(190) };
        v.popup_shadow = v.window_shadow;
        v.hyperlink_color = MINT;
        v.selection.bg_fill = alpha(LAV, 90);
        v.selection.stroke = Stroke::new(1.0, MINT);
        v.text_cursor.stroke = Stroke::new(1.5, MINT);
        v.widgets.noninteractive.bg_fill = PANEL;
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, LINE);
        v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, INK2);
        for w in [&mut v.widgets.noninteractive, &mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
            w.corner_radius = CornerRadius::ZERO;
        }
        v.widgets.inactive.bg_fill = BG2;
        v.widgets.inactive.weak_bg_fill = BG2;
        v.widgets.inactive.bg_stroke = Stroke::new(1.0, LINE2);
        v.widgets.inactive.fg_stroke = Stroke::new(1.0, INK);
        v.widgets.hovered.bg_fill = BG2;
        v.widgets.hovered.weak_bg_fill = BG2;
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, DIM);
        v.widgets.hovered.fg_stroke = Stroke::new(1.0, INK);
        v.widgets.active.bg_fill = BG2;
        v.widgets.active.weak_bg_fill = BG2;
        v.widgets.active.bg_stroke = Stroke::new(1.0, MINT);
        v.widgets.active.fg_stroke = Stroke::new(1.0, MINT);
        v.widgets.open.bg_fill = BG2;
        v.widgets.open.bg_stroke = Stroke::new(1.0, MINT);
        s.spacing.item_spacing = vec2(10.0, 10.0);
        s.spacing.button_padding = vec2(16.0, 10.0);
        s.spacing.interact_size.y = 40.0;
        s.spacing.scroll.bar_width = 8.0;
        s.spacing.scroll.floating = true;
        s.spacing.window_margin = Margin::same(28);
        s.interaction.selectable_labels = true;
    });
}

// ------------------------------------------------------------------------------------------------ text

/// Mono capitals with wide letter spacing (the eyebrows and the buttons of the site), drawn glyph by glyph.
pub fn tracked_width(ctx: &egui::Context, text: &str, font: &FontId, tracking: f32) -> f32 {
    ctx.fonts_mut(|f| text.chars().map(|c| f.glyph_width(font, c) + tracking).sum::<f32>()) - if text.is_empty() { 0.0 } else { tracking }
}

pub fn paint_tracked(painter: &egui::Painter, pos: Pos2, text: &str, font: &FontId, color: Color32, tracking: f32) -> f32 {
    let mut x = pos.x;
    for c in text.chars() {
        let g = painter.layout_no_wrap(c.to_string(), font.clone(), color);
        let w = g.size().x;
        painter.galley(Pos2::new(x, pos.y), g, color);
        x += w + tracking;
    }
    x - pos.x - tracking
}

pub fn tracked(ui: &mut Ui, text: &str, font: FontId, color: Color32, tracking: f32) -> Response {
    let w = tracked_width(ui.ctx(), text, &font, tracking);
    let h = font.size * 1.35;
    let (rect, resp) = ui.allocate_exact_size(vec2(w, h), Sense::hover());
    paint_tracked(ui.painter(), rect.min, text, &font, color, tracking);
    resp
}

/// "01  BALANCE": the small mint label above every section of the site.
pub fn eyebrow(ui: &mut Ui, n: Option<&str>, text: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 14.0;
        if let Some(n) = n {
            tracked(ui, n, mono_medium(11.5), DIM, 1.8);
        }
        tracked(ui, &text.to_uppercase(), mono_medium(11.5), MINT, 1.8);
    });
}

/// A small label above a field or a figure: mono capitals, muted.
pub fn label(ui: &mut Ui, text: &str) {
    tracked(ui, &text.to_uppercase(), mono_medium(10.5), MUTE, 1.5);
}

pub fn body(ui: &mut Ui, text: &str) {
    ui.add(egui::Label::new(RichText::new(text).font(sans(15.0)).color(INK2)).wrap());
}

pub fn muted(ui: &mut Ui, text: &str) {
    ui.add(egui::Label::new(RichText::new(text).font(sans(13.5)).color(MUTE)).wrap());
}

/// A headline with an italic serif part, like the site headings: ("Your keys, ", "on your machine.")
pub fn headline(ui: &mut Ui, plain: &str, accent: &str, size: f32) {
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = ui.available_width();
    job.append(plain, 0.0, egui::TextFormat { font_id: sans_semibold(size), color: INK, ..Default::default() });
    job.append(accent, 0.0, egui::TextFormat { font_id: serif_italic(size * 1.12), color: SERIF_INK, ..Default::default() });
    ui.add(egui::Label::new(job).wrap());
}

// ------------------------------------------------------------------------------------------------ gradient

/// The mint -> lavender -> rose gradient of the site, horizontally, in a rectangle.
pub fn nacre_rect(painter: &egui::Painter, rect: Rect, a: u8) {
    let stops = [(0.0, MINT), (0.55, LAV), (1.0, ROSE)];
    let mut mesh = Mesh::default();
    for (t, c) in stops {
        let x = rect.left() + rect.width() * t;
        let c = alpha(c, a);
        mesh.colored_vertex(Pos2::new(x, rect.top()), c);
        mesh.colored_vertex(Pos2::new(x, rect.bottom()), c);
    }
    for i in 0..2u32 {
        let k = i * 2;
        mesh.add_triangle(k, k + 1, k + 2);
        mesh.add_triangle(k + 1, k + 3, k + 2);
    }
    painter.add(Shape::mesh(mesh));
}

/// A two-pixel gradient hairline across the full width of the space left.
pub fn nacre_line(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 2.0), Sense::hover());
    nacre_rect(ui.painter(), rect, 255);
}

// ------------------------------------------------------------------------------------------------ containers

pub fn card<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    let r = egui::Frame::new().fill(PANEL).stroke(Stroke::new(1.0, LINE)).inner_margin(Margin::same(24)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        add(ui)
    });
    r.inner
}

/// A card with a one-pixel nacre edge on top (the highlighted blocks of the site).
pub fn hero_card<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    let r = egui::Frame::new().fill(PANEL).stroke(Stroke::new(1.0, LINE2)).inner_margin(Margin::same(28)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        add(ui)
    });
    let rr = r.response.rect;
    nacre_rect(ui.painter(), Rect::from_min_max(rr.left_top(), Pos2::new(rr.right(), rr.top() + 2.0)), 255);
    r.inner
}

#[derive(Clone, Copy, PartialEq)]
pub enum Tone {
    Info,
    Ok,
    Warn,
    Bad,
}

/// A message bar: a coloured edge, a tinted background, text.
pub fn notice(ui: &mut Ui, tone: Tone, text: &str) {
    let c = match tone {
        Tone::Info => MINT,
        Tone::Ok => OK,
        Tone::Warn => WARN,
        Tone::Bad => BAD,
    };
    let r = egui::Frame::new().fill(alpha(c, 18)).stroke(Stroke::new(1.0, alpha(c, 70))).inner_margin(Margin::symmetric(16, 12)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.add(egui::Label::new(RichText::new(text).font(sans(14.0)).color(INK)).wrap());
    });
    let rr = r.response.rect;
    ui.painter().rect_filled(Rect::from_min_size(rr.left_top(), vec2(3.0, rr.height())), 0.0, c);
}

// ------------------------------------------------------------------------------------------------ buttons

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Primary,
    Ghost,
}

pub fn draw_arrow(painter: &egui::Painter, center: Pos2, color: Color32) {
    let s = Stroke::new(1.5, color);
    painter.line_segment([center + vec2(-6.0, 0.0), center + vec2(6.0, 0.0)], s);
    painter.line_segment([center + vec2(2.0, -4.0), center + vec2(6.0, 0.0)], s);
    painter.line_segment([center + vec2(2.0, 4.0), center + vec2(6.0, 0.0)], s);
}

pub fn draw_check(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
    let s = Stroke::new(2.0, color);
    let k = size / 2.0;
    painter.line_segment([center + vec2(-k, 0.0), center + vec2(-k * 0.35, k * 0.7)], s);
    painter.line_segment([center + vec2(-k * 0.35, k * 0.7), center + vec2(k, -k * 0.7)], s);
}

pub fn button_ex(ui: &mut Ui, text: &str, kind: Kind, enabled: bool, arrow: bool, min_width: f32) -> Response {
    let font = mono_semibold(12.0);
    let tracking = 1.5;
    let label = text.to_uppercase();
    let tw = tracked_width(ui.ctx(), &label, &font, tracking);
    let w = (tw + 44.0 + if arrow { 26.0 } else { 0.0 }).max(min_width);
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 46.0), if enabled { Sense::click() } else { Sense::hover() });
    let hover = enabled && resp.hovered();
    let down = enabled && resp.is_pointer_button_down_on();
    let (fill, stroke, fg) = match (kind, enabled) {
        (_, false) => (Color32::TRANSPARENT, Stroke::new(1.0, LINE), DIM),
        (Kind::Primary, true) => {
            let f = if hover { MINT_HOVER } else { MINT };
            (f, Stroke::new(1.0, f), ACCENT_INK)
        }
        (Kind::Ghost, true) => (if down { alpha(MINT, 14) } else { Color32::TRANSPARENT }, Stroke::new(1.0, if hover { MINT } else { LINE2 }), if hover { MINT } else { INK }),
    };
    let p = ui.painter();
    p.rect(rect, 0.0, fill, stroke, StrokeKind::Inside);
    let content_w = tw + if arrow { 26.0 } else { 0.0 };
    let x0 = rect.center().x - content_w / 2.0;
    paint_tracked(p, Pos2::new(x0, rect.center().y - font.size * 0.68), &label, &font, fg, tracking);
    if arrow {
        let nudge = if hover { 3.0 } else { 0.0 };
        draw_arrow(p, Pos2::new(x0 + tw + 15.0 + nudge, rect.center().y), fg);
    }
    if enabled {
        ui.ctx().set_cursor_icon(if hover { egui::CursorIcon::PointingHand } else { egui::CursorIcon::Default });
    }
    resp
}

pub fn primary(ui: &mut Ui, text: &str, enabled: bool) -> Response {
    button_ex(ui, text, Kind::Primary, enabled, true, 0.0)
}

pub fn ghost(ui: &mut Ui, text: &str) -> Response {
    button_ex(ui, text, Kind::Ghost, true, false, 0.0)
}

/// A small text link (mint on hover), mono capitals.
pub fn link(ui: &mut Ui, text: &str) -> Response {
    let font = mono_medium(11.5);
    let label = text.to_uppercase();
    let w = tracked_width(ui.ctx(), &label, &font, 1.4);
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 22.0), Sense::click());
    let c = if resp.hovered() { MINT_HOVER } else { MINT };
    paint_tracked(ui.painter(), Pos2::new(rect.left(), rect.top() + 3.0), &label, &font, c, 1.4);
    if resp.hovered() {
        ui.painter().line_segment([Pos2::new(rect.left(), rect.bottom() - 1.0), Pos2::new(rect.right(), rect.bottom() - 1.0)], Stroke::new(1.0, c));
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

/// A small square button with the label "COPY" that turns into "COPIED" for two seconds.
pub fn copy_button(ui: &mut Ui, id: &str, text: &str) -> Response {
    let key = egui::Id::new(("copied", id));
    let now = ui.input(|i| i.time);
    let at: Option<f64> = ui.data(|d| d.get_temp(key));
    let just = at.is_some_and(|t| now - t < 2.0);
    let label = if just { "Copied" } else { "Copy" };
    let font = mono_semibold(10.5);
    let up = label.to_uppercase();
    let w = tracked_width(ui.ctx(), &up, &font, 1.3) + 24.0;
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 30.0), Sense::click());
    let hover = resp.hovered();
    let c = if just { OK } else if hover { MINT } else { INK2 };
    ui.painter().rect_stroke(rect, 0.0, Stroke::new(1.0, if just { alpha(OK, 140) } else if hover { MINT } else { LINE2 }), StrokeKind::Inside);
    paint_tracked(ui.painter(), Pos2::new(rect.left() + 12.0, rect.center().y - font.size * 0.68), &up, &font, c, 1.3);
    if resp.clicked() {
        ui.ctx().copy_text(text.to_string());
        ui.data_mut(|d| d.insert_temp(key, now));
    }
    if just {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(400));
    }
    if hover {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

// ------------------------------------------------------------------------------------------------ fields

/// A labelled text field in the style of the site: the label in mono capitals, then a sharp dark box that turns mint when focused.
pub fn field(ui: &mut Ui, title: &str, text: &mut String, hint: &str, password: bool, monospace: bool) -> Response {
    label(ui, title);
    ui.add_space(-4.0);
    ui.add(
        egui::TextEdit::singleline(text)
            .password(password)
            .hint_text(RichText::new(hint).color(DIM))
            .font(if monospace { mono(14.0) } else { sans(15.5) })
            .text_color(INK)
            .margin(Margin::symmetric(14, 12))
            .desired_width(f32::INFINITY),
    )
}

pub fn field_multiline(ui: &mut Ui, title: &str, text: &mut String, hint: &str, rows: usize, monospace: bool) -> Response {
    label(ui, title);
    ui.add_space(-4.0);
    ui.add(
        egui::TextEdit::multiline(text)
            .hint_text(RichText::new(hint).color(DIM))
            .font(if monospace { mono(14.0) } else { sans(15.5) })
            .text_color(INK)
            .margin(Margin::symmetric(14, 12))
            .desired_rows(rows)
            .desired_width(f32::INFINITY),
    )
}

/// A square check box in the style of the site.
pub fn checkbox(ui: &mut Ui, checked: &mut bool, text: &str) -> Response {
    let galley = ui.painter().layout(text.to_string(), sans(14.5), INK2, (ui.available_width() - 40.0).max(120.0));
    let h = galley.size().y.max(22.0);
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::click());
    if resp.clicked() {
        *checked = !*checked;
    }
    let b = Rect::from_min_size(rect.left_top() + vec2(0.0, (h - 20.0) / 2.0), vec2(20.0, 20.0));
    let hover = resp.hovered();
    ui.painter().rect(b, 0.0, if *checked { MINT } else { BG2 }, Stroke::new(1.0, if *checked || hover { MINT } else { LINE2 }), StrokeKind::Inside);
    if *checked {
        draw_check(ui.painter(), b.center(), 11.0, ACCENT_INK);
    }
    ui.painter().galley(rect.left_top() + vec2(32.0, 0.0), galley, INK2);
    if hover {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

/// A password-strength meter: four bars.
pub fn strength_meter(ui: &mut Ui, level: u8, text: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let colors = [BAD, WARN, MINT, OK];
        for i in 0..4u8 {
            let (r, _) = ui.allocate_exact_size(vec2(44.0, 4.0), Sense::hover());
            ui.painter().rect_filled(r, 0.0, if text.is_empty() { LINE } else if i <= level && !(level == 0 && i > 0) { colors[level as usize] } else { LINE });
        }
        ui.add_space(8.0);
        if !text.is_empty() {
            ui.label(RichText::new(text).font(sans(13.0)).color(MUTE));
        }
    });
}

/// A tag like "TESTNET": mono capitals in a thin box.
pub fn chip(ui: &mut Ui, text: &str, color: Color32) -> Response {
    let font = mono_semibold(10.0);
    let up = text.to_uppercase();
    let w = tracked_width(ui.ctx(), &up, &font, 1.3) + 18.0;
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 22.0), Sense::hover());
    ui.painter().rect(rect, 0.0, alpha(color, 16), Stroke::new(1.0, alpha(color, 90)), StrokeKind::Inside);
    paint_tracked(ui.painter(), Pos2::new(rect.left() + 9.0, rect.center().y - font.size * 0.68), &up, &font, color, 1.3);
    resp
}

pub fn dot(ui: &mut Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(10.0, 18.0), Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.5, color);
    ui.painter().circle_filled(rect.center(), 6.0, alpha(color, 36));
}
