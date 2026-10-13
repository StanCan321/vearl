//! Drawn effects of the wallet: the half-tone sphere of the Vearl mark (alive: light that circles, a slow wave, dots that rise towards the
//! pointer), a rotating arc for waiting, shimmering placeholders, a check mark that draws itself, a burst of dots, a progress ring.
use crate::anim::{ease_out, lerp, mix, noise};
use crate::theme::{self as t};
use eframe::egui::{self, pos2, vec2, Color32, Painter, Pos2, Response, Sense, Shape, Stroke, Ui};
use std::f32::consts::TAU;

/// The nacre gradient at `u` in 0..1.
pub fn nacre(u: f32) -> Color32 {
    let u = u.clamp(0.0, 1.0);
    if u < 0.5 {
        mix(t::MINT, t::LAV, u * 2.0)
    } else {
        mix(t::LAV, t::ROSE, (u - 0.5) * 2.0)
    }
}

/// One round dot as a small triangle fan in a mesh (much cheaper than a tessellated circle shape, and the sphere has over a thousand).
fn add_dot(mesh: &mut egui::epaint::Mesh, c: Pos2, r: f32, col: Color32) {
    // a solid disc plus a one-pixel feathered edge (a ring that fades out): smooth dots at any size
    let n: u32 = if r < 2.2 { 8 } else if r < 4.5 { 12 } else { 16 };
    let base = mesh.vertices.len() as u32;
    let clear = Color32::from_rgba_premultiplied(0, 0, 0, 0);
    mesh.colored_vertex(c, col);
    for i in 0..n {
        let a = i as f32 / n as f32 * TAU;
        mesh.colored_vertex(c + vec2(a.cos(), a.sin()) * (r - 0.35).max(0.1), col);
    }
    for i in 0..n {
        let a = i as f32 / n as f32 * TAU;
        mesh.colored_vertex(c + vec2(a.cos(), a.sin()) * (r + 0.75), clear);
    }
    for i in 0..n {
        let (a, b) = (base + 1 + i, base + 1 + (i + 1) % n);
        mesh.add_triangle(base, a, b);
        let (oa, ob) = (base + 1 + n + i, base + 1 + n + (i + 1) % n);
        mesh.add_triangle(a, oa, ob);
        mesh.add_triangle(a, ob, b);
    }
}

/// A sphere of dots lit from a point that slowly circles it; dots near `pointer` swell.
/// `opacity` 0..1; `now` is the time in seconds (the sphere is drawn as a function of it).
pub fn sphere(p: &Painter, center: Pos2, radius: f32, step: f32, opacity: f32, now: f64, pointer: Option<Pos2>) {
    let n = (radius / step).ceil() as i32;
    let a = now as f32 * 0.28;
    let (lx, ly) = (a.cos() * 0.62, a.sin() * 0.62 - 0.30);
    let mut mesh = egui::epaint::Mesh::default();
    for gy in -n..=n {
        for gx in -n..=n {
            let off = vec2(gx as f32 * step, gy as f32 * step);
            let d = off.length() / radius;
            if d >= 1.0 {
                continue;
            }
            let nz = (1.0 - d * d).sqrt();
            let light = (-(off.x / radius) * lx * 0.9 - (off.y / radius) * ly * 0.9 + nz * 0.62).clamp(0.0, 1.0);
            let wave = 1.0 + 0.10 * ((now as f32 * 1.1) + (gx as f32 * 0.33) + (gy as f32 * 0.27)).sin();
            let mut r = step * 0.5 * (0.12 + 0.95 * light) * wave;
            let pos = center + off;
            let mut boost = 0.0;
            if let Some(m) = pointer {
                let dist = (m - pos).length();
                boost = (1.0 - dist / 120.0).clamp(0.0, 1.0);
                boost = boost * boost * (3.0 - 2.0 * boost);
                r += step * 0.28 * boost;
            }
            if r < 0.6 {
                continue;
            }
            let u = ((off.x + off.y) / (2.0 * radius) * 0.5 + 0.5).clamp(0.0, 1.0);
            let col = mix(nacre(u), t::MINT_HOVER, boost * 0.6);
            let alpha = (opacity * 255.0 * (0.35 + 0.65 * light) + boost * 60.0).clamp(0.0, 255.0) as u8;
            add_dot(&mut mesh, pos, r, t::alpha(col, alpha));
        }
    }
    p.add(Shape::mesh(mesh));
}

/// A nacre arc that turns: waiting, working.
pub fn spinner(ui: &mut Ui, size: f32) -> Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(size, size), Sense::hover());
    spinner_at(ui.painter(), rect.center(), size * 0.38, size * 0.085, crate::anim::time(ui.ctx()));
    ui.ctx().request_repaint();
    resp
}

/// The dots of the arc: `radius` of the circle they turn on, `dot` the radius of the biggest dot.
pub fn spinner_at(p: &Painter, c: Pos2, radius: f32, dot: f32, now: f64) {
    let a0 = now as f32 * 4.6;
    for k in 0..26 {
        let ang = a0 - k as f32 * 0.115;
        let f = 1.0 - k as f32 / 26.0;
        p.circle_filled(c + vec2(ang.cos(), ang.sin()) * radius, dot * (0.45 + 0.55 * f), t::alpha(nacre(1.0 - f), (255.0 * f * f) as u8));
    }
}

/// A placeholder bar that shimmers while the real content loads.
pub fn skeleton(ui: &mut Ui, w: f32, h: f32) {
    let (rect, _) = ui.allocate_exact_size(vec2(w, h), Sense::hover());
    let now = crate::anim::time(ui.ctx()) as f32;
    let p = ui.painter().with_clip_rect(rect);
    p.rect_filled(rect, 0.0, t::alpha(t::LINE2, 90));
    let x = rect.left() + ((now * 0.7) % 1.7 - 0.35) * rect.width();
    let band = rect.width() * 0.35;
    let mut mesh = egui::epaint::Mesh::default();
    let (c0, c1) = (t::alpha(t::MINT, 0), t::alpha(t::MINT, 38));
    for (i, (dx, c)) in [(0.0, c0), (band * 0.5, c1), (band, c0)].into_iter().enumerate() {
        mesh.colored_vertex(pos2(x + dx, rect.top()), c);
        mesh.colored_vertex(pos2(x + dx, rect.bottom()), c);
        if i > 0 {
            let k = (i as u32 - 1) * 2;
            mesh.add_triangle(k, k + 1, k + 2);
            mesh.add_triangle(k + 1, k + 3, k + 2);
        }
    }
    p.add(Shape::mesh(mesh));
    ui.ctx().request_repaint();
}

/// A check mark that draws itself as `prog` goes 0 -> 1.
pub fn check(p: &Painter, center: Pos2, size: f32, prog: f32, color: Color32, width: f32) {
    let k = size / 2.0;
    let (a, b, c) = (center + vec2(-k, 0.05 * k), center + vec2(-0.30 * k, 0.72 * k), center + vec2(k, -0.62 * k));
    let (l1, l2) = ((b - a).length(), (c - b).length());
    let total = (l1 + l2) * prog.clamp(0.0, 1.0);
    let s = Stroke::new(width, color);
    if total > 0.0 {
        let e1 = a + (b - a) * (total.min(l1) / l1);
        p.line_segment([a, e1], s);
        p.circle_filled(a, width / 2.0, color);
        if total > l1 {
            let e2 = b + (c - b) * ((total - l1) / l2);
            p.line_segment([b, e2], s);
            p.circle_filled(e2, width / 2.0, color);
        }
    }
}

/// A ring of `frac` (0..1) filled with the nacre gradient, over a faint track. Starts at the top.
pub fn ring(p: &Painter, center: Pos2, radius: f32, width: f32, frac: f32) {
    let steps = 120;
    for i in 0..steps {
        let a0 = -TAU / 4.0 + TAU * i as f32 / steps as f32;
        let a1 = -TAU / 4.0 + TAU * (i as f32 + 1.05) / steps as f32;
        let s = |a: f32| center + vec2(a.cos(), a.sin()) * radius;
        let on = (i as f32 + 0.5) / steps as f32 <= frac;
        p.line_segment([s(a0), s(a1)], Stroke::new(width, if on { nacre(i as f32 / steps as f32) } else { t::alpha(t::LINE2, 140) }));
    }
}

/// A burst of dots flying out of `origin`, `secs` after it started. Gone after about 1.8 seconds.
pub fn burst(p: &Painter, origin: Pos2, secs: f32) {
    const LIFE: f32 = 1.8;
    if !(0.0..LIFE).contains(&secs) {
        return;
    }
    for i in 0..90u64 {
        let ang = noise(i, 1, 0) * TAU;
        let speed = lerp(90.0, 340.0, noise(i, 2, 0));
        let life = lerp(0.9, LIFE, noise(i, 3, 0));
        if secs > life {
            continue;
        }
        let f = secs / life;
        let travel = ease_out(f) * speed;
        let pos = origin + vec2(ang.cos(), ang.sin()) * travel + vec2(0.0, 60.0 * f * f);
        let r = lerp(1.4, 4.2, noise(i, 4, 0)) * (1.0 - f * 0.7);
        p.circle_filled(pos, r, t::alpha(nacre(noise(i, 5, 0)), (255.0 * (1.0 - f * f)) as u8));
    }
}

/// A soft glow behind a point (a round halo in the nacre colours).
pub fn halo(p: &Painter, center: Pos2, radius: f32, strength: f32, now: f64) {
    for k in (1..=6).rev() {
        let f = k as f32 / 6.0;
        let wobble = 1.0 + 0.04 * (now as f32 * 1.4 + k as f32).sin();
        p.circle_filled(center, radius * f * wobble, t::alpha(nacre(0.5 + 0.5 * (now as f32 * 0.3 + f).sin()), (strength * 255.0 * (1.0 - f) * 0.35) as u8));
    }
}

/// The connection dot: it sends out a ring every four seconds, and the screen is only redrawn while the ring moves.
pub fn pulse_dot_timed(ctx: &egui::Context, p: &Painter, center: Pos2, color: Color32) {
    let now = crate::anim::fx_time(ctx);
    let ph = (now % 4.0) as f32;
    p.circle_filled(center, 3.5, color);
    if crate::anim::reduced() {
        return;
    }
    if ph < 1.2 {
        let k = ph / 1.2;
        p.circle_filled(center, 3.5 + 5.5 * k, t::alpha(color, (70.0 * (1.0 - k)) as u8));
        ctx.request_repaint();
    } else {
        ctx.request_repaint_after(std::time::Duration::from_secs_f64(4.0 - ph as f64 + 0.02));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gradient_runs_mint_lavender_rose() {
        assert_eq!(nacre(0.0), t::MINT);
        assert_eq!(nacre(0.5), t::LAV);
        assert_eq!(nacre(1.0), t::ROSE);
        assert_eq!(nacre(-5.0), t::MINT);
    }
}
