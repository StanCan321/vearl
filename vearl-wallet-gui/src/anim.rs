//! Motion: easing, colour mixing, fade-and-rise reveals, hover fades, a count-up for amounts. Everything is a function of time (egui is
//! immediate-mode), so nothing is stored but a start time, and a frame is only requested while something is moving.
use eframe::egui::{self, emath::TSTransform, vec2, Color32, Context, Id, Ui};
use std::sync::atomic::{AtomicBool, Ordering};

static REDUCE: AtomicBool = AtomicBool::new(false);

/// Settings > no animations: everything jumps to its end state at once.
pub fn set_reduce_motion(on: bool) {
    REDUCE.store(on, Ordering::Relaxed);
}

pub fn reduced() -> bool {
    REDUCE.load(Ordering::Relaxed)
}

pub fn ease_out(t: f32) -> f32 {
    egui::emath::easing::cubic_out(t.clamp(0.0, 1.0))
}

pub fn ease_in_out(t: f32) -> f32 {
    egui::emath::easing::cubic_in_out(t.clamp(0.0, 1.0))
}

/// Overshoots a little then settles (the "pop" of a check mark or a card).
pub fn ease_out_back(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0) - 1.0;
    let c = 1.70158;
    t * t * ((c + 1.0) * t + c) + 1.0
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_unmultiplied(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()), f(a.a(), b.a()))
}

pub fn time(ctx: &Context) -> f64 {
    ctx.input(|i| i.time)
}

/// The clock of the endless effects (the sphere, the glints): frozen when motion is reduced.
pub fn fx_time(ctx: &Context) -> f64 {
    if reduced() {
        0.0
    } else {
        time(ctx)
    }
}

/// 0 -> 1 over `dur` seconds, starting `delay` seconds after `t0`. Asks for the next frame while it is moving.
pub fn progress(ctx: &Context, t0: f64, delay: f32, dur: f32) -> f32 {
    if reduced() {
        return 1.0;
    }
    let p = ((time(ctx) - t0) as f32 - delay) / dur.max(0.001);
    if p < 1.0 {
        ctx.request_repaint();
    }
    p.clamp(0.0, 1.0)
}

/// A hover (or any on/off state) that fades instead of jumping.
pub fn fade(ctx: &Context, id: Id, on: bool, secs: f32) -> f32 {
    ctx.animate_bool_with_time_and_easing(id, on, if reduced() { 0.0 } else { secs }, egui::emath::easing::cubic_out)
}

/// Fade-and-rise: the content appears `delay` seconds after the scene started `t0`, rising by `rise` pixels while it fades in.
pub fn reveal<R>(ui: &mut Ui, t0: f64, delay: f32, rise: f32, add: impl FnOnce(&mut Ui) -> R) -> R {
    let p = ease_out(progress(ui.ctx(), t0, delay, 0.5));
    if p >= 1.0 {
        return add(ui);
    }
    let layer = ui.layer_id();
    let start = ui.ctx().graphics_mut(|g| g.entry(layer).next_idx());
    let out = ui
        .scope(|ui| {
            ui.multiply_opacity(p);
            add(ui)
        })
        .inner;
    let end = ui.ctx().graphics_mut(|g| g.entry(layer).next_idx());
    ui.ctx().graphics_mut(|g| g.entry(layer).transform_range(start, end, TSTransform::from_translation(vec2(0.0, (1.0 - p) * rise))));
    out
}

// ------------------------------------------------------------------------------------------------ scenes (staggered cards)

/// Starts a scene: the cards drawn afterwards appear one after the other, `t0` being the moment the screen changed.
pub fn scene_begin(ctx: &Context, t0: f64) {
    ctx.data_mut(|d| d.insert_temp(Id::new("scene"), (t0, 0u32)));
}

/// The next card of the scene: (start time, delay in seconds).
pub fn scene_next(ctx: &Context) -> (f64, f32) {
    ctx.data_mut(|d| {
        let (t0, n): (f64, u32) = d.get_temp(Id::new("scene")).unwrap_or((-100.0, 0));
        d.insert_temp(Id::new("scene"), (t0, n + 1));
        (t0, 0.05 + n as f32 * 0.07)
    })
}

/// The time at which something was first drawn (so it can animate in once).
pub fn first_seen(ctx: &Context, id: Id) -> f64 {
    let now = time(ctx);
    ctx.data_mut(|d| {
        let key = id.with("first_seen");
        match d.get_temp::<f64>(key) {
            Some(t) => t,
            None => {
                d.insert_temp(key, now);
                now
            }
        }
    })
}

// ------------------------------------------------------------------------------------------------ amounts

/// A number that counts up (or down) to its new value instead of jumping. Returns the value to display.
pub fn count_to(ctx: &Context, id: Id, target: f64, secs: f32) -> f64 {
    if reduced() {
        return target;
    }
    #[derive(Clone, Copy)]
    struct St {
        from: f64,
        to: f64,
        t0: f64,
    }
    let now = time(ctx);
    let st: Option<St> = ctx.data(|d| d.get_temp(id));
    let (cur, st) = match st {
        None => (target, St { from: target, to: target, t0: now }),
        Some(s) => {
            let p = ease_out(((now - s.t0) as f32) / secs);
            let cur = s.from + (s.to - s.from) * p as f64;
            if (s.to - target).abs() > f64::EPSILON {
                (cur, St { from: cur, to: target, t0: now })
            } else {
                (cur, s)
            }
        }
    };
    ctx.data_mut(|d| d.insert_temp(id, st));
    if (st.to - cur).abs() > 0.5 {
        ctx.request_repaint();
    }
    cur
}

// ------------------------------------------------------------------------------------------------ small deterministic noise

/// A repeatable pseudo-random number in 0..1 from three integers (for the scramble effect and the particles).
pub fn noise(a: u64, b: u64, c: u64) -> f32 {
    let mut h = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ c.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^= h >> 33;
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// The word `text` while it "decrypts": letters resolve from left to right as `p` goes 0 -> 1, the others flicker.
pub fn scramble(text: &str, p: f32, seed: u64, now: f64) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let n = text.chars().count().max(1);
    let tick = (now * 24.0) as u64;
    text.chars()
        .enumerate()
        .map(|(k, c)| {
            if p >= 1.0 || (k as f32 + 1.0) / n as f32 <= p {
                c
            } else {
                CHARS[(noise(seed, k as u64, tick) * CHARS.len() as f32) as usize % CHARS.len()] as char
            }
        })
        .collect()
}

/// A text typed out: how much of it is visible `secs` after the start, at `cps` characters per second.
pub fn typed(text: &str, secs: f32, cps: f32) -> String {
    text.chars().take((secs.max(0.0) * cps) as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn easing_is_bounded_and_monotone_at_the_ends() {
        for f in [ease_out, ease_in_out] {
            assert_eq!(f(0.0), 0.0);
            assert!((f(1.0) - 1.0).abs() < 1e-6);
            assert_eq!(f(-3.0), 0.0);
            assert!((f(9.0) - 1.0).abs() < 1e-6);
        }
        assert!((ease_out_back(1.0) - 1.0).abs() < 1e-6 && ease_out_back(0.0).abs() < 1e-5);
    }

    #[test]
    fn colours_mix_at_the_ends() {
        let (a, b) = (Color32::from_rgb(0, 0, 0), Color32::from_rgb(200, 100, 50));
        assert_eq!(mix(a, b, 0.0), a);
        assert_eq!(mix(a, b, 1.0), b);
        assert_eq!(mix(a, b, 0.5), Color32::from_rgb(100, 50, 25));
    }

    #[test]
    fn scramble_resolves_left_to_right() {
        assert_eq!(scramble("abandon", 1.0, 1, 0.0), "abandon");
        let half = scramble("abandon", 0.5, 1, 0.0);
        assert_eq!(half.chars().count(), 7);
        assert!(half.starts_with("aba"));
        assert_eq!(scramble("x", 0.0, 1, 0.0).chars().count(), 1);
    }

    #[test]
    fn noise_is_repeatable_and_in_range() {
        for i in 0..200 {
            let v = noise(i, 7, 9);
            assert!((0.0..1.0).contains(&v));
            assert_eq!(v, noise(i, 7, 9));
        }
    }

    #[test]
    fn typing_reveals_progressively() {
        assert_eq!(typed("abcdef", 0.0, 10.0), "");
        assert_eq!(typed("abcdef", 0.3, 10.0), "abc");
        assert_eq!(typed("abcdef", 5.0, 10.0), "abcdef");
        assert_eq!(typed("abcdef", -1.0, 10.0), "");
    }
}
