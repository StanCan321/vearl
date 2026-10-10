//! vearl-l2: level 2 tools (Monte Carlo, rendering) without a network.
//!   vearl-l2 render WIDTH HEIGHT SPP OUTPUT.ppm     renders the demonstration scene (all the chunks, locally)
//!   vearl-l2 render2 WIDTH HEIGHT SPP BOUNCES ANGLE CAM_HEIGHT OUTPUT.ppm [gpu]   renders the Render2 scene (reflections, soft shadows, halo) on all the cores
//!   vearl-l2 pi NB_CHUNKS                              estimates pi by Monte Carlo (indicator of the quarter disc)
//!   vearl-l2 vectors                                   kernel test vectors (JSON), verified by tools/l2_ref.py
use vearl_core::l2::*;
use serde_json::json;

fn hx(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("render") => {
            let (w, h, spp): (u16, u16, u8) = (a[1].parse().unwrap(), a[2].parse().unwrap(), a[3].parse().unwrap());
            let k = Kernel::Render(demo_scene(w, h, spp, [42; 32]));
            k.validate(k.forced_chunks().unwrap()).expect("scene");
            let chunks = k.forced_chunks().unwrap();
            let tiles_x = w as usize / TILE;
            let mut img = vec![0u8; w as usize * h as usize * 3];
            let t0 = std::time::Instant::now();
            for c in 0..chunks {
                let t = k.chunk(c);
                let (tx, ty) = (c as usize % tiles_x, c as usize / tiles_x);
                for y in 0..TILE {
                    for x in 0..TILE {
                        let dst = ((ty * TILE + y) * w as usize + tx * TILE + x) * 3;
                        img[dst..dst + 3].copy_from_slice(&t[(y * TILE + x) * 3..(y * TILE + x) * 3 + 3]);
                    }
                }
            }
            let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
            ppm.extend_from_slice(&img);
            std::fs::write(&a[4], ppm).unwrap();
            println!("{chunks} tiles rendered in {:.2} s (cost {} units per tile) -> {}", t0.elapsed().as_secs_f64(), k.cost(), a[4]);
        }
        Some("render2") => {
            let (w, h, spp, bounces): (u16, u16, u8, u8) = (a[1].parse().unwrap(), a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap());
            let (angle, rise): (f64, f64) = (a[5].parse().unwrap(), a[6].parse().unwrap());
            let k = Kernel::Render2(vearl_core::render2::scene_orb(w, h, spp, bounces, angle, rise, [42; 32]));
            let chunks = k.forced_chunks().unwrap();
            k.validate(chunks).expect("scene");
            let on_gpu = a.get(8).map(String::as_str) == Some("gpu");
            vearl_core::work::set_gpu_exact(on_gpu);
            let t0 = std::time::Instant::now();
            let outs = compute_all(&k, chunks);
            let secs = t0.elapsed().as_secs_f64();
            let tiles_x = w as usize / TILE;
            let mut img = vec![0u8; w as usize * h as usize * 3];
            for (c, t) in outs.iter().enumerate() {
                let (tx, ty) = (c % tiles_x, c / tiles_x);
                for y in 0..TILE {
                    for x in 0..TILE {
                        let dst = ((ty * TILE + y) * w as usize + tx * TILE + x) * 3;
                        img[dst..dst + 3].copy_from_slice(&t[(y * TILE + x) * 3..(y * TILE + x) * 3 + 3]);
                    }
                }
            }
            let mut ppm = format!("P6
{w} {h}
255
").into_bytes();
            ppm.extend_from_slice(&img);
            std::fs::write(&a[7], ppm).unwrap();
            println!("{chunks} tiles rendered in {secs:.2} s on {} (cost bound {} units per tile) -> {}", if on_gpu { "GPU" } else { "CPU" }, k.cost(), a[7]);
        }
        Some("render2-watch") => {
            // vearl-l2 render2-watch WIDTH HEIGHT SPP BOUNCES ANGLE CAM_HEIGHT OUTPUT.ppm [gpu]
            let (w, h, spp, bounces): (u16, u16, u8, u8) = (a[1].parse().unwrap(), a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap());
            let (angle, rise): (f64, f64) = (a[5].parse().unwrap(), a[6].parse().unwrap());
            let k = Kernel::Render2(vearl_core::render2::scene_watch(w, h, spp, bounces, angle, rise, [42; 32]));
            let chunks = k.forced_chunks().unwrap();
            k.validate(chunks).expect("scene");
            let on_gpu = a.get(8).map(String::as_str) == Some("gpu");
            vearl_core::work::set_gpu_exact(on_gpu);
            let t0 = std::time::Instant::now();
            let outs = compute_all(&k, chunks);
            let secs = t0.elapsed().as_secs_f64();
            let tiles_x = w as usize / TILE;
            let mut img = vec![0u8; w as usize * h as usize * 3];
            for (c, t) in outs.iter().enumerate() {
                let (tx, ty) = (c % tiles_x, c / tiles_x);
                for y in 0..TILE {
                    let dst = ((ty * TILE + y) * w as usize + tx * TILE) * 3;
                    img[dst..dst + TILE * 3].copy_from_slice(&t[y * TILE * 3..(y + 1) * TILE * 3]);
                }
            }
            let mut ppm = format!("P6
{w} {h}
255
").into_bytes();
            ppm.extend_from_slice(&img);
            std::fs::write(&a[7], ppm).unwrap();
            println!("{chunks} tiles (watch) rendered in {secs:.2} s on {} (cost bound {} units per tile) -> {}", if on_gpu { "GPU" } else { "CPU" }, k.cost(), a[7]);
        }
        Some("render3") => {
            // vearl-l2 render3 SCENE WIDTH HEIGHT SPP_PER_PASS PASSES OUTPUT.ppm [gpu]
            let (w, h, spp, passes): (u16, u16, u8, u16) = (a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap(), a[5].parse().unwrap());
            let params = match a[1].as_str() {
                "watch" => vearl_core::render3::scene3_watch(w, h, spp, passes, [42; 32]),
                "mini" => vearl_core::render3::scene3_mini(w, h, spp, passes, 6, [42; 32]),
                "cocktail" => vearl_core::render3::scene3_cocktail(w, h, spp, passes, [42; 32]),
                "orb" => vearl_core::render3::scene3_orb(w, h, spp, passes, [42; 32]),
                "gallery" => vearl_core::render3::scene3_gallery(w, h, spp, passes, [42; 32]),
                other => panic!("unknown scene '{other}' (watch | mini | cocktail | orb | gallery)"),
            };
            let k = Kernel::Render3(params.clone());
            let chunks = k.forced_chunks().unwrap();
            k.validate(chunks).expect("scene");
            let on_gpu = a.get(7).map(String::as_str) == Some("gpu");
            vearl_core::work::set_gpu_exact(on_gpu);
            let t0 = std::time::Instant::now();
            let outs = compute_all(&k, chunks);
            let secs = t0.elapsed().as_secs_f64();
            let img = vearl_core::render3::assemble(&params, &outs, 1.1);
            let mut ppm = format!("P6
{w} {h}
255
").into_bytes();
            ppm.extend_from_slice(&img);
            std::fs::write(&a[6], ppm).unwrap();
            println!("{chunks} chunks ({} x {spp} samples) in {secs:.1} s on {} (cost bound {} units per chunk) -> {}", passes, if on_gpu { "GPU" } else { "CPU" }, k.cost(), a[6]);
        }
        Some("render2-film") => {
            // vearl-l2 render2-film WIDTH HEIGHT SPP BOUNCES FRAMES FOLDER [gpu]: a camera loop around the scene, one image per PPM file
            let (w, h, spp, bounces, frames): (u16, u16, u8, u8, u32) = (a[1].parse().unwrap(), a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap(), a[5].parse().unwrap());
            let dir = std::path::PathBuf::from(&a[6]);
            std::fs::create_dir_all(&dir).unwrap();
            let on_gpu = a.get(7).map(String::as_str) == Some("gpu");
            vearl_core::work::set_gpu_exact(on_gpu);
            let t0 = std::time::Instant::now();
            for i in 0..frames {
                let u = i as f64 / frames as f64;
                let tau = std::f64::consts::TAU;
                let p = vearl_core::render2::scene_orb_anim(w, h, spp, bounces, 360.0 * u + 8.0, 1.45 + 0.75 * (tau * u).sin(), 5.7 + 0.9 * (tau * u).cos(), tau * u, [42; 32]);
                let k = Kernel::Render2(p);
                let chunks = k.forced_chunks().unwrap();
                k.validate(chunks).expect("scene");
                let outs = compute_all(&k, chunks);
                let tiles_x = w as usize / TILE;
                let mut img = vec![0u8; w as usize * h as usize * 3];
                for (c, t) in outs.iter().enumerate() {
                    let (tx, ty) = (c % tiles_x, c / tiles_x);
                    for y in 0..TILE {
                        let dst = ((ty * TILE + y) * w as usize + tx * TILE) * 3;
                        img[dst..dst + TILE * 3].copy_from_slice(&t[y * TILE * 3..(y + 1) * TILE * 3]);
                    }
                }
                let mut ppm = format!("P6
{w} {h}
255
").into_bytes();
                ppm.extend_from_slice(&img);
                std::fs::write(dir.join(format!("frame_{i:04}.ppm")), ppm).unwrap();
            }
            println!("{frames} images {w}x{h} (spp {spp}, {bounces} bounces) in {:.1} s on {}", t0.elapsed().as_secs_f64(), if on_gpu { "GPU" } else { "CPU" });
        }
        Some("pi") => {
            let n: u32 = a[1].parse().unwrap();
            let k = Kernel::MonteCarlo(McParams { seed: [1; 32], dim: 2, samples: 4096, out: 6, program: mc_pi_program() });
            let mut sum = 0i128;
            for c in 0..n {
                sum += i128::from_le_bytes(k.chunk(c)[..16].try_into().unwrap());
            }
            let m = sum as f64 / (1i128 << 32) as f64 / (n as f64 * 4096.0);
            println!("pi ~ {:.5} ({} samples)", 4.0 * m, n as u64 * 4096);
        }
        Some("vectors") => {
            let sqrt_prog = |imm: i64| vec![Instr { op: 0, a: 0, b: 0, c: 0, imm }, Instr { op: 1, a: 1, b: 0, c: 0, imm: 0 }, Instr { op: 4, a: 2, b: 0, c: 1, imm: 0 }, Instr { op: 9, a: 3, b: 2, c: 0, imm: 0 }, Instr { op: 2, a: 4, b: 3, c: 0, imm: 0 }, Instr { op: 7, a: 5, b: 4, c: 0, imm: 0 }, Instr { op: 10, a: 6, b: 5, c: 3, imm: 0 }, Instr { op: 11, a: 7, b: 6, c: 5, imm: 3 }, Instr { op: 2, a: 8, b: 7, c: 4, imm: 0 }];
            let mcs = vec![
                McParams { seed: [1; 32], dim: 2, samples: 512, out: 6, program: mc_pi_program() },
                McParams { seed: [9; 32], dim: 1, samples: 300, out: 8, program: sqrt_prog(-3_000_000_000) },
                McParams { seed: [5; 32], dim: 3, samples: 64, out: 8, program: sqrt_prog(i64::MAX / 2) },
            ];
            let mut out = vec![];
            for p in mcs {
                let k = Kernel::MonteCarlo(p.clone());
                let chunks: Vec<serde_json::Value> = [0u32, 1, 7].iter().map(|c| json!({"i": c, "out": hx(&k.chunk(*c))})).collect();
                out.push(json!({"kernel": "mc", "params": serde_json::to_value(&p).unwrap(), "chunks": chunks}));
            }
            for (w, h, spp, seed) in [(32u16, 32u16, 1u8, 3u8), (32, 32, 3, 4), (48, 32, 2, 5)] {
                let p = demo_scene(w, h, spp, [seed; 32]);
                let k = Kernel::Render(p.clone());
                let n = k.forced_chunks().unwrap();
                let chunks: Vec<serde_json::Value> = [0u32, 1, n - 1].iter().map(|c| json!({"i": c, "out": hx(&k.chunk(*c))})).collect();
                out.push(json!({"kernel": "render", "params": serde_json::to_value(&p).unwrap(), "chunks": chunks}));
            }
            use vearl_core::render2::{scene_orb, scene_prism, scene_watch};
            let r2: Vec<vearl_core::render2::RenderParams2> = vec![scene_orb(48, 32, 1, 2, 20.0, 1.3, [11; 32]), scene_orb(32, 32, 3, 1, 140.0, 0.9, [12; 32]), scene_prism(48, 32, 2, 2, [13; 32]), scene_prism(32, 32, 1, 0, [14; 32]), scene_watch(48, 32, 1, 1, 30.0, 2.4, [15; 32])];
            for p in r2 {
                let k = Kernel::Render2(p.clone());
                k.validate(k.forced_chunks().unwrap()).expect("scene render2");
                let n = k.forced_chunks().unwrap();
                let chunks: Vec<serde_json::Value> = [0u32, n - 1].iter().map(|c| json!({"i": c, "out": hx(&k.chunk(*c))})).collect();
                out.push(json!({"kernel": "render2", "params": serde_json::to_value(&p).unwrap(), "chunks": chunks}));
            }
            use vearl_core::render3::{scene3_mini, scene3_textures};
            for p in [scene3_mini(32, 32, 1, 2, 3, [21; 32]), scene3_mini(32, 16, 2, 1, 5, [22; 32]), scene3_textures(32, 32, 1, 1, 4, [23; 32]), scene3_textures(48, 16, 2, 1, 3, [24; 32])] {
                let k = Kernel::Render3(p.clone());
                let n = k.forced_chunks().unwrap();
                k.validate(n).expect("scene render3");
                let chunks: Vec<serde_json::Value> = [0u32, n - 1].iter().map(|c| json!({"i": c, "out": hx(&k.chunk(*c))})).collect();
                out.push(json!({"kernel": "render3", "params": serde_json::to_value(&p).unwrap(), "chunks": chunks}));
            }
            println!("{}", serde_json::to_string_pretty(&json!({ "cases": out })).unwrap());
        }
        _ => eprintln!("usage: vearl-l2 (render W H SPP OUT.ppm | pi N | vectors)"),
    }
}
