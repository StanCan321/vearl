//! Level 2 (Monte Carlo, rendering): deterministic kernels, job life cycle, fraud proof on a chunk, data
//! availability, access controls. Weaker guarantee than level 1 (audit by sampling): docs/10.
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::l2::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;

const P: WorkParams = WorkParams::TESTNET;

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn new_chain() -> Chain {
    let alloc: Vec<(PubKey, u64)> = [10u8, 11, 12, 13].iter().map(|s| (pubkey(&key(*s)), 1000 * COIN)).collect();
    let mut c = Chain::genesis(P, &alloc, 1_000, u64::MAX / 64);
    c.state.timing = Timing { challenge_secs: 60, serve_secs: 100 };
    c
}

fn mine(c: &mut Chain, txs: Vec<SignedTx>) -> Result<(), String> {
    let t = c.tip_ts() + BLOCK_TIME;
    let out = c.mine(&key(12), WorkClass::Native, None, txs, t)?;
    c.validate_and_apply(out.block)
}

fn ltx(c: &Chain, who: u8, m: L2Tx) -> SignedTx {
    make_tx(c, &key(who), 0, TxKind::L2(m))
}

fn mc(chunks_seed: u8) -> Kernel {
    Kernel::MonteCarlo(McParams { seed: [chunks_seed; 32], dim: 2, samples: 512, out: 6, program: mc_pi_program() })
}

fn instr(op: u8, a: u8, b: u8, c: u8, imm: i64) -> Instr {
    Instr { op, a, b, c, imm }
}

// ------------------------------------------------------------ kernels

#[test]
fn monte_carlo_estimates_pi_and_is_deterministic() {
    let k = mc(1);
    assert_eq!(k.chunk(3), k.chunk(3), "deterministic");
    assert_ne!(k.chunk(3), k.chunk(4), "independent chunks");
    assert_ne!(mc(1).chunk(0), mc(2).chunk(0), "the seed changes the draws");
    let (mut sum, mut n) = (0i128, 0u64);
    for i in 0..128u32 {
        let o = k.chunk(i);
        sum += i128::from_le_bytes(o[..16].try_into().unwrap());
        n += 512;
    }
    let mean = sum as f64 / (1i128 << 32) as f64 / n as f64; // = pi/4
    assert!((4.0 * mean - std::f64::consts::PI).abs() < 0.03, "estimated pi = {}", 4.0 * mean);
    // sum of squares: the indicator is 0 or 1, so sum of squares = sum
    let o = k.chunk(0);
    assert_eq!(o[..16], o[16..], "indicator: sum(v^2) = sum(v)");
}

#[test]
fn monte_carlo_vm_has_exact_integer_semantics() {
    let run = |program: Vec<Instr>, out: u8| {
        let k = Kernel::MonteCarlo(McParams { seed: [0; 32], dim: 1, samples: 1, out, program });
        let o = k.chunk(0);
        i128::from_le_bytes(o[..16].try_into().unwrap()) as i64
    };
    let one = MC_ONE;
    // sqrt(4.0) = 2.0 exactly; sqrt of a negative = 0
    assert_eq!(run(vec![instr(0, 0, 0, 0, 4 * one), instr(9, 1, 0, 0, 0)], 1), 2 * one);
    assert_eq!(run(vec![instr(0, 0, 0, 0, -4 * one), instr(9, 1, 0, 0, 0)], 1), 0);
    // Q32 multiplication with arithmetic shift (rounding toward -inf): (-1 ulp) * 0.5 = -1 ulp/2 -> -1
    assert_eq!(run(vec![instr(0, 0, 0, 0, -1), instr(0, 1, 0, 0, one / 2), instr(4, 2, 0, 1, 0)], 2), -1);
    // saturation
    assert_eq!(run(vec![instr(0, 0, 0, 0, i64::MAX), instr(0, 1, 0, 0, 5), instr(2, 2, 0, 1, 0)], 2), i64::MAX);
    assert_eq!(run(vec![instr(0, 0, 0, 0, i64::MIN), instr(7, 1, 0, 0, 0)], 1), i64::MAX);
    // selection and comparison
    assert_eq!(run(vec![instr(0, 0, 0, 0, 3), instr(0, 1, 0, 0, 7), instr(10, 2, 0, 1, 0)], 2), one);
    assert_eq!(run(vec![instr(0, 0, 0, 0, one), instr(0, 1, 0, 0, 111), instr(0, 2, 0, 0, 222), instr(11, 3, 0, 1, 2)], 3), 111);
    assert_eq!(run(vec![instr(0, 0, 0, 0, 0), instr(0, 1, 0, 0, 111), instr(0, 2, 0, 0, 222), instr(11, 3, 0, 1, 2)], 3), 222);
    // isqrt128 exact
    for x in [0u128, 1, 2, 3, 4, 15, 16, 17, (1 << 64) - 1, 1 << 90, (1u128 << 95) + 12345] {
        let r = isqrt128(x);
        assert!(r * r <= x && (r + 1) * (r + 1) > x, "isqrt128({x})");
    }
}

#[test]
fn render_is_deterministic_and_draws_the_scene() {
    let scene = demo_scene(64, 64, 1, [7; 32]);
    let k = Kernel::Render(scene.clone());
    assert_eq!(k.forced_chunks(), Some(16));
    let tile = k.chunk(5);
    assert_eq!(tile.len(), 16 * 16 * 3);
    assert_eq!(tile, k.chunk(5));
    let img: Vec<Vec<u8>> = (0..16).map(|i| k.chunk(i)).collect();
    let distinct: std::collections::BTreeSet<&[u8]> = img.iter().flat_map(|t| t.chunks(3)).collect();
    assert!(distinct.len() > 40, "the image contains many colours ({})", distinct.len());
    // the sky (top of the image) is bluer than red; the floor and the objects exist
    let top_px = &img[1][..3];
    assert!(top_px[2] > top_px[0], "bluish sky {top_px:?}");
    // anti-aliasing: spp=3 changes some edge pixels, stays deterministic
    let k3 = Kernel::Render(demo_scene(64, 64, 3, [7; 32]));
    assert_eq!(k3.chunk(6), k3.chunk(6));
    assert!((0..16).any(|i| k3.chunk(i) != k.chunk(i)), "spp>1 averages the shifted samples");
    assert!(k3.cost() == 3 * k.cost());
}

#[test]
fn render2_is_deterministic_rich_and_its_limits_hold() {
    use vearl_core::render2::*;
    let p = scene_orb(64, 48, 2, 2, 25.0, 1.2, [3; 32]);
    let k = Kernel::Render2(p.clone());
    assert_eq!(k.forced_chunks(), Some(12));
    k.validate(12).unwrap();
    let tiles: Vec<Vec<u8>> = (0..12).map(|i| k.chunk(i)).collect();
    assert!(tiles.iter().all(|t| t.len() == 768));
    assert_eq!(tiles[5], k.chunk(5), "deterministic");
    let distinct: std::collections::BTreeSet<&[u8]> = tiles.iter().flat_map(|t| t.chunks(3)).collect();
    assert!(distinct.len() > 300, "the image is rich ({} colours)", distinct.len());
    // moving the camera changes the image
    let moved = Kernel::Render2(scene_orb(64, 48, 2, 2, 55.0, 1.2, [3; 32]));
    assert!((0..12).any(|i| moved.chunk(i) != tiles[i as usize]));
    // the seed only moves the sampling offsets (spp > 1)
    let other_seed = Kernel::Render2(scene_orb(64, 48, 2, 2, 25.0, 1.2, [4; 32]));
    assert!((0..12).any(|i| other_seed.chunk(i) != tiles[i as usize]));
    // reflections change the picture, and the cost grows with the bounces
    let flat = Kernel::Render2(scene_orb(64, 48, 2, 0, 25.0, 1.2, [3; 32]));
    assert!((0..12).any(|i| flat.chunk(i) != tiles[i as usize]));
    assert!(k.cost() > flat.cost());
    // bounds
    let mut bad = p.clone();
    bad.bounces = 3;
    assert!(Kernel::Render2(bad).validate(12).is_err(), "at most 2 bounces");
    let mut bad = p.clone();
    bad.prims = vec![p.prims[1].clone(); 33];
    assert!(Kernel::Render2(bad).validate(12).is_err(), "at most 32 shapes");
    let watch = Kernel::Render2(scene_watch(64, 48, 2, 2, 30.0, 2.4, [3; 32]));
    watch.validate(12).unwrap();
    assert!(watch.chunk(4) != watch.chunk(9), "the watch scene draws something different in different tiles");
    assert!(Kernel::Render2(scene_watch(1920, 1088, 4, 2, 30.0, 2.4, [0; 32])).validate(8160).is_err(), "27 shapes at 4 samples and 2 bounces exceed the cost bound");
    assert!(Kernel::Render2(scene_watch(1920, 1088, 3, 2, 30.0, 2.4, [0; 32])).validate(8160).is_ok());
    assert!(Kernel::Render2(p.clone()).validate(11).is_err(), "one chunk per tile");
    assert!(Kernel::Render2(scene_orb(65, 48, 2, 2, 25.0, 1.2, [3; 32])).validate(12).is_err());
    let mut bad = p.clone();
    bad.cam[0] = i32::MAX;
    assert!(Kernel::Render2(bad).validate(12).is_err(), "coordinates out of range");
    // a full HD frame fits in the chunk limit and in the cost bound
    let hd = Kernel::Render2(scene_orb(1920, 1088, 4, 2, 25.0, 1.2, [0; 32]));
    assert_eq!(hd.forced_chunks(), Some(8160));
    hd.validate(8160).unwrap();
    assert!(hd.cost() <= MAX_CHUNK_COST);
    assert!(Kernel::Render2(scene_orb(2064, 1088, 1, 0, 25.0, 1.2, [0; 32])).validate(8_000).is_err(), "side at most 2048");
}

#[test]
fn render3_is_deterministic_bounded_and_its_chunks_are_passes_of_tiles() {
    use vearl_core::render3::*;
    let p = scene3_mini(32, 32, 2, 3, 4, [9; 32]);
    let k = Kernel::Render3(p.clone());
    assert_eq!(k.forced_chunks(), Some(12), "4 tiles x 3 passes");
    k.validate(12).unwrap();
    let a = k.chunk(5);
    assert_eq!(a.len(), 16 * 16 * 6);
    assert_eq!(a, k.chunk(5), "deterministic");
    assert!(a != k.chunk(9), "two passes of the same tile are different random draws");
    assert!(k.chunk(1) != k.chunk(2));
    // more passes only add chunks: the first pass of a tile does not depend on the number of passes
    let more = Kernel::Render3(RenderParams3 { passes: 5, ..p.clone() });
    assert_eq!(more.chunk(2), k.chunk(2));
    // the image assembled from the passes is a plausible picture
    let outs: Vec<Vec<u8>> = (0..12).map(|c| k.chunk(c)).collect();
    let img = assemble(&p, &outs, 1.0);
    assert_eq!(img.len(), 32 * 32 * 3);
    let distinct: std::collections::BTreeSet<&[u8]> = img.chunks(3).collect();
    assert!(distinct.len() > 100, "the image has many colours ({})", distinct.len());
    // limits
    let mut bad = p.clone();
    bad.bounces = 15;
    assert!(Kernel::Render3(bad).validate(12).is_err());
    let mut bad = p.clone();
    bad.spp = 33;
    assert!(Kernel::Render3(bad).validate(12).is_err());
    let mut bad = p.clone();
    bad.parts = vec![p.parts[1].clone(); 49];
    assert!(Kernel::Render3(bad).validate(12).is_err(), "at most 48 parts");
    let mut bad = p.clone();
    bad.parts[1].mat = 99;
    assert!(Kernel::Render3(bad).validate(12).is_err(), "material index out of range");
    assert!(k.validate(11).is_err(), "chunks = tiles x passes");
    let big = Kernel::Render3(scene3_watch(1920, 1088, 8, 32, [0; 32]));
    assert_eq!(big.forced_chunks(), Some(8160 * 32));
    big.validate(8160 * 32).unwrap();
    assert!(Kernel::Render3(scene3_watch(1920, 1088, 8, 33, [0; 32])).validate(8160 * 33).is_err(), "more than 262,144 chunks");
}

#[test]
fn kernel_validation_bounds_the_cost_of_a_fraud_proof() {
    assert!(mc(1).validate(64).is_ok());
    assert!(mc(1).validate(0).is_err() && mc(1).validate(1025).is_err(), "number of chunks bounded");
    let bad_prog = |ops: Vec<Instr>| Kernel::MonteCarlo(McParams { seed: [0; 32], dim: 2, samples: 16, out: 0, program: ops }).validate(4);
    assert!(bad_prog(vec![]).is_err(), "empty program");
    assert!(bad_prog(vec![instr(99, 0, 0, 0, 0)]).is_err(), "unknown operation");
    assert!(bad_prog(vec![instr(0, 16, 0, 0, 0)]).is_err(), "register out of bounds");
    assert!(bad_prog(vec![instr(1, 0, 2, 0, 0)]).is_err(), "input beyond dim");
    assert!(bad_prog(vec![instr(0, 0, 0, 0, 1); 65]).is_err(), "program too long");
    let heavy = Kernel::MonteCarlo(McParams { seed: [0; 32], dim: 8, samples: 4096, out: 0, program: vec![instr(0, 0, 0, 0, 1); 64] });
    assert!(heavy.cost() <= MAX_CHUNK_COST || heavy.validate(4).is_err());
    // rendering: imposed tiles, size multiple of 16, bounded cost
    assert!(Kernel::Render(demo_scene(64, 64, 1, [0; 32])).validate(15).is_err(), "the number of chunks must equal the number of tiles");
    assert!(Kernel::Render(demo_scene(65, 64, 1, [0; 32])).validate(16).is_err());
    assert!(Kernel::Render(demo_scene(64, 64, 5, [0; 32])).validate(16).is_err(), "spp <= 4");
    let mut s = demo_scene(64, 64, 4, [0; 32]);
    s.prims = vec![s.prims[0].clone(); 9];
    assert!(Kernel::Render(s).validate(16).is_err(), "at most 8 primitives");
    assert!(Kernel::Render(demo_scene(512, 512, 4, [0; 32])).validate(1024).is_ok());
}

// ------------------------------------------------------------ life cycle on the chain

struct Job {
    c: Chain,
    id: JobId,
    outs: Vec<Vec<u8>>,
    tree: MerkleTree,
    kernel: Kernel,
}

/// Submits (client 10), has it claimed (worker 11), and publishes the root of the outputs (possibly falsified).
fn setup(kernel: Kernel, chunks: u32, tamper: Option<u32>) -> Job {
    let mut c = new_chain();
    let id = l2_id(&pubkey(&key(10)), c.state.nonce(&pubkey(&key(10))), &kernel, chunks);
    let submit = ltx(&c, 10, L2Tx::Submit { kernel: kernel.clone(), chunks, escrow: 20 * COIN, deadline: 900 });
    mine(&mut c, vec![submit]).unwrap();
    let claim = ltx(&c, 11, L2Tx::Claim { job_id: id });
    mine(&mut c, vec![claim]).unwrap();
    let mut outs: Vec<Vec<u8>> = (0..chunks).map(|i| kernel.chunk(i)).collect();
    if let Some(t) = tamper {
        outs[t as usize][0] ^= 1;
    }
    let tree = chunk_tree(&outs);
    let post = ltx(&c, 11, L2Tx::Post { job_id: id, root: tree.root() });
    mine(&mut c, vec![post]).unwrap();
    Job { c, id, outs, tree, kernel }
}

#[test]
fn honest_job_is_paid_on_accept_and_by_the_window() {
    let mut j = setup(mc(1), 8, None);
    let before = j.c.state.balance(&pubkey(&key(11)));
    let accept = ltx(&j.c, 10, L2Tx::Accept { job_id: j.id });
    mine(&mut j.c, vec![accept]).unwrap();
    assert_eq!(j.c.state.l2[&j.id].status, L2Status::Settled);
    let net = 20 * COIN - bps(20 * COIN, BURN_BPS);
    assert_eq!(j.c.state.balance(&pubkey(&key(11))), before + net + bps(net, BOND_BPS));
    assert!(j.c.check_supply_invariant());
    // settlement by the window (height AND seconds): nobody accepts, a third party settles
    let mut j = setup(mc(2), 8, None);
    let early = ltx(&j.c, 13, L2Tx::Settle { job_id: j.id });
    assert!(mine(&mut j.c, vec![early]).is_err(), "no payment before the end of the window");
    for _ in 0..12 {
        mine(&mut j.c, vec![]).unwrap();
    }
    let settle = ltx(&j.c, 13, L2Tx::Settle { job_id: j.id });
    mine(&mut j.c, vec![settle]).unwrap();
    assert_eq!(j.c.state.l2[&j.id].status, L2Status::Settled);
    assert!(j.c.check_supply_invariant());
}

#[test]
fn a_falsified_chunk_is_proven_with_one_chunk_and_slashed() {
    for (kernel, chunks, bad) in [(mc(3), 8u32, 5u32), (Kernel::Render(demo_scene(32, 32, 1, [1; 32])), 4, 2), (Kernel::Render2(vearl_core::render2::scene_orb(32, 32, 1, 1, 30.0, 1.2, [1; 32])), 4, 3)] {
        let mut j = setup(kernel, chunks, Some(bad));
        let (client0, challenger0) = (j.c.state.balance(&pubkey(&key(10))), j.c.state.balance(&pubkey(&key(13))));
        let fraud = ltx(&j.c, 13, L2Tx::Fraud { job_id: j.id, chunk: bad, out: j.outs[bad as usize].clone(), path: j.tree.proof(bad as usize) });
        mine(&mut j.c, vec![fraud]).unwrap();
        assert_eq!(j.c.state.l2[&j.id].status, L2Status::Slashed);
        let net = 20 * COIN - bps(20 * COIN, BURN_BPS);
        assert_eq!(j.c.state.balance(&pubkey(&key(10))), client0 + net, "the client recovers the net escrow");
        assert_eq!(j.c.state.balance(&pubkey(&key(13))), challenger0 + bps(net, BOND_BPS) / 2 + 0, "half of the bond to the reporter (block mined by 12)");
        assert_eq!(j.c.state.reputation(&pubkey(&key(11))).slashed, 1);
        assert!(j.c.check_supply_invariant());
    }
}

#[test]
fn false_accusations_and_forged_proofs_are_rejected() {
    let mut j = setup(mc(4), 8, Some(5));
    // correct chunk (3): the chain recomputes it, no fraud
    let ok = ltx(&j.c, 13, L2Tx::Fraud { job_id: j.id, chunk: 3, out: j.outs[3].clone(), path: j.tree.proof(3) });
    assert!(mine(&mut j.c, vec![ok]).is_err(), "a correct chunk is not a fraud");
    // output that is not the committed one: invalid Merkle opening
    let mut forged = j.outs[5].clone();
    forged[3] ^= 0x40;
    let f = ltx(&j.c, 13, L2Tx::Fraud { job_id: j.id, chunk: 5, out: forged, path: j.tree.proof(5) });
    assert!(mine(&mut j.c, vec![f]).is_err(), "the opening must match the committed root");
    // path of another chunk
    let f = ltx(&j.c, 13, L2Tx::Fraud { job_id: j.id, chunk: 5, out: j.outs[5].clone(), path: j.tree.proof(4) });
    assert!(mine(&mut j.c, vec![f]).is_err());
    // indices out of bounds / output too large
    let f = ltx(&j.c, 13, L2Tx::Fraud { job_id: j.id, chunk: 99, out: j.outs[5].clone(), path: j.tree.proof(5) });
    assert!(mine(&mut j.c, vec![f]).is_err());
    let f = ltx(&j.c, 13, L2Tx::Fraud { job_id: j.id, chunk: 5, out: vec![0; MAX_OUT + 1], path: j.tree.proof(5) });
    assert!(mine(&mut j.c, vec![f]).is_err());
    // the true fraud passes, then the closed window refuses it
    let good = ltx(&j.c, 13, L2Tx::Fraud { job_id: j.id, chunk: 5, out: j.outs[5].clone(), path: j.tree.proof(5) });
    mine(&mut j.c, vec![good]).unwrap();
    let mut j2 = setup(mc(5), 8, Some(2));
    for _ in 0..12 {
        mine(&mut j2.c, vec![]).unwrap();
    }
    let late = ltx(&j2.c, 13, L2Tx::Fraud { job_id: j2.id, chunk: 2, out: j2.outs[2].clone(), path: j2.tree.proof(2) });
    assert!(mine(&mut j2.c, vec![late]).is_err(), "dispute window closed");
}

#[test]
fn data_availability_fetch_serve_and_slash_unserved() {
    // the worker serves: it receives the fee, the window is extended
    let mut j = setup(mc(6), 8, None);
    let fetch = ltx(&j.c, 10, L2Tx::Fetch { job_id: j.id, chunk: 2 });
    mine(&mut j.c, vec![fetch]).unwrap();
    let dup = ltx(&j.c, 10, L2Tx::Fetch { job_id: j.id, chunk: 2 });
    assert!(mine(&mut j.c, vec![dup]).is_err(), "duplicate request");
    let third = ltx(&j.c, 13, L2Tx::Fetch { job_id: j.id, chunk: 3 });
    assert!(mine(&mut j.c, vec![third]).is_err(), "only the client can request");
    let wrong = ltx(&j.c, 11, L2Tx::Serve { job_id: j.id, chunk: 2, out: vec![1, 2, 3], path: j.tree.proof(2) });
    assert!(mine(&mut j.c, vec![wrong]).is_err(), "an output unrelated to the root is not a valid service");
    let w0 = j.c.state.balance(&pubkey(&key(11)));
    let serve = ltx(&j.c, 11, L2Tx::Serve { job_id: j.id, chunk: 2, out: j.outs[2].clone(), path: j.tree.proof(2) });
    mine(&mut j.c, vec![serve]).unwrap();
    assert_eq!(j.c.state.balance(&pubkey(&key(11))), w0 + FETCH_FEE);
    assert!(j.c.state.l2[&j.id].fetches.is_empty());
    assert!(j.c.check_supply_invariant());
    // the worker does not serve: after serve_secs, anyone gets it penalised
    let mut j = setup(mc(7), 8, None);
    let fetch = ltx(&j.c, 10, L2Tx::Fetch { job_id: j.id, chunk: 1 });
    mine(&mut j.c, vec![fetch]).unwrap();
    let early = ltx(&j.c, 13, L2Tx::SlashUnserved { job_id: j.id });
    assert!(mine(&mut j.c, vec![early]).is_err(), "not before the deadline");
    for _ in 0..SERVE_SECS / BLOCK_TIME {
        mine(&mut j.c, vec![]).unwrap();
    }
    let block_settle = ltx(&j.c, 13, L2Tx::Settle { job_id: j.id });
    assert!(mine(&mut j.c, vec![block_settle]).is_err(), "no settlement with a pending request");
    let slash = ltx(&j.c, 13, L2Tx::SlashUnserved { job_id: j.id });
    mine(&mut j.c, vec![slash]).unwrap();
    assert_eq!(j.c.state.l2[&j.id].status, L2Status::Slashed);
    assert!(j.c.check_supply_invariant());
}

#[test]
fn validation_access_control_expiry_and_supply() {
    let mut c = new_chain();
    let k = mc(8);
    // escrow below the floor, deadline passed, invalid kernel, inconsistent chunks
    for bad in [
        L2Tx::Submit { kernel: k.clone(), chunks: 4, escrow: COIN / 2, deadline: 900 },
        L2Tx::Submit { kernel: k.clone(), chunks: 4, escrow: 5 * COIN, deadline: 0 },
        L2Tx::Submit { kernel: k.clone(), chunks: 0, escrow: 5 * COIN, deadline: 900 },
        L2Tx::Submit { kernel: Kernel::Render(demo_scene(64, 64, 1, [0; 32])), chunks: 3, escrow: 5 * COIN, deadline: 900 },
    ] {
        let t = ltx(&c, 10, bad);
        assert!(mine(&mut c, vec![t]).is_err());
    }
    let id = l2_id(&pubkey(&key(10)), 0, &k, 4);
    let submit = ltx(&c, 10, L2Tx::Submit { kernel: k.clone(), chunks: 4, escrow: 5 * COIN, deadline: 10 });
    mine(&mut c, vec![submit]).unwrap();
    // access controls: Post by a non-worker, Accept by a third party, Serve before Post
    let post = ltx(&c, 13, L2Tx::Post { job_id: id, root: [1; 32] });
    assert!(mine(&mut c, vec![post]).is_err(), "Post requires being the worker");
    // never claimed: expiry after the deadline, the client is refunded
    let bal = c.state.balance(&pubkey(&key(10)));
    let early = ltx(&c, 13, L2Tx::Settle { job_id: id });
    assert!(mine(&mut c, vec![early]).is_err());
    while c.height() <= 10 {
        mine(&mut c, vec![]).unwrap();
    }
    let settle = ltx(&c, 13, L2Tx::Settle { job_id: id });
    mine(&mut c, vec![settle]).unwrap();
    assert_eq!(c.state.l2[&id].status, L2Status::Expired);
    assert_eq!(c.state.balance(&pubkey(&key(10))), bal + 5 * COIN - bps(5 * COIN, BURN_BPS));
    assert!(c.check_supply_invariant());
    // claimed but never published: bond burned at the deadline
    let k2 = mc(9);
    let id2 = l2_id(&pubkey(&key(10)), c.state.nonce(&pubkey(&key(10))), &k2, 2);
    let submit = ltx(&c, 10, L2Tx::Submit { kernel: k2, chunks: 2, escrow: 5 * COIN, deadline: c.height() + 3 });
    mine(&mut c, vec![submit]).unwrap();
    let claim = ltx(&c, 11, L2Tx::Claim { job_id: id2 });
    mine(&mut c, vec![claim]).unwrap();
    let burned = c.state.burned;
    for _ in 0..5 {
        mine(&mut c, vec![]).unwrap();
    }
    let settle = ltx(&c, 13, L2Tx::Settle { job_id: id2 });
    mine(&mut c, vec![settle]).unwrap();
    assert_eq!(c.state.l2[&id2].status, L2Status::Expired);
    assert_eq!(c.state.burned, burned + bps(5 * COIN - bps(5 * COIN, BURN_BPS), BOND_BPS));
    assert_eq!(c.state.reputation(&pubkey(&key(11))).expired, 1);
    assert!(c.check_supply_invariant());
}

#[test]
fn a_block_may_carry_only_a_few_level_two_fraud_proofs() {
    // judging a fraud proof recomputes a whole chunk: the number per block is a consensus rule (a flood must not stall validators)
    let j = setup(mc(7), 8, Some(5));
    let sk = key(13);
    let base = j.c.state.nonce(&pubkey(&sk));
    let fraud = |n: u64| SignedTx::new(Tx { from: pubkey(&sk), nonce: base + n, fee: 0, kind: TxKind::L2(L2Tx::Fraud { job_id: j.id, chunk: 5, out: j.outs[5].clone(), path: j.tree.proof(5) }) }, &sk);
    let ok: Vec<SignedTx> = (0..MAX_L2_FRAUD_PER_BLOCK as u64).map(fraud).collect();
    assert_eq!(l2_fraud_count(&ok), MAX_L2_FRAUD_PER_BLOCK);
    let too_many: Vec<SignedTx> = (0..MAX_L2_FRAUD_PER_BLOCK as u64 + 1).map(fraud).collect();
    assert_eq!(l2_fraud_count(&too_many), MAX_L2_FRAUD_PER_BLOCK + 1);
    let mut c = j.c;
    let err = mine(&mut c, too_many).unwrap_err();
    assert!(err.contains("level-2 fraud proofs"), "{err}");
}

#[test]
fn the_explorer_index_follows_the_life_of_a_job_and_survives_reorganisations() {
    use vearl_core::index::Index;
    let mut j = setup(mc(11), 8, None);
    let accept = ltx(&j.c, 10, L2Tx::Accept { job_id: j.id });
    mine(&mut j.c, vec![accept]).unwrap();
    let mut ix = Index::default();
    ix.sync(&j.c);
    assert_eq!(ix.len(), j.c.blocks.len());
    let evs = &ix.jobs[&j.id];
    let whats: Vec<&str> = evs.iter().map(|e| e.what).collect();
    assert_eq!(whats, ["submitted", "claimed", "posted", "accepted"], "{whats:?}");
    assert!(evs.windows(2).all(|w| w[0].height < w[1].height), "events follow the chain order");
    assert_eq!(evs[0].by, pubkey(&key(10)), "the client submitted");
    assert_eq!(evs[1].by, pubkey(&key(11)), "the worker claimed");
    assert!(evs.iter().all(|e| e.tx.is_some()));
    // per-day totals add up to the chain
    let (blocks, submitted, closed): (u64, u64, u64) = ix.days.values().fold((0, 0, 0), |a, d| (a.0 + d.blocks, a.1 + d.jobs_submitted, a.2 + d.jobs_closed));
    assert_eq!((blocks, submitted, closed), (j.c.blocks.len() as u64, 1, 1));
    assert!(ix.recent.iter().any(|(id, kind, e)| *id == j.id && *kind == "level2" && e.what == "accepted"));
    // incremental sync equals a fresh rebuild
    mine(&mut j.c, vec![]).unwrap();
    ix.sync(&j.c);
    let mut fresh = Index::default();
    fresh.sync(&j.c);
    assert_eq!((ix.len(), ix.jobs[&j.id].len(), ix.days.len()), (fresh.len(), fresh.jobs[&j.id].len(), fresh.days.len()));
    // a different chain under the same index: everything is rebuilt, nothing of the old chain stays
    let other = setup(mc(12), 8, None);
    ix.sync(&other.c);
    assert!(ix.jobs.get(&j.id).is_none(), "the old chain's job is gone");
    assert_eq!(ix.jobs[&other.id].iter().map(|e| e.what).collect::<Vec<_>>(), ["submitted", "claimed", "posted"]);
    assert_eq!(ix.len(), other.c.blocks.len());
}

// ------------------------------------------------------------ a job whose worker disappeared goes back to the queue

fn mine_at(c: &mut Chain, txs: Vec<SignedTx>, ts: u64) -> Result<(), String> {
    let out = c.mine(&key(12), WorkClass::Native, None, txs, ts)?;
    c.validate_and_apply(out.block)
}

fn claim_start(c: &Chain, id: &JobId) -> u64 {
    match c.state.l2[id].status {
        L2Status::Claimed { at_ts, .. } => at_ts,
        ref s => panic!("not claimed: {s:?}"),
    }
}

#[test]
fn a_job_claimed_by_a_worker_that_disappeared_is_taken_back_after_the_lease() {
    let mut c = new_chain();
    let lease = c.state.timing.claim_lease_secs();
    assert_eq!(lease, 15 * c.state.timing.serve_secs);
    assert_eq!(Timing::default().claim_lease_secs(), 1_800, "30 minutes with the genesis of the testnet");
    let kernel = mc(21);
    let id = l2_id(&pubkey(&key(10)), c.state.nonce(&pubkey(&key(10))), &kernel, 4);
    let submit = ltx(&c, 10, L2Tx::Submit { kernel: kernel.clone(), chunks: 4, escrow: 20 * COIN, deadline: 900 });
    mine(&mut c, vec![submit]).unwrap();
    let claim = ltx(&c, 11, L2Tx::Claim { job_id: id });
    mine(&mut c, vec![claim]).unwrap();
    let t0 = claim_start(&c, &id);
    let net = 20 * COIN - bps(20 * COIN, BURN_BPS);
    let bond = bps(net, BOND_BPS);

    // while the lease runs, nobody can take the job away from its worker
    let early = ltx(&c, 13, L2Tx::Claim { job_id: id });
    assert!(mine(&mut c, vec![early]).is_err(), "a claimed job is not claimable at once");
    let early = ltx(&c, 13, L2Tx::Claim { job_id: id });
    assert!(mine_at(&mut c, vec![early], t0 + lease - 1).is_err(), "not one second before the end of the lease");
    assert!(matches!(c.state.l2[&id].status, L2Status::Claimed { worker, .. } if worker == pubkey(&key(11))));

    // the lease has run out: another worker takes the job, the first one loses its bond
    let burned = c.state.burned;
    let bal13 = c.state.balance(&pubkey(&key(13)));
    let takeover = ltx(&c, 13, L2Tx::Claim { job_id: id });
    mine_at(&mut c, vec![takeover], t0 + lease).unwrap();
    assert!(matches!(c.state.l2[&id].status, L2Status::Claimed { worker, at_ts, .. } if worker == pubkey(&key(13)) && at_ts == t0 + lease));
    assert_eq!(c.state.burned, burned + bond, "the bond of the worker that disappeared is burned");
    assert_eq!(c.state.balance(&pubkey(&key(13))), bal13 - bond);
    assert_eq!(c.state.reputation(&pubkey(&key(11))).expired, 1);
    assert!(c.check_supply_invariant());

    // the first worker can no longer publish; the new one can, and is paid
    let late = ltx(&c, 11, L2Tx::Post { job_id: id, root: [7; 32] });
    assert!(mine(&mut c, vec![late]).is_err(), "the former worker has lost the job");
    let outs: Vec<Vec<u8>> = (0..4).map(|i| kernel.chunk(i)).collect();
    let tree = chunk_tree(&outs);
    let post = ltx(&c, 13, L2Tx::Post { job_id: id, root: tree.root() });
    mine(&mut c, vec![post]).unwrap();
    let accept = ltx(&c, 10, L2Tx::Accept { job_id: id });
    mine(&mut c, vec![accept]).unwrap();
    assert_eq!(c.state.l2[&id].status, L2Status::Settled);
    assert_eq!(c.state.balance(&pubkey(&key(13))), bal13 + net, "the new worker gets the price and its bond back");
    assert!(c.check_supply_invariant());
}

#[test]
fn a_slow_worker_that_publishes_before_anybody_takes_over_keeps_its_job() {
    let mut c = new_chain();
    let lease = c.state.timing.claim_lease_secs();
    let kernel = mc(22);
    let id = l2_id(&pubkey(&key(10)), c.state.nonce(&pubkey(&key(10))), &kernel, 2);
    let submit = ltx(&c, 10, L2Tx::Submit { kernel: kernel.clone(), chunks: 2, escrow: 10 * COIN, deadline: 900 });
    mine(&mut c, vec![submit]).unwrap();
    let claim = ltx(&c, 11, L2Tx::Claim { job_id: id });
    mine(&mut c, vec![claim]).unwrap();
    let t0 = claim_start(&c, &id);
    let outs: Vec<Vec<u8>> = (0..2).map(|i| kernel.chunk(i)).collect();
    let post = ltx(&c, 11, L2Tx::Post { job_id: id, root: chunk_tree(&outs).root() });
    mine_at(&mut c, vec![post], t0 + lease + 5).unwrap();
    assert!(matches!(c.state.l2[&id].status, L2Status::Posted { worker, .. } if worker == pubkey(&key(11))));
    // a posted job is never claimable again, whatever the time
    let steal = ltx(&c, 13, L2Tx::Claim { job_id: id });
    assert!(mine_at(&mut c, vec![steal], t0 + 10 * lease).is_err());
    assert!(c.check_supply_invariant());
}

#[test]
fn a_job_past_its_deadline_is_not_taken_back_it_expires_and_the_client_is_refunded() {
    let mut c = new_chain();
    let lease = c.state.timing.claim_lease_secs();
    let kernel = mc(23);
    let id = l2_id(&pubkey(&key(10)), c.state.nonce(&pubkey(&key(10))), &kernel, 2);
    let submit = ltx(&c, 10, L2Tx::Submit { kernel, chunks: 2, escrow: 10 * COIN, deadline: c.height() + 4 });
    mine(&mut c, vec![submit]).unwrap();
    let claim = ltx(&c, 11, L2Tx::Claim { job_id: id });
    mine(&mut c, vec![claim]).unwrap();
    let t0 = claim_start(&c, &id);
    while c.height() <= c.state.l2[&id].deadline {
        mine(&mut c, vec![]).unwrap();
    }
    let late = ltx(&c, 13, L2Tx::Claim { job_id: id });
    assert!(mine_at(&mut c, vec![late], t0 + lease + 1).is_err(), "no new claim after the deadline");
    let settle = ltx(&c, 13, L2Tx::Settle { job_id: id });
    mine_at(&mut c, vec![settle], t0 + lease + 2).unwrap();
    assert_eq!(c.state.l2[&id].status, L2Status::Expired);
    assert!(c.check_supply_invariant());
}
