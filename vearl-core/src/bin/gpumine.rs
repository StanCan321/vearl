//! End-to-end GPU miner: mines Native blocks with the CUDA kernel, and EVERY block is re-verified
//! by the CPU validator (Merkle tile proof + recomputation of a single tile).
//! Usage: gpumine [n=4096] [fold_k=64] [blocks=10]   (rank=63; fold_k must respect security_check)
use ed25519_dalek::SigningKey;
use vearl_core::chain::*;
use vearl_core::crypto::*;
use vearl_core::state::*;
use vearl_core::types::*;
use vearl_core::work::*;
use std::time::Instant;

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|x| x.parse().ok()).collect();
    let (n, fold_k, blocks) = (*a.first().unwrap_or(&4096), *a.get(1).unwrap_or(&64), *a.get(2).unwrap_or(&10));
    let p = WorkParams { n, tile: 16, rank: 63, fold_k };
    let tiles2 = (p.tiles() * p.tiles()) as u64;
    // target: ~1 winning attempt per sigma on average (p = 1 / nb_tiles)
    let mut chain = Chain::genesis(p, &[], 1_000, u64::MAX / tiles2);
    let miner = SigningKey::from_bytes(&[42; 32]);

    println!("n={n} fold_k={fold_k} | {tiles2} attempts per sigma | target p~1/{tiles2}");
    let t_init = Instant::now();
    chain.native_committed(0); // Merkle commitment of the native matrices (once per epoch)
    println!("commitment of the native matrices (Merkle): {:.0} ms", t_init.elapsed().as_secs_f64() * 1e3);

    let (mut mine_ms, mut verify_ms, mut sigmas) = (0f64, 0f64, 0u64);
    for _ in 0..blocks {
        let ts = chain.tip_ts() + BLOCK_TIME;
        let t0 = Instant::now();
        let out = chain.mine_gpu(&miner, WorkClass::Native, None, vec![], ts).expect("gpu mining");
        let m = t0.elapsed().as_secs_f64() * 1e3;
        sigmas += out.attempts_sigma;
        let h = out.block.header.height;
        let t1 = Instant::now();
        chain.validate_and_apply(out.block).expect("CPU validator rejects a GPU block");
        let v = t1.elapsed().as_secs_f64() * 1e3;
        mine_ms += m;
        verify_ms += v;
        println!("  block #{h:<3} mined={m:>8.1} ms ({} sigma)   verified (CPU, 1 tile)={v:>7.2} ms", chain.blocks[h as usize].header.extra_nonce + 1);
    }
    println!("\n{blocks} blocks valid for the CPU validator. Average: mining {:.1} ms/block ({} sigma in total), verification {:.2} ms/block",
        mine_ms / blocks as f64, sigmas, verify_ms / blocks as f64);
    println!("supply OK = {}, miner balance = {} VRL", chain.check_supply_invariant(), chain.state.balance(&pubkey(&miner)) / COIN);
}
