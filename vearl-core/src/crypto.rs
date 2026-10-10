//! Primitives: blake3 (hash), ed25519-dalek (signatures), binary Merkle tree.
//! No home-made primitive: existing building blocks are composed.
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};

pub type Hash = [u8; 32];
pub type PubKey = [u8; 32];

/// Hash with domain separation (blake3 derive_key).
pub fn h(domain: &str, parts: &[&[u8]]) -> Hash {
    let mut x = blake3::Hasher::new_derive_key(domain);
    for p in parts {
        x.update(p);
    }
    *x.finalize().as_bytes()
}

pub fn pubkey(sk: &SigningKey) -> PubKey {
    sk.verifying_key().to_bytes()
}

pub fn sign(sk: &SigningKey, msg: &[u8]) -> Vec<u8> {
    sk.sign(msg).to_bytes().to_vec()
}

/// Verifies a transaction signature. Two schemes, distinguished by the LENGTH of the signature: 64 bytes = Ed25519,
/// `PQ_SIG_LEN` = SLH-DSA-SHAKE-128s (post-quantum, see pq.rs). Any other length is refused.
pub fn verify_sig(pk: &PubKey, msg: &[u8], sig: &[u8]) -> bool {
    if sig.len() == crate::pq::PQ_SIG_LEN {
        return crate::pq::pq_verify(pk, msg, sig);
    }
    let Ok(vk) = VerifyingKey::from_bytes(pk) else { return false };
    let Ok(arr) = <[u8; 64]>::try_from(sig) else { return false };
    vk.verify_strict(msg, &Signature::from_bytes(&arr)).is_ok()
}

// ---------------- Merkle ----------------

/// `(0..n).map(f).collect()` on all the cores as soon as n is large (the Merkle trees of 10^6 leaves dominate the cost of a
/// job at n = 1024). The result is identical to the sequential version: each element only depends on its index.
pub fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    const PAR_MIN: usize = 8192;
    if n < PAR_MIN {
        return (0..n).map(f).collect();
    }
    let threads = std::thread::available_parallelism().map_or(1, |x| x.get()).min(32);
    let per = (n + threads - 1) / threads;
    let mut out: Vec<Vec<T>> = Vec::with_capacity(threads);
    std::thread::scope(|sc| {
        let f = &f;
        let handles: Vec<_> = (0..threads).map(|t| sc.spawn(move || ((t * per).min(n)..((t + 1) * per).min(n)).map(f).collect::<Vec<T>>())).collect();
        for h in handles {
            out.push(h.join().unwrap());
        }
    });
    out.into_iter().flatten().collect()
}

/// `(0..n).map(f).collect()` for FEW EXPENSIVE elements (e.g. the operations of a program): one thread per core draws the indices
/// from a common queue (no minimum threshold, no fixed splitting). Result identical to the sequential version.
pub fn par_map_heavy<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    par_map_limit(n, std::thread::available_parallelism().map_or(1, |x| x.get()), f)
}

/// Like `par_map_heavy` with at most `threads` threads (e.g. bounded simultaneous network requests).
pub fn par_map_limit<T: Send, F: Fn(usize) -> T + Sync>(n: usize, threads: usize, f: F) -> Vec<T> {
    let threads = threads.min(n).max(1);
    if threads == 1 {
        return (0..n).map(f).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<(usize, T)> = Vec::with_capacity(n);
    std::thread::scope(|sc| {
        let (f, next) = (&f, &next);
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(move || {
                    let mut mine = vec![];
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= n {
                            break mine;
                        }
                        mine.push((i, f(i)));
                    }
                })
            })
            .collect();
        for h in hs {
            out.extend(h.join().unwrap());
        }
    });
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, v)| v).collect()
}

pub fn leaf_hash(index: u64, data: &[u8]) -> Hash {
    h("vearl/merkle/leaf/v1", &[&index.to_le_bytes(), data])
}

fn node(l: &Hash, r: &Hash) -> Hash {
    h("vearl/merkle/node/v1", &[l, r])
}

pub fn depth_for(count: usize) -> usize {
    let mut d = 0;
    while (1usize << d) < count {
        d += 1;
    }
    d
}

#[derive(Clone, Debug)]
pub struct MerkleTree {
    levels: Vec<Vec<Hash>>,
}

impl MerkleTree {
    pub fn new(mut leaves: Vec<Hash>) -> Self {
        let size = 1usize << depth_for(leaves.len().max(1));
        leaves.resize(size, [0u8; 32]);
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let cur = levels.last().unwrap();
            let next = par_map(cur.len() / 2, |i| node(&cur[2 * i], &cur[2 * i + 1]));
            levels.push(next);
        }
        Self { levels }
    }

    /// Root only, without keeping the levels (O(n) memory instead of O(n) levels, half the copy time): same value
    /// as `MerkleTree::new(leaves).root()`.
    pub fn root_of(mut leaves: Vec<Hash>) -> Hash {
        let size = 1usize << depth_for(leaves.len().max(1));
        leaves.resize(size, [0u8; 32]);
        while leaves.len() > 1 {
            let next = par_map(leaves.len() / 2, |i| node(&leaves[2 * i], &leaves[2 * i + 1]));
            leaves = next;
        }
        leaves[0]
    }

    /// Like `root_of`, on a SINGLE thread: to be called from several threads that are already parallel (avoids oversubscription).
    pub fn root_of_st(mut leaves: Vec<Hash>) -> Hash {
        let size = 1usize << depth_for(leaves.len().max(1));
        leaves.resize(size, [0u8; 32]);
        let mut len = size;
        while len > 1 {
            for i in 0..len / 2 {
                leaves[i] = node(&leaves[2 * i], &leaves[2 * i + 1]);
            }
            len /= 2;
        }
        leaves[0]
    }

    pub fn root(&self) -> Hash {
        self.levels.last().unwrap()[0]
    }

    pub fn proof(&self, mut idx: usize) -> Vec<Hash> {
        let mut p = Vec::new();
        for lvl in &self.levels[..self.levels.len() - 1] {
            p.push(lvl[idx ^ 1]);
            idx >>= 1;
        }
        p
    }
}

/// Verifies a Merkle path. `depth` fixes the expected depth (avoids leaf/node ambiguity).
pub fn verify_proof(root: &Hash, leaf: Hash, index: usize, path: &[Hash], depth: usize) -> bool {
    if path.len() != depth || index >= (1usize << depth) {
        return false;
    }
    let mut cur = leaf;
    let mut idx = index;
    for sib in path {
        cur = if idx & 1 == 0 { node(&cur, sib) } else { node(sib, &cur) };
        idx >>= 1;
    }
    &cur == root
}

pub fn i8_bytes(v: &[i8]) -> Vec<u8> {
    v.iter().map(|x| *x as u8).collect()
}
