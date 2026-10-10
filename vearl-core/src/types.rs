use crate::crypto::*;
use crate::mlp::MlpTx;
use crate::prog::ProgTx;
use crate::tiled::TiledTx;
use crate::work::TileProof;
use serde::{Deserialize, Serialize};

pub type JobId = Hash;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FraudProof {
    pub job_id: JobId,
    pub i: u32,
    pub j: u32,
    pub a_row: Vec<i8>,
    pub a_path: Vec<Hash>,
    pub b_col: Vec<i8>,
    pub b_path: Vec<Hash>,
    /// Row `i` of the published result (row-wise commitment, docs/07 section 3); the accused entry is `c_row[j]`.
    pub c_row: Vec<i32>,
    pub c_path: Vec<Hash>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TxKind {
    Transfer { to: PubKey, amount: u64 },
    /// The client commits A (rows) and B (columns) and locks the escrow. Part of it is burned.
    SubmitJob { a_root: Hash, b_root: Hash, escrow: u64, deadline: u64 },
    ClaimJob { job_id: JobId },
    PostResult { job_id: JobId, c_root: Hash },
    Accept { job_id: JobId },
    Fraud(FraudProof),
    Settle { job_id: JobId },
    /// Like SubmitJob, with a REVERSE AUCTION of `auction_secs` (timestamp seconds): the workers propose a
    /// price, the lowest wins. The client recovers the difference between the net escrow and the retained price.
    SubmitJobAuction { a_root: Hash, b_root: Hash, escrow: u64, deadline: u64, auction_secs: u64 },
    /// Bid of a worker on an auction job: must be strictly lower than the current best bid.
    Bid { job_id: JobId, price: u64 },
    /// The client demands, ON-CHAIN, an entry (i,j) of the published result (C data committed by c_root).
    Fetch { job_id: JobId, i: u32, j: u32 },
    /// The worker serves the ROW `i` containing the requested entry, with its Merkle path. Otherwise, after SERVE_SECS: SlashUnserved.
    Serve { job_id: JobId, i: u32, j: u32, row: Vec<i32>, path: Vec<Hash> },
    /// Anyone: a data request was left unanswered -> the worker is slashed.
    SlashUnserved { job_id: JobId },
    /// Multi-layer quantised inference jobs (see mlp.rs).
    Mlp(MlpTx),
    /// Block-wise matrix products (rectangular shapes), see tiled.rs.
    Tiled(TiledTx),
    /// Verifiable tensor programs (convolutional networks, residuals): see prog.rs.
    Prog(ProgTx),
    /// Level 2: Monte Carlo and rendering, audits by sampling (see l2.rs, docs/10).
    L2(crate::l2::L2Tx),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Tx {
    pub from: PubKey,
    pub nonce: u64,
    pub fee: u64,
    pub kind: TxKind,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SignedTx {
    pub tx: Tx,
    pub sig: Vec<u8>,
}

impl SignedTx {
    pub fn new(tx: Tx, sk: &ed25519_dalek::SigningKey) -> Self {
        let msg = bincode::serialize(&tx).unwrap();
        let sig = sign(sk, &msg);
        Self { tx, sig }
    }
    /// Transaction signed by a POST-QUANTUM account (SLH-DSA, see pq.rs): `tx.from` must be `key.pubkey()`.
    pub fn new_pq(tx: Tx, key: &crate::pq::PqKey) -> Self {
        assert_eq!(tx.from, key.pubkey(), "from != post-quantum key");
        let msg = bincode::serialize(&tx).unwrap();
        let sig = key.sign(&msg);
        Self { tx, sig }
    }
    pub fn id(&self) -> Hash {
        h("vearl/tx/v1", &[&bincode::serialize(self).unwrap()])
    }
}

pub fn job_id(client: &PubKey, nonce: u64, a_root: &Hash, b_root: &Hash) -> JobId {
    h("vearl/job/v1", &[client, &nonce.to_le_bytes(), a_root, b_root])
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum WorkClass {
    /// Matrices derived from the protocol: pure security, NO usefulness claimed.
    Native,
    /// Matrices committed by a paid job (escrow + burn). The miner must be its worker.
    Job(JobId),
    /// Layer `layer` of an inference job: A = W_layer, B = X_{layer-1} (X_0 supplied by the client).
    Layer { job: JobId, layer: u8 },
    /// Product P_ijk = A_ik * B_kj of a block-wise job: A = block of A, B = block of B (roots committed by the client).
    Product { job: JobId, i: u8, j: u8, k: u8 },
    /// Gemm operation `op` of a tensor program: A = W_w, B = input tensor (already published).
    Op { job: JobId, op: u8 },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Header {
    pub version: u32,
    pub height: u64,
    pub prev: Hash,
    pub timestamp: u64,
    pub miner: PubKey,
    pub class: WorkClass,
    pub extra_nonce: u64,
    pub ti: u32,
    pub tj: u32,
    pub jackpot: Hash,
    pub tx_root: Hash,
    pub target: u64,
    /// (version 2) roots of A and B of the work: a light client verifies the tile proof without knowing the state.
    pub a_root: Hash,
    pub b_root: Hash,
    /// (version 2) Merkle root of the accounts AFTER this block (see `State::accounts_root`): allows balance proofs.
    pub accounts_root: Hash,
}

impl Header {
    pub fn hash(&self) -> Hash {
        h("vearl/header/v1", &[&bincode::serialize(self).unwrap()])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Block {
    pub header: Header,
    pub proof: TileProof,
    pub txs: Vec<SignedTx>,
}

/// Noise seed: binds the work to the parent, the miner, the class, the txs and the timestamp.
/// (binding tx_root prevents replaying work found with another set of transactions)
pub fn sigma(prev: &Hash, miner: &PubKey, class: &WorkClass, extra_nonce: u64, tx_root: &Hash, timestamp: u64) -> Hash {
    h(
        "vearl/sigma/v1",
        &[prev, miner, &bincode::serialize(class).unwrap(), &extra_nonce.to_le_bytes(), tx_root, &timestamp.to_le_bytes()],
    )
}

pub fn tx_root(txs: &[SignedTx]) -> Hash {
    if txs.is_empty() {
        return [0u8; 32];
    }
    MerkleTree::new(txs.iter().enumerate().map(|(i, t)| leaf_hash(i as u64, &t.id())).collect()).root()
}
