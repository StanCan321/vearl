//! VERIFIABLE tensor programs: a sequence of operations on n x n int8 tensors, each one verifiable
//! independently by an element-wise fraud proof. This is what allows convolutional networks with residuals,
//! beyond the dense layers of mlp.rs.
//!
//! Tensors: T_0 = client input; T_{k+1} = output of operation k. All n x n, committed BY COLUMNS (one leaf
//! per column): a column serves as the B operand of a GEMM, and a cell (i,j) is opened with column j.
//! Operations:
//!   Gemm{w,x,shift,relu} : T = act((W_w * T_x) >> shift)     (Z int32 committed per entry; this GEMM is what is MINED)
//!   Add{a,b,relu}        : T = saturation(T_a + T_b)          (residual connection)
//!   Map{x,map}           : T[i][j] = T_x[src(i,j)] or 0        (transposition, im2col: convolution = GEMM on patches)
//!   Pool{x,pool}         : T[i][j] = integer mean of a kxk window of T_x   (rounding by shift)
//!   GemmAT{a,x,shift,relu}: T = act((T_a^T * T_x) >> shift)   (product between TWO computed tensors: attention scores K^T Q;
//!                           the "row-wise" root of T_a^T is exactly the "column-wise" root of T_a: nothing more to commit)
//!   Softmax{x,len}       : INTEGER softmax of each column over its first `len` cells (fixed exp2 table), output in [0,64]
//!   SoftmaxLo{x,len}     : FINE part of the same softmax: round(4096 p) - 64 round(64 p), in [-32,32]; with Softmax, p to within 1/4096
//!   CausalSoftmax{x,len,l,fine} : CAUSAL softmax (or its fine part): column j = query q = j mod l, keys 0..=q only (language models)
//!   CausalSoftmaxS{x,len,l,fine,scale} : the same with a COARSER exponent step: one unit of the score = 2^scale * ln2/8 nats (wide logits)
//!   LayerNorm{x,len}     : INTEGER normalisation of each column over `len` cells (mean, integer standard deviation), output in [-64,64]
//!   Lut{x,w,row}         : T[i][j] = W_w[row][clamp(T_x[i][j], -64, 63) + 64] : TABLE of 128 entries (one row of a weight block) applied to
//!                           each cell: quantised GELU, SiLU, tanh, sigmoid... Same proof shape as Bias (input column, table row, output column).
//!   Bias{x,w,row,relu}   : T[i][j] = saturation(T_x[i][j] + W_w[row][i]) : per-row BIAS (a bias vector = one row of a
//!                           weight block; a single block carries up to n bias vectors). Column by column, like softmax: the fraud proof
//!                           opens the input column, the bias row (row-wise root of the weight) and the output column.
//! Layout (features x samples/tokens): one column = one token; LayerNorm and attention softmax therefore work
//! on COLUMNS, already committed one by one: the fraud proof opens two columns (input and output), O(n).
//! FIXED INTEGER semantics (arithmetic shift, saturation to [-64,64], ReLU = lower bound 0): two honest workers
//! compute bit-identical tensors, a condition for a fraud proof to be indisputable.
//! Fraud proofs (on-chain, O(n) or O(window * n)): Gemm, Act, Elem (Add / Map / Pool).
//! Limits: n x n tensors (zero padding), no softmax/normalisation (P10c), a single worker, no guarantee of
//! availability of the intermediate tensors.
use crate::crypto::*;
use crate::state::*;
use crate::types::*;
use crate::work::*;
use serde::{Deserialize, Serialize};

/// Bounds of a program. A large network is cut into n x n blocks (block products + additions, docs/11): up to 240
/// operations and 64 weight matrices (indices on one byte; v1: 16 and 8, then 64 and 32).
pub const MAX_OPS: usize = 240;
pub const MAX_WEIGHTS: usize = 64;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum MapSpec {
    /// out[i][j] = in[j][i]
    Transpose,
    /// Convolution kh x kw, stride 1, padding `pad`: in = (channels) x (y, x, batch); out row r = (c, dy, dx), column q = (y, x, b).
    Im2col { c_in: u16, h: u16, w: u16, batch: u16, kh: u8, kw: u8, pad: u8 },
    /// Flattening (c, y, x, batch) -> (c*h*w rows, batch columns): row i = c*h*w + y*w + x, column = sample of the batch.
    /// Input: layout (channel row, position*batch + b column). Used to go from a convolutional network to a dense classifier.
    Flatten { c: u16, h: u16, w: u16, batch: u16 },
    /// GENERAL convolution kh x kw, step `stride`, padding `pad` (stride 1 and padding kh/2: like Im2col; stride > 1 or padding 0:
    /// output on a smaller grid). Input (channel, (y, x, batch)) on h x w; output row r = (c, dy, dx), column q = (oy, ox, b) on the
    /// grid oh x ow, oh = (h + 2 pad - kh) / stride + 1 (same for ow); source outside the image = 0.
    Conv { c_in: u16, h: u16, w: u16, batch: u16, kh: u8, kw: u8, pad: u8, stride: u8 },
    /// Shift: out[i][j] = in[i + dr][j + dc] (0 beyond the tensor). Used to extract a block of a WIDE input tensor (more than n
    /// features: the feature blocks are laid side by side in columns, docs/12).
    Shift { dr: u16, dc: u16 },
    /// MULTI-HEAD attention in one product: head h (rows h*dh .. h*dh+dh-1) is copied into the column block h*l .. h*l+l-1:
    /// out[i][h*l + j] = in[i][j] if i / dh = h (i < heads*dh, j < l), 0 elsewhere. With K^T . out, we get the scores of all
    /// heads side by side (keys x (head, query)).
    HeadSpread { dh: u16, heads: u16, l: u16 },
    /// Inverse: out[i][j] = in[i][(i / dh)*l + j] for i < heads*dh, j < l (each row takes back the column block of its head).
    HeadGather { dh: u16, heads: u16, l: u16 },
}

/// Dimensions (oh, ow) of the output grid of a general convolution (None if the window does not fit).
pub fn conv_out_dims(h: usize, w: usize, kh: usize, kw: usize, pad: usize, stride: usize) -> Option<(usize, usize)> {
    if stride == 0 || h + 2 * pad < kh || w + 2 * pad < kw {
        return None;
    }
    Some(((h + 2 * pad - kh) / stride + 1, (w + 2 * pad - kw) / stride + 1))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PoolSpec {
    /// Mean over disjoint k x k windows; output (c, y/k, x/k, b).
    Avg { c: u16, h: u16, w: u16, batch: u16, k: u8 },
    /// Maximum over disjoint k x k windows (same dimensions and same layout as `Avg`).
    Max { c: u16, h: u16, w: u16, batch: u16, k: u8 },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Op {
    Gemm { w: u8, x: u8, shift: u8, relu: bool },
    Add { a: u8, b: u8, relu: bool },
    Map { x: u8, map: MapSpec },
    Pool { x: u8, pool: PoolSpec },
    GemmAT { a: u8, x: u8, shift: u8, relu: bool },
    Softmax { x: u8, len: u16 },
    LayerNorm { x: u8, len: u16 },
    Bias { x: u8, w: u8, row: u16, relu: bool },
    Lut { x: u8, w: u8, row: u16 },
    /// PER-ROW multiplication (gamma of a LayerNorm): out[i][j] = sat((x[i][j] * g[i]) >> shift), g = row `row` of the weight `w`
    /// (arbitrary 8-bit integers), arithmetic shift. Same fraud proof shape as Bias (input column + row + column).
    Mul { x: u8, w: u8, row: u16, shift: u8 },
    /// FINE part of the integer softmax of each column: q4096 - 64 q64, where q64 = output of Softmax (p in 1/64) and q4096 = p in 1/4096
    /// (same e_j and S). In [-32, 32]. A model computes V.A_high + (V.A_fine)/64: attention over hundreds of tokens no longer loses
    /// the probabilities < 1/128 (rounded to 0 by Softmax alone). Same fraud proof as Softmax (column).
    SoftmaxLo { x: u8, len: u16 },
    /// CAUSAL softmax (language model: a token only sees the previous tokens). Column j = query q = j mod l (columns
    /// (head, query) after HeadSpread, l = length of a head's block): only its first min(len, q + 1) cells (keys 0..=q)
    /// count, the following ones are 0. fine = false: output of Softmax (1/64); fine = true: fine part of SoftmaxLo (1/4096).
    /// Same fraud proof as Softmax (column).
    CausalSoftmax { x: u8, len: u16, l: u16, fine: bool },
    /// RMS NORMALISATION of each column, in groups of `group` consecutive rows over the first `len` rows (`len` a multiple of `group`; group = len:
    /// one normalisation of the whole column; group = head size: one per attention head, the QK-norm of the Qwen3 models). Like LayerNorm
    /// WITHOUT the subtraction of the mean: r = isqrt(floor(256 sum(v^2) / group) + 256) (16 sqrt(ms + 1), 4 fractional bits), output = clamp(floor((256 v + r/2) / r), -64, 64).
    /// Cells >= len: 0. Same fraud proof as LayerNorm (column).
    RmsNorm { x: u8, len: u16, group: u16, shift: u8 },
    /// ELEMENT-WISE product of two tensors: out[i][j] = sat((a[i][j] * b[i][j]) >> shift), arithmetic shift (the gate of a SwiGLU feed-forward network:
    /// silu(gate) * up). Same fraud proof as Add (Elem: the two source columns and the output column).
    MulT { a: u8, b: u8, shift: u8 },
    /// ELEMENT-WISE product of a tensor and a WEIGHT block: out[i][j] = sat((x[i][j] * W[i][j]) >> shift), W = weight block `w` (the cos and sin tables
    /// of the rotary position embedding). Fraud proof `MulW`: the source column j, ROW i of the weight, the output column j.
    MulW { x: u8, w: u8, shift: u8 },
    /// CAUSAL softmax with a COARSER EXPONENT STEP: like CausalSoftmax, but e_j = EXP2_TABLE[min(128, (max - v_j) << scale)], i.e. one unit of the
    /// score is 2^scale * ln2/8 nats instead of ln2/8 (scale 0 is CausalSoftmax; scale <= 4). The attention logits of the Qwen3 models span more than
    /// 20 nats relative to the sink token, the 8-bit score tensor of CausalSoftmax only 11 (+-64 units of 0.087 nat). Same fraud proof (column).
    CausalSoftmaxS { x: u8, len: u16, l: u16, fine: bool, scale: u8 },
}

/// Largest exponent step shift of `CausalSoftmaxS`.
pub const MAX_SOFTMAX_SCALE: u8 = 4;
/// Largest output shift of `RmsNorm`.
pub const MAX_RMS_SHIFT: u8 = 4;

impl Op {
    /// Operation whose computation is a mined GEMM (Gemm or GemmAT).
    pub fn is_gemm(&self) -> bool {
        matches!(self, Op::Gemm { .. } | Op::GemmAT { .. })
    }
    /// Operation applied column by column (softmax, layer normalisation).
    pub fn is_col_op(&self) -> bool {
        matches!(self, Op::Softmax { .. } | Op::SoftmaxLo { .. } | Op::LayerNorm { .. } | Op::CausalSoftmax { .. } | Op::CausalSoftmaxS { .. } | Op::RmsNorm { .. })
    }
    /// Tensor read by a per-column operation.
    pub fn col_input(&self) -> Option<u8> {
        match self {
            Op::Softmax { x, .. } | Op::SoftmaxLo { x, .. } | Op::LayerNorm { x, .. } | Op::CausalSoftmax { x, .. } | Op::CausalSoftmaxS { x, .. } | Op::RmsNorm { x, .. } => Some(*x),
            _ => None,
        }
    }
    /// Addition of a bias vector (row of a weight): column by column, with one extra opening (the bias row).
    pub fn is_bias(&self) -> bool {
        matches!(self, Op::Bias { .. })
    }
    /// Lookup table (arbitrary activation): same proof shape as Bias.
    pub fn is_lut(&self) -> bool {
        matches!(self, Op::Lut { .. })
    }
    /// Per-row multiplication (gamma): fraud proof of the Bias shape.
    pub fn is_mul(&self) -> bool {
        matches!(self, Op::Mul { .. })
    }
    /// Element-wise product with a weight block: fraud proof `MulW`.
    pub fn is_mulw(&self) -> bool {
        matches!(self, Op::MulW { .. })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OpenedCol {
    pub tensor: u8,
    pub col: u32,
    pub vals: Vec<i8>,
    pub path: Vec<Hash>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ProgFraud {
    Gemm { i: u32, j: u32, w_row: Vec<i8>, w_path: Vec<Hash>, x_col: Vec<i8>, x_path: Vec<Hash>, z_row: Vec<i32>, z_path: Vec<Hash> },
    /// `z_row` = row `i` of Z (row-wise commitment); the accused entry is `z_row[j]`.
    Act { i: u32, j: u32, z_row: Vec<i32>, z_path: Vec<Hash>, out_col: Vec<i8>, out_path: Vec<Hash> },
    /// Add / Map / Pool: openings of the source columns (in the order of `op_inputs`) and of the output column.
    Elem { i: u32, j: u32, ins: Vec<OpenedCol>, out_col: Vec<i8>, out_path: Vec<Hash> },
    /// Softmax / LayerNorm: output column `j` is not the function of input column `j`.
    Col { j: u32, in_col: Vec<i8>, in_path: Vec<Hash>, out_col: Vec<i8>, out_path: Vec<Hash> },
    /// Bias: output column `j` is not saturation(input column + bias row).
    Bias { j: u32, in_col: Vec<i8>, in_path: Vec<Hash>, b_row: Vec<i8>, b_path: Vec<Hash>, out_col: Vec<i8>, out_path: Vec<Hash> },
    /// Lut: output column `j` is not the table applied to the input column.
    Lut { j: u32, in_col: Vec<i8>, in_path: Vec<Hash>, t_row: Vec<i8>, t_path: Vec<Hash>, out_col: Vec<i8>, out_path: Vec<Hash> },
    /// MulW: the entry (i, j) of the output is not (x[i][j] * W[i][j]) >> shift; `w_row` = row `i` of the weight block, `in_col` / `out_col` = columns `j`.
    MulW { i: u32, j: u32, in_col: Vec<i8>, in_path: Vec<Hash>, w_row: Vec<i8>, w_path: Vec<Hash>, out_col: Vec<i8>, out_path: Vec<Hash> },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ProgTx {
    Submit { x0_root: Hash, w_roots: Vec<Hash>, ops: Vec<Op>, escrow: u64, deadline: u64 },
    Claim { job_id: JobId },
    /// z_root only makes sense for a Gemm (otherwise [0;32]).
    PostOp { job_id: JobId, op: u8, z_root: Hash, x_root: Hash },
    Accept { job_id: JobId },
    Fraud { job_id: JobId, op: u8, kind: ProgFraud },
    Settle { job_id: JobId },
    /// Publishes SEVERAL consecutive operations in one transaction: `posts[i] = (z_root, x_root)` of operation `op + i`. Exactly equivalent
    /// to a sequence of `PostOp` (same checks for each, same imperative order); reduces the load on the chain by a factor ~ #ops.
    PostBatch { job_id: JobId, op: u8, posts: Vec<(Hash, Hash)> },
    /// DATA AVAILABILITY: the client demands, ON-CHAIN, column `col` of tensor `tensor` (1..=#ops: output of operation `tensor - 1`)
    /// of a finished program. It pays FETCH_FEE (paid to the worker if it serves). As for simple product jobs (docs/07 section 11).
    Fetch { job_id: JobId, tensor: u8, col: u32 },
    /// The worker serves the requested column with its Merkle path (against the published root of the tensor); the dispute window is extended.
    Serve { job_id: JobId, tensor: u8, col: u32, vals: Vec<i8>, path: Vec<Hash> },
    /// Anyone: a data request was left unanswered after SERVE_SECS -> the worker is penalised (like a proven fraud).
    SlashUnserved { job_id: JobId },
}

pub fn prog_id(client: &PubKey, nonce: u64, x0_root: &Hash, w_roots: &[Hash], ops: &[Op]) -> JobId {
    let nb = nonce.to_le_bytes();
    let ob = bincode::serialize(ops).unwrap();
    let mut parts: Vec<&[u8]> = vec![client, &nb, x0_root, &ob];
    for w in w_roots {
        parts.push(w);
    }
    h("vearl/prog/v1", &parts)
}

// ---------------- semantics of the operations ----------------

pub fn act_fn(z: i32, shift: u8, relu: bool) -> i8 {
    let v = z >> shift;
    (if relu { v.clamp(0, 64) } else { v.clamp(-64, 64) }) as i8
}

/// Value of the table for cell `v`: index = clamp(v, -64, 63) + 64 (128 entries; +64 is treated as +63; the other cells of the row are ignored).
pub fn lut_apply(table: &[i8], v: i8) -> i8 {
    table[(v.clamp(-64, 63) as i32 + 64) as usize]
}

fn sat(v: i32, relu: bool) -> i8 {
    (if relu { v.clamp(0, 64) } else { v.clamp(-64, 64) }) as i8
}

/// Op::Mul on one cell: (x * g) >> shift (arithmetic shift), saturated to +-64.
pub fn mul_apply(x: i8, g: i8, shift: u8) -> i8 {
    ((x as i32 * g as i32) >> shift).clamp(-64, 64) as i8
}

/// exp2(-d/8) in 16-bit fixed point: floor(65536 * 2^(-d/8)), d = 0..=128 (exact table, computed offline in
/// 80-digit decimal arithmetic; no floating-point function in consensus).
pub const EXP2_TABLE: [u32; 129] = [
    65536, 60096, 55108, 50535, 46340, 42494, 38967, 35733,
    32768, 30048, 27554, 25267, 23170, 21247, 19483, 17866,
    16384, 15024, 13777, 12633, 11585, 10623, 9741, 8933,
    8192, 7512, 6888, 6316, 5792, 5311, 4870, 4466,
    4096, 3756, 3444, 3158, 2896, 2655, 2435, 2233,
    2048, 1878, 1722, 1579, 1448, 1327, 1217, 1116,
    1024, 939, 861, 789, 724, 663, 608, 558,
    512, 469, 430, 394, 362, 331, 304, 279,
    256, 234, 215, 197, 181, 165, 152, 139,
    128, 117, 107, 98, 90, 82, 76, 69,
    64, 58, 53, 49, 45, 41, 38, 34,
    32, 29, 26, 24, 22, 20, 19, 17,
    16, 14, 13, 12, 11, 10, 9, 8,
    8, 7, 6, 6, 5, 5, 4, 4,
    4, 3, 3, 3, 2, 2, 2, 2,
    2, 1, 1, 1, 1, 1, 1, 1,
    1,
];

/// Integer softmax of a column: e_j = EXP2_TABLE[max - v_j], output = (64 e_j + S/2) / S in [0,64]; cells >= len: 0.
pub fn softmax_col(col: &[i8], len: usize) -> Vec<i8> {
    softmax_col_s(col, len, 0)
}

/// `softmax_col` with the exponent step 2^scale: e_j = EXP2_TABLE[min(128, (max - v_j) << scale)].
pub fn softmax_col_s(col: &[i8], len: usize, scale: u8) -> Vec<i8> {
    let n = col.len();
    let len = len.min(n);
    let mut out = vec![0i8; n];
    if len == 0 {
        return out;
    }
    let m = col[..len].iter().map(|v| *v as i32).max().unwrap();
    let e: Vec<u64> = col[..len].iter().map(|v| EXP2_TABLE[(((m - *v as i32) << scale).clamp(0, 128)) as usize] as u64).collect();
    let s: u64 = e.iter().sum();
    for (j, ej) in e.iter().enumerate() {
        out[j] = ((ej * 64 + s / 2) / s) as i8;
    }
    out
}

/// Fine part of the integer softmax of a column: q4096_j - 64 q64_j with q_k = (k e_j + S/2) / S (same e_j, S as `softmax_col`);
/// in [-32, 32]; cells >= len: 0.
pub fn softmax_lo_col(col: &[i8], len: usize) -> Vec<i8> {
    softmax_lo_col_s(col, len, 0)
}

/// `softmax_lo_col` with the exponent step 2^scale.
pub fn softmax_lo_col_s(col: &[i8], len: usize, scale: u8) -> Vec<i8> {
    let n = col.len();
    let len = len.min(n);
    let mut out = vec![0i8; n];
    if len == 0 {
        return out;
    }
    let m = col[..len].iter().map(|v| *v as i32).max().unwrap();
    let e: Vec<u64> = col[..len].iter().map(|v| EXP2_TABLE[(((m - *v as i32) << scale).clamp(0, 128)) as usize] as u64).collect();
    let s: u64 = e.iter().sum();
    for (j, ej) in e.iter().enumerate() {
        let hi = ((ej * 64 + s / 2) / s) as i64;
        let fine = ((ej * 4096 + s / 2) / s) as i64;
        out[j] = (fine - 64 * hi).clamp(-64, 64) as i8;
    }
    out
}

/// Integer square root (integer part), without floats.
pub fn isqrt(x: u64) -> u64 {
    if x < 2 {
        return x;
    }
    let (mut lo, mut hi) = (1u64, x.min(u32::MAX as u64));
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if mid * mid <= x {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Integer layer normalisation of a column: mu = round(sum/len), var = sum((v-mu)^2)/len,
/// std = round(sqrt(var+1)) (at least 1), output = clamp(floor(((v-mu)*16 + std/2) / std), -64, 64); cells >= len: 0.
pub fn layernorm_col(col: &[i8], len: usize) -> Vec<i8> {
    let n = col.len();
    let len = len.min(n);
    let mut out = vec![0i8; n];
    if len == 0 {
        return out;
    }
    let l = len as i64;
    let sum: i64 = col[..len].iter().map(|v| *v as i64).sum();
    let mu = (sum + l / 2).div_euclid(l); // mean rounded to nearest (bias <= 1/2)
    let var: i64 = col[..len].iter().map(|v| (*v as i64 - mu) * (*v as i64 - mu)).sum::<i64>() / l;
    let std = ((isqrt(4 * (var as u64 + 1)) + 1) / 2).max(1) as i64; // root rounded to nearest: floor(sqrt(4x)) = 2*sqrt(x)
    for j in 0..len {
        out[j] = (((col[j] as i64 - mu) * 16 + std / 2).div_euclid(std)).clamp(-64, 64) as i8;
    }
    out
}

/// Integer RMS normalisation of a column: for each group of `group` rows of the first `len` rows: r = isqrt(floor(256 sum(v^2) / group) + 256)
/// (= 16 sqrt(ms + 1) with 4 fractional bits, at least 16), output = clamp(floor((256 v + r/2) / r), -64, 64) (= 16 v / rms); cells >= len: 0.
/// (LayerNorm without the mean; the 4 fractional bits keep the amplitude exact to 1/(32 rms) when the input has a few LSB of rms.)
pub fn rmsnorm_col(col: &[i8], len: usize, group: usize) -> Vec<i8> {
    rmsnorm_col_s(col, len, group, 0)
}

/// `rmsnorm_col` with the output shift: output = clamp(floor((256 v + R/2) / R), -64, 64), R = r << shift, i.e. 16 v / (rms 2^shift): the output
/// unit is 2^shift / 16 of the normalised value, the output range 4 * 2^shift sigma (the normalised queries and keys of the Qwen3 models reach 8 sigma).
pub fn rmsnorm_col_s(col: &[i8], len: usize, group: usize, shift: u8) -> Vec<i8> {
    let n = col.len();
    let len = len.min(n);
    let mut out = vec![0i8; n];
    if len == 0 || group == 0 {
        return out;
    }
    let g = group.min(len);
    let mut start = 0;
    while start + g <= len {
        let sum: i64 = col[start..start + g].iter().map(|v| (*v as i64) * (*v as i64)).sum::<i64>();
        let r = (isqrt((256 * sum / g as i64) as u64 + 256) as i64) << shift;
        for j in start..start + g {
            out[j] = (((col[j] as i64) * 256 + r / 2).div_euclid(r)).clamp(-64, 64) as i8;
        }
        start += g;
    }
    out
}

/// Complete output of column `j` of a softmax / normalisation operation from the input column.
pub fn col_op(op: &Op, j: usize, in_col: &[i8]) -> Vec<i8> {
    match op {
        Op::Softmax { len, .. } => softmax_col(in_col, *len as usize),
        Op::SoftmaxLo { len, .. } => softmax_lo_col(in_col, *len as usize),
        Op::LayerNorm { len, .. } => layernorm_col(in_col, *len as usize),
        Op::RmsNorm { len, group, shift, .. } => rmsnorm_col_s(in_col, *len as usize, *group as usize, *shift),
        Op::CausalSoftmax { len, l, fine, .. } => {
            let eff = (*len as usize).min(j % (*l).max(1) as usize + 1);
            if *fine {
                softmax_lo_col(in_col, eff)
            } else {
                softmax_col(in_col, eff)
            }
        }
        Op::CausalSoftmaxS { len, l, fine, scale, .. } => {
            let eff = (*len as usize).min(j % (*l).max(1) as usize + 1);
            if *fine {
                softmax_lo_col_s(in_col, eff, *scale)
            } else {
                softmax_col_s(in_col, eff, *scale)
            }
        }
        _ => vec![0; in_col.len()],
    }
}

/// Source cells (tensor, row, column) on which output (i, j) of a NON-Gemm operation depends. Empty => output 0.
pub fn op_inputs(op: &Op, i: usize, j: usize, n: usize) -> Vec<(usize, usize, usize)> {
    match op {
        Op::Gemm { .. } | Op::GemmAT { .. } | Op::Softmax { .. } | Op::SoftmaxLo { .. } | Op::LayerNorm { .. } | Op::CausalSoftmax { .. } | Op::CausalSoftmaxS { .. } | Op::Bias { .. }
        | Op::Lut { .. } | Op::Mul { .. } | Op::RmsNorm { .. } | Op::MulW { .. } => vec![],
        Op::Add { a, b, .. } | Op::MulT { a, b, .. } => vec![(*a as usize, i, j), (*b as usize, i, j)],
        Op::Map { x, map } => match map {
            MapSpec::Transpose => vec![(*x as usize, j, i)],
            MapSpec::Flatten { c, h, w, batch } => {
                let (c, h, w, batch) = (*c as usize, *h as usize, *w as usize, *batch as usize);
                if i >= c * h * w || j >= batch || i >= n || j >= n {
                    return vec![];
                }
                let (ch, rem) = (i / (h * w), i % (h * w));
                let (py, px) = (rem / w, rem % w);
                vec![(*x as usize, ch, (py * w + px) * batch + j)]
            }
            MapSpec::Shift { dr, dc } => {
                let (si, sj) = (i + *dr as usize, j + *dc as usize);
                if i >= n || j >= n || si >= n || sj >= n {
                    return vec![];
                }
                vec![(*x as usize, si, sj)]
            }
            MapSpec::HeadSpread { dh, heads, l } => {
                let (dh, heads, l) = (*dh as usize, *heads as usize, *l as usize);
                if dh == 0 || l == 0 || i >= heads * dh || i >= n || j >= n {
                    return vec![];
                }
                let h = i / dh;
                if j < h * l || j >= h * l + l {
                    return vec![];
                }
                vec![(*x as usize, i, j - h * l)]
            }
            MapSpec::HeadGather { dh, heads, l } => {
                let (dh, heads, l) = (*dh as usize, *heads as usize, *l as usize);
                if dh == 0 || i >= heads * dh || j >= l || i >= n || j >= n {
                    return vec![];
                }
                let sj = (i / dh) * l + j;
                if sj >= n {
                    return vec![];
                }
                vec![(*x as usize, i, sj)]
            }
            MapSpec::Conv { c_in, h, w, batch, kh, kw, pad, stride } => {
                let (c_in, h, w, batch, kh, kw, pad, stride) = (*c_in as usize, *h as usize, *w as usize, *batch as usize, *kh as usize, *kw as usize, *pad as usize, *stride as usize);
                let Some((oh, ow)) = conv_out_dims(h, w, kh, kw, pad, stride) else { return vec![] };
                let (k_rows, cols) = (c_in * kh * kw, oh * ow * batch);
                if i >= k_rows || j >= cols || i >= n || j >= n {
                    return vec![];
                }
                let (c, rem) = (i / (kh * kw), i % (kh * kw));
                let (dy, dx) = (rem / kw, rem % kw);
                let (pos, b) = (j / batch, j % batch);
                let (oy, ox) = (pos / ow, pos % ow);
                let (sy, sx) = ((oy * stride + dy) as i64 - pad as i64, (ox * stride + dx) as i64 - pad as i64);
                if sy < 0 || sx < 0 || sy >= h as i64 || sx >= w as i64 {
                    return vec![];
                }
                vec![(*x as usize, c, (sy as usize * w + sx as usize) * batch + b)]
            }
            MapSpec::Im2col { c_in, h, w, batch, kh, kw, pad } => {
                let (c_in, h, w, batch, kh, kw, pad) = (*c_in as usize, *h as usize, *w as usize, *batch as usize, *kh as usize, *kw as usize, *pad as usize);
                let (k_rows, cols) = (c_in * kh * kw, h * w * batch);
                if i >= k_rows || j >= cols || i >= n || j >= n {
                    return vec![];
                }
                let (c, rem) = (i / (kh * kw), i % (kh * kw));
                let (dy, dx) = (rem / kw, rem % kw);
                let (pos, b) = (j / batch, j % batch);
                let (py, px) = (pos / w, pos % w);
                let (sy, sx) = (py as i64 + dy as i64 - pad as i64, px as i64 + dx as i64 - pad as i64);
                if sy < 0 || sx < 0 || sy >= h as i64 || sx >= w as i64 {
                    return vec![];
                }
                vec![(*x as usize, c, (sy as usize * w + sx as usize) * batch + b)]
            }
        },
        Op::Pool { x, pool } => match pool {
            PoolSpec::Avg { c, h, w, batch, k } | PoolSpec::Max { c, h, w, batch, k } => {
                let (c, h, w, batch, k) = (*c as usize, *h as usize, *w as usize, *batch as usize, *k as usize);
                let (oh, ow) = (h / k, w / k);
                if k == 0 || i >= c || j >= oh * ow * batch || i >= n || j >= n {
                    return vec![];
                }
                let (pos, b) = (j / batch, j % batch);
                let (y, xx) = (pos / ow, pos % ow);
                let mut v = Vec::with_capacity(k * k);
                for dy in 0..k {
                    for dx in 0..k {
                        v.push((*x as usize, i, ((y * k + dy) * w + xx * k + dx) * batch + b));
                    }
                }
                v
            }
        },
    }
}

/// Output value of a non-Gemm operation from the values of the source cells (in the order of `op_inputs`).
pub fn op_eval(op: &Op, vals: &[i8]) -> i8 {
    match op {
        Op::Gemm { .. } | Op::GemmAT { .. } | Op::Softmax { .. } | Op::SoftmaxLo { .. } | Op::LayerNorm { .. } | Op::CausalSoftmax { .. } | Op::CausalSoftmaxS { .. } | Op::Bias { .. }
        | Op::Lut { .. } | Op::Mul { .. } | Op::RmsNorm { .. } | Op::MulW { .. } => 0,
        Op::Add { relu, .. } => sat(vals[0] as i32 + vals[1] as i32, *relu),
        Op::MulT { shift, .. } => mul_apply(vals[0], vals[1], *shift),
        Op::Map { .. } => vals.first().copied().unwrap_or(0),
        Op::Pool { pool, .. } => match pool {
            PoolSpec::Avg { k, .. } => {
                if vals.is_empty() {
                    return 0;
                }
                let s: i32 = vals.iter().map(|v| *v as i32).sum();
                let w = (*k as i32) * (*k as i32);
                ((s + w / 2).div_euclid(w)) as i8 // rounded mean (div_euclid: rounds down, defined for negatives)
            }
            PoolSpec::Max { .. } => vals.iter().copied().max().unwrap_or(0),
        },
    }
}

/// Bounds (rows, columns) outside which the output of an element-wise operation is structurally 0.
pub fn elem_region(op: &Op, n: usize) -> (usize, usize) {
    let (r, c) = match op {
        Op::Map { map: MapSpec::Flatten { c, h, w, batch }, .. } => (*c as usize * *h as usize * *w as usize, *batch as usize),
        Op::Map { map: MapSpec::Im2col { c_in, h, w, batch, kh, kw, .. }, .. } => (*c_in as usize * *kh as usize * *kw as usize, *h as usize * *w as usize * *batch as usize),
        Op::Map { map: MapSpec::Conv { c_in, h, w, batch, kh, kw, pad, stride }, .. } => match conv_out_dims(*h as usize, *w as usize, *kh as usize, *kw as usize, *pad as usize, *stride as usize) {
            Some((oh, ow)) => (*c_in as usize * *kh as usize * *kw as usize, oh * ow * *batch as usize),
            None => (0, 0),
        },
        Op::Pool { pool: PoolSpec::Avg { c, h, w, batch, k } | PoolSpec::Max { c, h, w, batch, k }, .. } => (*c as usize, (*h as usize / (*k as usize).max(1)) * (*w as usize / (*k as usize).max(1)) * *batch as usize),
        Op::Map { map: MapSpec::HeadSpread { dh, heads, l }, .. } => (*heads as usize * *dh as usize, *heads as usize * *l as usize),
        Op::Map { map: MapSpec::HeadGather { dh, heads, l }, .. } => (*heads as usize * *dh as usize, *l as usize),
        _ => (n, n),
    };
    (r.min(n), c.min(n))
}

/// Evaluates a complete operation (worker / client side). Returns (Z int32 for a Gemm, output tensor).
pub fn eval_op(op: &Op, tensors: &[Vec<i8>], weights: &[Vec<i8>], n: usize) -> (Option<Vec<i32>>, Vec<i8>) {
    match op {
        Op::Gemm { w, x, shift, relu } => {
            let z = plain_gemm(&weights[*w as usize], &tensors[*x as usize], n);
            let out = z.iter().map(|&v| act_fn(v, *shift, *relu)).collect();
            (Some(z), out)
        }
        Op::GemmAT { a, x, shift, relu } => {
            let ta = &tensors[*a as usize];
            let at: Vec<i8> = (0..n * n).map(|idx| ta[(idx % n) * n + idx / n]).collect(); // A[i][k] = T_a[k][i]
            let z = plain_gemm(&at, &tensors[*x as usize], n);
            let out = z.iter().map(|&v| act_fn(v, *shift, *relu)).collect();
            (Some(z), out)
        }
        Op::Softmax { x, .. } | Op::SoftmaxLo { x, .. } | Op::LayerNorm { x, .. } | Op::CausalSoftmax { x, .. } | Op::CausalSoftmaxS { x, .. } | Op::RmsNorm { x, .. } => {
            let tx = &tensors[*x as usize];
            let mut out = vec![0i8; n * n];
            for j in 0..n {
                let col: Vec<i8> = (0..n).map(|i| tx[i * n + j]).collect();
                for (i, v) in col_op(op, j, &col).into_iter().enumerate() {
                    out[i * n + j] = v;
                }
            }
            (None, out)
        }
        Op::Lut { x, w, row } => {
            let (tx, table) = (&tensors[*x as usize], &weights[*w as usize][*row as usize * n..(*row as usize + 1) * n]);
            (None, tx.iter().map(|v| lut_apply(table, *v)).collect())
        }
        Op::Bias { x, w, row, relu } => {
            let (tx, brow) = (&tensors[*x as usize], &weights[*w as usize][*row as usize * n..(*row as usize + 1) * n]);
            (None, (0..n * n).map(|idx| sat(tx[idx] as i32 + brow[idx / n] as i32, *relu)).collect())
        }
        Op::Mul { x, w, row, shift } => {
            let (tx, grow) = (&tensors[*x as usize], &weights[*w as usize][*row as usize * n..(*row as usize + 1) * n]);
            (None, (0..n * n).map(|idx| mul_apply(tx[idx], grow[idx / n], *shift)).collect())
        }
        Op::MulW { x, w, shift } => {
            let (tx, tw) = (&tensors[*x as usize], &weights[*w as usize]);
            (None, (0..n * n).map(|idx| mul_apply(tx[idx], tw[idx], *shift)).collect())
        }
        Op::MulT { a, b, shift } => {
            let (ta, tb) = (&tensors[*a as usize], &tensors[*b as usize]);
            (None, (0..n * n).map(|idx| mul_apply(ta[idx], tb[idx], *shift)).collect())
        }
        _ => {
            let mut out = vec![0i8; n * n];
            match op {
                Op::Add { a, b, relu } => {
                    let (ta, tb) = (&tensors[*a as usize], &tensors[*b as usize]);
                    for idx in 0..n * n {
                        out[idx] = sat(ta[idx] as i32 + tb[idx] as i32, *relu);
                    }
                }
                Op::Map { x, map: MapSpec::Transpose } => {
                    let tx = &tensors[*x as usize];
                    for i in 0..n {
                        for j in 0..n {
                            out[i * n + j] = tx[j * n + i];
                        }
                    }
                }
                Op::Map { x, map: MapSpec::Shift { dr, dc } } => {
                    let (tx, dr, dc) = (&tensors[*x as usize], *dr as usize, *dc as usize);
                    for i in 0..n.saturating_sub(dr) {
                        out[i * n..i * n + n.saturating_sub(dc)].copy_from_slice(&tx[(i + dr) * n + dc..(i + dr) * n + n]);
                    }
                }
                _ => {
                    // only the useful region can be non-zero (the other cells have no source): we evaluate only that one
                    let (rows, cols) = elem_region(op, n);
                    for i in 0..rows {
                        for j in 0..cols {
                            let vals: Vec<i8> = op_inputs(op, i, j, n).iter().map(|&(t, si, sj)| tensors[t][si * n + sj]).collect();
                            out[i * n + j] = op_eval(op, &vals);
                        }
                    }
                }
            }
            (None, out)
        }
    }
}

/// Executes the whole program: (tensors T_1.., Z of the Gemms per operation).
pub fn run_program(x0: &[i8], weights: &[Vec<i8>], ops: &[Op], n: usize) -> (Vec<Vec<i8>>, Vec<Option<Vec<i32>>>) {
    let mut tensors = vec![x0.to_vec()];
    let mut zs = vec![];
    for op in ops {
        let (z, out) = eval_op(op, &tensors, weights, n);
        zs.push(z);
        tensors.push(out);
    }
    (tensors, zs)
}

// ---------------- state ----------------

#[derive(Clone, Debug, PartialEq)]
pub struct OpPost {
    pub z_root: Hash,
    pub x_root: Hash,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ProgStatus {
    Open,
    Working { worker: PubKey, bond: u64 },
    Complete { worker: PubKey, bond: u64, at: u64, at_ts: u64 },
    Settled,
    Slashed,
    Expired,
}

#[derive(Clone, Debug)]
pub struct ProgJob {
    pub client: PubKey,
    pub x0_root: Hash,
    pub w_roots: Vec<Hash>,
    pub ops: Vec<Op>,
    pub escrow_net: u64,
    pub deadline: u64,
    pub status: ProgStatus,
    pub rebate_left: u64,
    pub posted: Vec<OpPost>,
    /// Pending data requests and the client's fees set aside.
    pub fetches: Vec<PFetch>,
    pub fetch_held: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PFetch {
    pub tensor: u8,
    pub col: u32,
    pub deadline_ts: u64,
}

impl ProgJob {
    /// Root of tensor `t` (0 = client input) if already known.
    pub fn tensor_root(&self, t: usize) -> Option<Hash> {
        if t == 0 {
            Some(self.x0_root)
        } else {
            self.posted.get(t - 1).map(|p| p.x_root)
        }
    }
    pub fn gemm_count(&self) -> usize {
        self.ops.iter().filter(|o| o.is_gemm()).count()
    }
    /// (root of W, root of the input tensor) of a Gemm, if its input is already published.
    pub fn gemm_roots(&self, op: usize) -> Option<(Hash, Hash)> {
        match self.ops.get(op)? {
            Op::Gemm { w, x, .. } => Some((self.w_roots[*w as usize], self.tensor_root(*x as usize)?)),
            // A = T_a^T: its row-wise root is the column-wise root of T_a
            Op::GemmAT { a, x, .. } => Some((self.tensor_root(*a as usize)?, self.tensor_root(*x as usize)?)),
            _ => None,
        }
    }
}

fn bad<T>(m: &str) -> Result<T, String> {
    Err(m.to_string())
}

pub fn validate_ops(ops: &[Op], n_weights: usize, n: usize) -> Result<(), String> {
    if ops.is_empty() || ops.len() > MAX_OPS || n_weights > MAX_WEIGHTS {
        return bad("bad program size");
    }
    for (k, op) in ops.iter().enumerate() {
        let ok_t = |t: u8| (t as usize) <= k; // tensors 0..=k exist before operation k
        let ok = match op {
            Op::Gemm { w, x, shift, .. } => (*w as usize) < n_weights && ok_t(*x) && *shift <= crate::mlp::MAX_SHIFT,
            Op::GemmAT { a, x, shift, .. } => ok_t(*a) && ok_t(*x) && *shift <= crate::mlp::MAX_SHIFT,
            Op::Softmax { x, len } | Op::SoftmaxLo { x, len } | Op::LayerNorm { x, len } => ok_t(*x) && *x >= 1 && *len >= 1 && (*len as usize) <= n,
            Op::RmsNorm { x, len, group, shift } => {
                ok_t(*x) && *x >= 1 && *len >= 1 && (*len as usize) <= n && *group >= 1 && group <= len && *len % *group == 0 && *shift <= MAX_RMS_SHIFT
            }
            Op::MulT { a, b, shift } => ok_t(*a) && ok_t(*b) && *shift <= crate::mlp::MAX_SHIFT,
            Op::MulW { x, w, shift } => ok_t(*x) && (*w as usize) < n_weights && *shift <= crate::mlp::MAX_SHIFT,
            Op::CausalSoftmax { x, len, l, .. } => ok_t(*x) && *x >= 1 && *len >= 1 && (*len as usize) <= n && *l >= 1 && (*l as usize) <= n,
            Op::CausalSoftmaxS { x, len, l, scale, .. } => {
                ok_t(*x) && *x >= 1 && *len >= 1 && (*len as usize) <= n && *l >= 1 && (*l as usize) <= n && *scale <= MAX_SOFTMAX_SCALE
            }
            Op::Bias { x, w, row, .. } => ok_t(*x) && (*w as usize) < n_weights && (*row as usize) < n,
            Op::Lut { x, w, row } => ok_t(*x) && (*w as usize) < n_weights && (*row as usize) < n,
            Op::Mul { x, w, row, shift } => ok_t(*x) && (*w as usize) < n_weights && (*row as usize) < n && *shift <= crate::mlp::MAX_SHIFT,
            Op::Add { a, b, .. } => ok_t(*a) && ok_t(*b),
            Op::Map { x, map } => {
                ok_t(*x)
                    && match map {
                        MapSpec::Transpose => true,
                        MapSpec::Flatten { c, h, w, batch } => {
                            *c > 0 && *h > 0 && *w > 0 && *batch > 0 && *c as usize * *h as usize * *w as usize <= n && *h as usize * *w as usize * *batch as usize <= n && *c as usize <= n
                        }
                        MapSpec::Shift { dr, dc } => (*dr as usize) < n && (*dc as usize) < n,
                        MapSpec::HeadSpread { dh, heads, l } | MapSpec::HeadGather { dh, heads, l } => {
                            *dh > 0 && *heads > 0 && *l > 0 && *heads as usize * *dh as usize <= n && *heads as usize * *l as usize <= n
                        }
                        MapSpec::Conv { c_in, h, w, batch, kh, kw, pad, stride } => {
                            let (kk, inc) = (*c_in as usize * *kh as usize * *kw as usize, *h as usize * *w as usize * *batch as usize);
                            match conv_out_dims(*h as usize, *w as usize, *kh as usize, *kw as usize, *pad as usize, *stride as usize) {
                                Some((oh, ow)) => *c_in > 0 && *h > 0 && *w > 0 && *batch > 0 && *kh > 0 && *kw > 0 && kk <= n && inc <= n && oh * ow * *batch as usize <= n && *c_in as usize <= n && (*pad as usize) < *kh as usize + *kw as usize && *stride >= 1,
                                None => false,
                            }
                        }
                        MapSpec::Im2col { c_in, h, w, batch, kh, kw, pad } => {
                            let (kk, cols) = (*c_in as usize * *kh as usize * *kw as usize, *h as usize * *w as usize * *batch as usize);
                            *c_in > 0 && *h > 0 && *w > 0 && *batch > 0 && *kh > 0 && *kw > 0 && kk <= n && cols <= n && *c_in as usize <= n && (*pad as usize) < *kh as usize + *kw as usize
                        }
                    }
            }
            Op::Pool { x, pool } => {
                ok_t(*x)
                    && match pool {
                        PoolSpec::Avg { c, h, w, batch, k } | PoolSpec::Max { c, h, w, batch, k } => *k >= 1 && *h % *k as u16 == 0 && *w % *k as u16 == 0 && *c as usize <= n && *h as usize * *w as usize * *batch as usize <= n,
                    }
            }
        };
        if !ok {
            return bad("invalid operation");
        }
    }
    Ok(())
}

impl State {
    pub(crate) fn apply_prog(&mut self, from: &PubKey, nonce: u64, m: &ProgTx, height: u64, ts: u64, p: &WorkParams) -> Result<(), String> {
        match m {
            ProgTx::Submit { x0_root, w_roots, ops, escrow, deadline } => {
                validate_ops(ops, w_roots.len(), p.n)?;
                let gemms = ops.iter().filter(|o| o.is_gemm()).count() as u64;
                if gemms == 0 {
                    return bad("a program needs at least one Gemm (the mined work)");
                }
                if *escrow < MIN_ESCROW * gemms {
                    return bad("escrow below floor price (per Gemm)");
                }
                if *deadline <= height {
                    return bad("deadline in the past");
                }
                let id = prog_id(from, nonce, x0_root, w_roots, ops);
                if self.progs.contains_key(&id) {
                    return bad("duplicate job");
                }
                self.debit(from, *escrow)?;
                let burn = bps(*escrow, BURN_BPS);
                self.burned += burn;
                self.progs.insert(
                    id,
                    ProgJob { client: *from, x0_root: *x0_root, w_roots: w_roots.clone(), ops: ops.clone(), escrow_net: escrow - burn, deadline: *deadline, status: ProgStatus::Open, rebate_left: burn, posted: vec![], fetches: vec![], fetch_held: 0 },
                );
            }
            ProgTx::Claim { job_id } => {
                let (net, status, deadline) = {
                    let j = self.progs.get(job_id).ok_or("unknown job")?;
                    (j.escrow_net, j.status.clone(), j.deadline)
                };
                if status != ProgStatus::Open || height > deadline {
                    return bad("job not claimable");
                }
                let bond = bps(net, BOND_BPS);
                self.debit(from, bond)?;
                self.progs.get_mut(job_id).unwrap().status = ProgStatus::Working { worker: *from, bond };
            }
            ProgTx::PostOp { job_id, op, z_root, x_root } => {
                let j = self.progs.get_mut(job_id).ok_or("unknown job")?;
                let ProgStatus::Working { worker, bond } = j.status.clone() else { return bad("job is not in progress") };
                if worker != *from || height > j.deadline {
                    return bad("cannot post");
                }
                if *op as usize != j.posted.len() || j.posted.len() >= j.ops.len() {
                    return bad("operations must be posted in order");
                }
                if !j.ops[*op as usize].is_gemm() && *z_root != [0u8; 32] {
                    return bad("z_root only makes sense for a Gemm");
                }
                j.posted.push(OpPost { z_root: *z_root, x_root: *x_root });
                if j.posted.len() == j.ops.len() {
                    j.status = ProgStatus::Complete { worker, bond, at: height, at_ts: ts };
                }
            }
            ProgTx::PostBatch { job_id, op, posts } => {
                let j = self.progs.get_mut(job_id).ok_or("unknown job")?;
                let ProgStatus::Working { worker, bond } = j.status.clone() else { return bad("job is not in progress") };
                if worker != *from || height > j.deadline {
                    return bad("cannot post");
                }
                if posts.is_empty() || *op as usize != j.posted.len() || j.posted.len() + posts.len() > j.ops.len() {
                    return bad("operations must be posted in order");
                }
                for (i, (z_root, _)) in posts.iter().enumerate() {
                    if !j.ops[*op as usize + i].is_gemm() && *z_root != [0u8; 32] {
                        return bad("z_root only makes sense for a Gemm");
                    }
                }
                j.posted.extend(posts.iter().map(|(z_root, x_root)| OpPost { z_root: *z_root, x_root: *x_root }));
                if j.posted.len() == j.ops.len() {
                    j.status = ProgStatus::Complete { worker, bond, at: height, at_ts: ts };
                }
            }
            ProgTx::Accept { job_id } => {
                let (client, status, net) = {
                    let j = self.progs.get(job_id).ok_or("unknown job")?;
                    (j.client, j.status.clone(), j.escrow_net)
                };
                match status {
                    ProgStatus::Complete { worker, bond, .. } if client == *from => {
                        self.credit(&worker, net + bond);
                        self.rep.entry(worker).or_default().done += 1;
                        let jm = self.progs.get_mut(job_id).unwrap();
                        jm.status = ProgStatus::Settled;
                        let held = std::mem::take(&mut jm.fetch_held);
                        jm.fetches.clear();
                        self.credit(&client, held);
                    }
                    _ => return bad("cannot accept"),
                }
            }
            ProgTx::Fraud { job_id, op, kind } => {
                let job = self.progs.get(job_id).ok_or("unknown job")?.clone();
                let (bond, accused) = match &job.status {
                    ProgStatus::Working { bond, worker } => (*bond, *worker),
                    ProgStatus::Complete { bond, at, at_ts, worker } => {
                        if !self.timing.challenge_open(height, ts, *at, *at_ts) {
                            return bad("challenge window closed");
                        }
                        (*bond, *worker)
                    }
                    _ => return bad("no result to challenge"),
                };
                verify_prog_fraud(&job, *op as usize, kind, p)?;
                let reward = bond / 2;
                self.credit(from, reward);
                self.burned += bond - reward;
                self.credit(&job.client, job.escrow_net + job.fetch_held);
                self.rep.entry(accused).or_default().slashed += 1;
                let jm = self.progs.get_mut(job_id).unwrap();
                jm.status = ProgStatus::Slashed;
                jm.fetch_held = 0;
                jm.fetches.clear();
            }
            ProgTx::Fetch { job_id, tensor, col } => {
                let j = self.progs.get(job_id).ok_or("unknown job")?;
                let ProgStatus::Complete { at, at_ts, .. } = j.status.clone() else { return bad("no posted result") };
                if j.client != *from {
                    return bad("only the client can request data");
                }
                if *tensor == 0 || *tensor as usize > j.ops.len() || *col as usize >= p.n {
                    return bad("index out of range");
                }
                if !self.timing.challenge_open(height, ts, at, at_ts) {
                    return bad("challenge window closed");
                }
                if j.fetches.len() >= MAX_PENDING_FETCHES || j.fetches.iter().any(|f| f.tensor == *tensor && f.col == *col) {
                    return bad("too many or duplicate pending requests");
                }
                self.debit(from, FETCH_FEE)?;
                let jm = self.progs.get_mut(job_id).unwrap();
                jm.fetch_held += FETCH_FEE;
                jm.fetches.push(PFetch { tensor: *tensor, col: *col, deadline_ts: ts + self.timing.serve_secs });
            }
            ProgTx::Serve { job_id, tensor, col, vals, path } => {
                let j = self.progs.get(job_id).ok_or("unknown job")?;
                let ProgStatus::Complete { worker, bond, at, at_ts } = j.status.clone() else { return bad("no posted result") };
                if worker != *from {
                    return bad("only the worker can serve");
                }
                let Some(pos) = j.fetches.iter().position(|f| f.tensor == *tensor && f.col == *col) else { return bad("no such pending request") };
                if vals.len() != p.n || vals.iter().any(|v| (*v as i32).abs() > VAL_MAX) {
                    return bad("malformed served column");
                }
                let root = j.tensor_root(*tensor as usize).ok_or("tensor not posted")?;
                if !verify_proof(&root, leaf_hash(*col as u64, &i8_bytes(vals)), *col as usize, path, p.depth()) {
                    return bad("bad Merkle proof for the served column");
                }
                let jm = self.progs.get_mut(job_id).unwrap();
                jm.fetches.remove(pos);
                jm.fetch_held -= FETCH_FEE;
                // the client must have time to verify the served column
                let new_at_ts = (ts + FETCH_EXTEND_SECS).saturating_sub(self.timing.challenge_secs).max(at_ts);
                let new_at = (height + 2).saturating_sub(CHALLENGE_WINDOW).max(at);
                jm.status = ProgStatus::Complete { worker, bond, at: new_at, at_ts: new_at_ts };
                self.credit(&worker, FETCH_FEE);
            }
            ProgTx::SlashUnserved { job_id } => {
                let j = self.progs.get(job_id).ok_or("unknown job")?.clone();
                let ProgStatus::Complete { worker, bond, .. } = j.status else { return bad("no posted result") };
                if !j.fetches.iter().any(|f| ts > f.deadline_ts) {
                    return bad("no expired data request");
                }
                let reward = bond / 2;
                self.credit(from, reward);
                self.burned += bond - reward;
                self.credit(&j.client, j.escrow_net + j.fetch_held);
                self.rep.entry(worker).or_default().slashed += 1;
                let jm = self.progs.get_mut(job_id).unwrap();
                jm.fetch_held = 0;
                jm.fetches.clear();
                jm.status = ProgStatus::Slashed;
            }
            ProgTx::Settle { job_id } => {
                let (client, status, net, deadline, pending) = {
                    let j = self.progs.get(job_id).ok_or("unknown job")?;
                    (j.client, j.status.clone(), j.escrow_net, j.deadline, !j.fetches.is_empty())
                };
                match status {
                    ProgStatus::Complete { .. } if pending => return bad("data requests pending: serve or slash first"),
                    ProgStatus::Complete { worker, bond, at, at_ts } if !self.timing.challenge_open(height, ts, at, at_ts) => {
                        self.credit(&worker, net + bond);
                        self.rep.entry(worker).or_default().done += 1;
                        self.progs.get_mut(job_id).unwrap().status = ProgStatus::Settled;
                    }
                    ProgStatus::Open if height > deadline => {
                        self.credit(&client, net);
                        self.progs.get_mut(job_id).unwrap().status = ProgStatus::Expired;
                    }
                    ProgStatus::Working { worker, bond } if height > deadline => {
                        self.rep.entry(worker).or_default().expired += 1;
                        self.burned += bond;
                        self.credit(&client, net);
                        self.progs.get_mut(job_id).unwrap().status = ProgStatus::Expired;
                    }
                    _ => return bad("cannot settle yet"),
                }
            }
        }
        Ok(())
    }

    pub(crate) fn prog_held(&self) -> u64 {
        self.progs
            .values()
            .map(|j| match &j.status {
                ProgStatus::Open => j.escrow_net,
                ProgStatus::Working { bond, .. } => j.escrow_net + bond,
                ProgStatus::Complete { bond, .. } => j.escrow_net + bond + j.fetch_held,
                _ => 0,
            })
            .sum()
    }
}

fn verify_prog_fraud(job: &ProgJob, op_idx: usize, kind: &ProgFraud, p: &WorkParams) -> Result<(), String> {
    let n = p.n;
    if op_idx >= job.posted.len() {
        return bad("operation not posted");
    }
    let op = &job.ops[op_idx];
    let out_root = job.posted[op_idx].x_root;
    match (kind, op) {
        (ProgFraud::Gemm { i, j, w_row, w_path, x_col, x_path, z_row, z_path }, Op::Gemm { .. } | Op::GemmAT { .. }) => {
            let (i, j) = (*i as usize, *j as usize);
            if i >= n || j >= n || w_row.len() != n || x_col.len() != n {
                return bad("malformed fraud proof");
            }
            if w_row.iter().chain(x_col.iter()).any(|v| (*v as i32).abs() > VAL_MAX) {
                return bad("values out of range");
            }
            let (wr, xr) = job.gemm_roots(op_idx).ok_or("inputs not available")?;
            if !verify_proof(&wr, leaf_hash(i as u64, &i8_bytes(w_row)), i, w_path, p.depth()) {
                return bad("bad W opening");
            }
            if !verify_proof(&xr, leaf_hash(j as u64, &i8_bytes(x_col)), j, x_path, p.depth()) {
                return bad("bad input opening");
            }
            if !verify_row(&job.posted[op_idx].z_root, i, z_row, z_path, p) {
                return bad("bad Z opening");
            }
            let dot: i32 = w_row.iter().zip(x_col).map(|(a, b)| *a as i32 * *b as i32).sum();
            if dot == z_row[j] {
                return bad("entry is correct: not a fraud");
            }
            Ok(())
        }
        (ProgFraud::Act { i, j, z_row, z_path, out_col, out_path }, Op::Gemm { shift, relu, .. } | Op::GemmAT { shift, relu, .. }) => {
            let (i, j) = (*i as usize, *j as usize);
            if i >= n || j >= n || out_col.len() != n {
                return bad("malformed fraud proof");
            }
            if !verify_row(&job.posted[op_idx].z_root, i, z_row, z_path, p) {
                return bad("bad Z opening");
            }
            if !verify_proof(&out_root, leaf_hash(j as u64, &i8_bytes(out_col)), j, out_path, p.depth()) {
                return bad("bad output opening");
            }
            if out_col[i] == act_fn(z_row[j], *shift, *relu) {
                return bad("activation is correct: not a fraud");
            }
            Ok(())
        }
        (ProgFraud::Elem { i, j, ins, out_col, out_path }, Op::Add { .. } | Op::MulT { .. } | Op::Map { .. } | Op::Pool { .. }) => {
            let (i, j) = (*i as usize, *j as usize);
            if i >= n || j >= n || out_col.len() != n {
                return bad("malformed fraud proof");
            }
            let required = op_inputs(op, i, j, n);
            if ins.len() != required.len() {
                return bad("wrong number of source openings");
            }
            let mut vals = Vec::with_capacity(ins.len());
            for (o, &(t, si, sj)) in ins.iter().zip(&required) {
                if o.tensor as usize != t || o.col as usize != sj || o.vals.len() != n {
                    return bad("source opening does not match the operation's inputs");
                }
                let root = job.tensor_root(t).ok_or("source tensor not available")?;
                if !verify_proof(&root, leaf_hash(sj as u64, &i8_bytes(&o.vals)), sj, &o.path, p.depth()) {
                    return bad("bad source opening");
                }
                vals.push(o.vals[si]);
            }
            if !verify_proof(&out_root, leaf_hash(j as u64, &i8_bytes(out_col)), j, out_path, p.depth()) {
                return bad("bad output opening");
            }
            if out_col[i] == op_eval(op, &vals) {
                return bad("entry is correct: not a fraud");
            }
            Ok(())
        }
        (ProgFraud::Col { j, in_col, in_path, out_col, out_path }, Op::Softmax { x, .. } | Op::SoftmaxLo { x, .. } | Op::LayerNorm { x, .. } | Op::CausalSoftmax { x, .. } | Op::CausalSoftmaxS { x, .. } | Op::RmsNorm { x, .. }) => {
            let j = *j as usize;
            if j >= n || in_col.len() != n || out_col.len() != n {
                return bad("malformed fraud proof");
            }
            let in_root = job.tensor_root(*x as usize).ok_or("input tensor not available")?;
            if !verify_proof(&in_root, leaf_hash(j as u64, &i8_bytes(in_col)), j, in_path, p.depth()) {
                return bad("bad input opening");
            }
            if !verify_proof(&out_root, leaf_hash(j as u64, &i8_bytes(out_col)), j, out_path, p.depth()) {
                return bad("bad output opening");
            }
            if col_op(op, j, in_col) == *out_col {
                return bad("column is correct: not a fraud");
            }
            Ok(())
        }
        (ProgFraud::Bias { j, in_col, in_path, b_row, b_path, out_col, out_path }, Op::Bias { x, w, row, relu }) => {
            let j = *j as usize;
            if j >= n || in_col.len() != n || b_row.len() != n || out_col.len() != n {
                return bad("malformed fraud proof");
            }
            if b_row.iter().any(|v| (*v as i32).abs() > VAL_MAX) {
                return bad("values out of range");
            }
            let in_root = job.tensor_root(*x as usize).ok_or("input tensor not available")?;
            if !verify_proof(&in_root, leaf_hash(j as u64, &i8_bytes(in_col)), j, in_path, p.depth()) {
                return bad("bad input opening");
            }
            if !verify_proof(&job.w_roots[*w as usize], leaf_hash(*row as u64, &i8_bytes(b_row)), *row as usize, b_path, p.depth()) {
                return bad("bad bias opening");
            }
            if !verify_proof(&out_root, leaf_hash(j as u64, &i8_bytes(out_col)), j, out_path, p.depth()) {
                return bad("bad output opening");
            }
            let want: Vec<i8> = (0..n).map(|i| sat(in_col[i] as i32 + b_row[i] as i32, *relu)).collect();
            if want == *out_col {
                return bad("column is correct: not a fraud");
            }
            Ok(())
        }
        (ProgFraud::Bias { j, in_col, in_path, b_row, b_path, out_col, out_path }, Op::Mul { x, w, row, shift }) => {
            // same shape as Bias: input column, gamma row (arbitrary 8-bit integers), output column
            let j = *j as usize;
            if j >= n || in_col.len() != n || b_row.len() != n || out_col.len() != n {
                return bad("malformed fraud proof");
            }
            let in_root = job.tensor_root(*x as usize).ok_or("input tensor not available")?;
            if !verify_proof(&in_root, leaf_hash(j as u64, &i8_bytes(in_col)), j, in_path, p.depth()) {
                return bad("bad input opening");
            }
            if !verify_proof(&job.w_roots[*w as usize], leaf_hash(*row as u64, &i8_bytes(b_row)), *row as usize, b_path, p.depth()) {
                return bad("bad scale opening");
            }
            if !verify_proof(&out_root, leaf_hash(j as u64, &i8_bytes(out_col)), j, out_path, p.depth()) {
                return bad("bad output opening");
            }
            let want: Vec<i8> = (0..n).map(|i| mul_apply(in_col[i], b_row[i], *shift)).collect();
            if want == *out_col {
                return bad("column is correct: not a fraud");
            }
            Ok(())
        }
        (ProgFraud::MulW { i, j, in_col, in_path, w_row, w_path, out_col, out_path }, Op::MulW { x, w, shift }) => {
            let (i, j) = (*i as usize, *j as usize);
            if i >= n || j >= n || in_col.len() != n || w_row.len() != n || out_col.len() != n {
                return bad("malformed fraud proof");
            }
            if w_row.iter().chain(in_col.iter()).any(|v| (*v as i32).abs() > VAL_MAX) {
                return bad("values out of range");
            }
            let in_root = job.tensor_root(*x as usize).ok_or("input tensor not available")?;
            if !verify_proof(&in_root, leaf_hash(j as u64, &i8_bytes(in_col)), j, in_path, p.depth()) {
                return bad("bad input opening");
            }
            if !verify_proof(&job.w_roots[*w as usize], leaf_hash(i as u64, &i8_bytes(w_row)), i, w_path, p.depth()) {
                return bad("bad weight opening");
            }
            if !verify_proof(&out_root, leaf_hash(j as u64, &i8_bytes(out_col)), j, out_path, p.depth()) {
                return bad("bad output opening");
            }
            if out_col[i] == mul_apply(in_col[i], w_row[j], *shift) {
                return bad("entry is correct: not a fraud");
            }
            Ok(())
        }
        (ProgFraud::Lut { j, in_col, in_path, t_row, t_path, out_col, out_path }, Op::Lut { x, w, row }) => {
            let j = *j as usize;
            if j >= n || in_col.len() != n || t_row.len() != n || out_col.len() != n {
                return bad("malformed fraud proof");
            }
            if t_row.iter().chain(in_col.iter()).any(|v| (*v as i32).abs() > VAL_MAX) {
                return bad("values out of range");
            }
            let in_root = job.tensor_root(*x as usize).ok_or("input tensor not available")?;
            if !verify_proof(&in_root, leaf_hash(j as u64, &i8_bytes(in_col)), j, in_path, p.depth()) {
                return bad("bad input opening");
            }
            if !verify_proof(&job.w_roots[*w as usize], leaf_hash(*row as u64, &i8_bytes(t_row)), *row as usize, t_path, p.depth()) {
                return bad("bad table opening");
            }
            if !verify_proof(&out_root, leaf_hash(j as u64, &i8_bytes(out_col)), j, out_path, p.depth()) {
                return bad("bad output opening");
            }
            let want: Vec<i8> = in_col.iter().map(|v| lut_apply(t_row, *v)).collect();
            if want == *out_col {
                return bad("column is correct: not a fraud");
            }
            Ok(())
        }
        _ => bad("fraud kind does not match the operation"),
    }
}

// ---------------- demonstration program: small CNN with a residual block ----------------

/// (x0, weights, operations) of a toy CNN: 3x3 conv -> 2x2 pooling -> 3x3 conv -> residual block -> classifier.
/// Input: 4 channels, 8x8 images, batch of 2 (n = 128 columns). Deterministic given `seed`.
pub fn demo_cnn(n: usize, seed: u64) -> (Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x0 = vec![0i8; n * n];
    for c in 0..4 {
        for q in 0..128.min(n) {
            x0[c * n + q] = rng.gen_range(0..=32) as i8;
        }
    }
    // weights (rows = output channels, columns = input channels x window); zeros elsewhere
    let mk = |rows: usize, cols: usize, rng: &mut StdRng| {
        let mut w = vec![0i8; n * n];
        for r in 0..rows {
            for c in 0..cols {
                w[r * n + c] = rng.gen_range(-4..=4) as i8;
            }
        }
        w
    };
    let weights = vec![mk(8, 36, &mut rng), mk(16, 72, &mut rng), mk(16, 16, &mut rng), mk(10, 16, &mut rng)];
    let ops = vec![
        Op::Map { x: 0, map: MapSpec::Im2col { c_in: 4, h: 8, w: 8, batch: 2, kh: 3, kw: 3, pad: 1 } }, // T1 : 36 x 128
        Op::Gemm { w: 0, x: 1, shift: 5, relu: true },                                                   // T2 : 8 x 128
        Op::Pool { x: 2, pool: PoolSpec::Avg { c: 8, h: 8, w: 8, batch: 2, k: 2 } },                      // T3 : 8 x 32
        Op::Map { x: 3, map: MapSpec::Im2col { c_in: 8, h: 4, w: 4, batch: 2, kh: 3, kw: 3, pad: 1 } }, // T4 : 72 x 32
        Op::Gemm { w: 1, x: 4, shift: 5, relu: true },                                                   // T5 : 16 x 32
        Op::Gemm { w: 2, x: 5, shift: 4, relu: false },                                                  // T6 : 16 x 32
        Op::Add { a: 5, b: 6, relu: true },                                                              // T7: residual
        Op::Gemm { w: 3, x: 7, shift: 3, relu: false },                                                  // T8 : 10 x 32 (scores)
    ];
    (x0, weights, ops)
}

/// CLIENT side: finds the first wrong operation of a result (tensors T_0..T_L, Z of the Gemms) and builds the proof.
/// `ws` = client weights; the intermediate tensors are those published by the worker.
pub fn find_fraud_proof(ops: &[Op], ws: &[Vec<i8>], tens: &[Vec<i8>], zs: &[Option<Vec<i32>>], n: usize) -> Option<(u8, ProgFraud)> {
    // all the inputs are published: each operation is verified independently, in parallel; we keep the FIRST wrong one
    // (same result as the sequential loop, hence the same fraud proof)
    par_map_heavy(ops.len(), |k| check_op(k, &ops[k], ws, tens, zs, n)).into_iter().flatten().next()
}

/// Verifies operation `k` (Freivalds for a Gemm, exact recomputation otherwise); `Some` = fraud proof with valid openings.
fn check_op(k: usize, op: &Op, ws: &[Vec<i8>], tens: &[Vec<i8>], zs: &[Option<Vec<i32>>], n: usize) -> Option<(u8, ProgFraud)> {
    let out = &tens[k + 1];
    let out_tree = cols_tree(out, n);
    if op.is_col_op() {
        let x = op.col_input().expect("per-column operation");
        let tx = &tens[x as usize];
        for j in 0..n {
            let in_col: Vec<i8> = (0..n).map(|i| tx[i * n + j]).collect();
            let out_col: Vec<i8> = (0..n).map(|i| out[i * n + j]).collect();
            if col_op(op, j, &in_col) != out_col {
                return Some((k as u8, ProgFraud::Col { j: j as u32, in_col, in_path: cols_tree(tx, n).proof(j), out_col, out_path: out_tree.proof(j) }));
            }
        }
        return None;
    }
    if let Op::Bias { x, w, row, relu } = op {
        let tx = &tens[*x as usize];
        let brow = &ws[*w as usize][*row as usize * n..(*row as usize + 1) * n];
        for j in 0..n {
            let in_col: Vec<i8> = (0..n).map(|i| tx[i * n + j]).collect();
            let out_col: Vec<i8> = (0..n).map(|i| out[i * n + j]).collect();
            if (0..n).map(|i| sat(in_col[i] as i32 + brow[i] as i32, *relu)).collect::<Vec<i8>>() != out_col {
                return Some((
                    k as u8,
                    ProgFraud::Bias { j: j as u32, in_col, in_path: cols_tree(tx, n).proof(j), b_row: brow.to_vec(), b_path: rows_tree(&ws[*w as usize], n).proof(*row as usize), out_col, out_path: out_tree.proof(j) },
                ));
            }
        }
        return None;
    }
    if let Op::Mul { x, w, row, shift } = op {
        let tx = &tens[*x as usize];
        let grow = &ws[*w as usize][*row as usize * n..(*row as usize + 1) * n];
        for j in 0..n {
            let in_col: Vec<i8> = (0..n).map(|i| tx[i * n + j]).collect();
            let out_col: Vec<i8> = (0..n).map(|i| out[i * n + j]).collect();
            if (0..n).map(|i| mul_apply(in_col[i], grow[i], *shift)).collect::<Vec<i8>>() != out_col {
                return Some((
                    k as u8,
                    ProgFraud::Bias { j: j as u32, in_col, in_path: cols_tree(tx, n).proof(j), b_row: grow.to_vec(), b_path: rows_tree(&ws[*w as usize], n).proof(*row as usize), out_col, out_path: out_tree.proof(j) },
                ));
            }
        }
        return None;
    }
    if let Op::MulW { x, w, shift } = op {
        let (tx, tw) = (&tens[*x as usize], &ws[*w as usize]);
        if let Some(idx) = (0..n * n).find(|&e| out[e] != mul_apply(tx[e], tw[e], *shift)) {
            let (i, j) = (idx / n, idx % n);
            return Some((
                k as u8,
                ProgFraud::MulW {
                    i: i as u32,
                    j: j as u32,
                    in_col: (0..n).map(|q| tx[q * n + j]).collect(),
                    in_path: cols_tree(tx, n).proof(j),
                    w_row: tw[i * n..(i + 1) * n].to_vec(),
                    w_path: rows_tree(tw, n).proof(i),
                    out_col: (0..n).map(|q| out[q * n + j]).collect(),
                    out_path: out_tree.proof(j),
                },
            ));
        }
        return None;
    }
    if let Op::Lut { x, w, row } = op {
        let tx = &tens[*x as usize];
        let table = &ws[*w as usize][*row as usize * n..(*row as usize + 1) * n];
        for j in 0..n {
            let in_col: Vec<i8> = (0..n).map(|i| tx[i * n + j]).collect();
            let out_col: Vec<i8> = (0..n).map(|i| out[i * n + j]).collect();
            if in_col.iter().map(|v| lut_apply(table, *v)).collect::<Vec<i8>>() != out_col {
                return Some((
                    k as u8,
                    ProgFraud::Lut { j: j as u32, in_col, in_path: cols_tree(tx, n).proof(j), t_row: table.to_vec(), t_path: rows_tree(&ws[*w as usize], n).proof(*row as usize), out_col, out_path: out_tree.proof(j) },
                ));
            }
        }
        return None;
    }
    if op.is_gemm() {
        let (shift, relu, x) = match op {
            Op::Gemm { shift, relu, x, .. } | Op::GemmAT { shift, relu, x, .. } => (*shift, *relu, *x),
            _ => unreachable!(),
        };
        let (z, tx) = (zs[k].as_ref()?, &tens[x as usize]);
        // matrix A and its row tree: weights W_w, or T_a^T (whose row tree = column tree of T_a)
        let (wm, w_tree): (Vec<i8>, MerkleTree) = match op {
            Op::Gemm { w, .. } => (ws[*w as usize].clone(), rows_tree(&ws[*w as usize], n)),
            Op::GemmAT { a, .. } => {
                let ta = &tens[*a as usize];
                ((0..n * n).map(|idx| ta[(idx % n) * n + idx / n]).collect(), cols_tree(ta, n))
            }
            _ => unreachable!(),
        };
        let (shift, relu) = (&shift, &relu);
        let wm = &wm;
        if !freivalds(wm, tx, z, n, 24, &[3u8; 32]) {
            let (i, j) = find_bad_entry(wm, tx, z, n, &[3u8; 32])?;
            return Some((
                k as u8,
                ProgFraud::Gemm {
                    i: i as u32,
                    j: j as u32,
                    w_row: wm[i * n..(i + 1) * n].to_vec(),
                    w_path: w_tree.proof(i),
                    x_col: (0..n).map(|t| tx[t * n + j]).collect(),
                    x_path: cols_tree(tx, n).proof(j),
                    z_row: row_of(z, n, i),
                    z_path: c_tree(z, n).proof(i),
                },
            ));
        }
        if let Some(idx) = (0..n * n).find(|&e| out[e] != act_fn(z[e], *shift, *relu)) {
            let (i, j) = (idx / n, idx % n);
            return Some((
                k as u8,
                ProgFraud::Act { i: i as u32, j: j as u32, z_row: row_of(z, n, i), z_path: c_tree(z, n).proof(i), out_col: (0..n).map(|t| out[t * n + j]).collect(), out_path: out_tree.proof(j) },
            ));
        }
    } else {
        if eval_op(op, tens, ws, n).1 == *out {
            return None; // honest case: a quick evaluation is enough, the cell-by-cell search is only needed if the operation is wrong
        }
        for idx in 0..n * n {
            let (i, j) = (idx / n, idx % n);
            let req = op_inputs(op, i, j, n);
            let vals: Vec<i8> = req.iter().map(|&(t, si, sj)| tens[t][si * n + sj]).collect();
            if out[idx] != op_eval(op, &vals) {
                let ins = req
                    .iter()
                    .map(|&(t, _, sj)| OpenedCol { tensor: t as u8, col: sj as u32, vals: (0..n).map(|q| tens[t][q * n + sj]).collect(), path: cols_tree(&tens[t], n).proof(sj) })
                    .collect();
                return Some((k as u8, ProgFraud::Elem { i: i as u32, j: j as u32, ins, out_col: (0..n).map(|q| out[q * n + j]).collect(), out_path: out_tree.proof(j) }));
            }
        }
    }
    None
}

/// (x0, weights, operations) of a TRANSFORMER ENCODER BLOCK toy, in integer arithmetic: single-head attention
/// (Q, K, V, scores K^T Q, per-column softmax, output V.A, projection) + residual + normalisation + feed-forward
/// network + residual + normalisation. 32 features x 16 tokens (columns 16.. = padding).
pub fn demo_transformer(n: usize, seed: u64) -> (Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let (d, l, ff) = (32usize, 16usize, 64usize);
    let mut x0 = vec![0i8; n * n];
    for r in 0..d {
        for c in 0..l {
            x0[r * n + c] = rng.gen_range(-20..=20) as i8;
        }
    }
    let mk = |rows: usize, cols: usize, rng: &mut StdRng| {
        let mut w = vec![0i8; n * n];
        for r in 0..rows {
            for c in 0..cols {
                w[r * n + c] = rng.gen_range(-3..=3) as i8;
            }
        }
        w
    };
    let weights = vec![mk(d, d, &mut rng), mk(d, d, &mut rng), mk(d, d, &mut rng), mk(d, d, &mut rng), mk(ff, d, &mut rng), mk(d, ff, &mut rng)];
    let ops = vec![
        Op::Gemm { w: 0, x: 0, shift: 3, relu: false },            // T1  Q
        Op::Gemm { w: 1, x: 0, shift: 3, relu: false },            // T2  K
        Op::Gemm { w: 2, x: 0, shift: 3, relu: false },            // T3  V
        Op::GemmAT { a: 2, x: 1, shift: 5, relu: false },          // T4  scores K^T Q (key i, query j)
        Op::Softmax { x: 4, len: l as u16 },                       // T5  attention (softmax over the keys, per query)
        Op::Map { x: 3, map: MapSpec::Transpose },                 // T6  V^T
        Op::GemmAT { a: 6, x: 5, shift: 6, relu: false },          // T7  output V.A
        Op::Gemm { w: 3, x: 7, shift: 3, relu: false },            // T8  projection
        Op::Add { a: 0, b: 8, relu: false },                       // T9  residual
        Op::LayerNorm { x: 9, len: d as u16 },                     // T10
        Op::Gemm { w: 4, x: 10, shift: 4, relu: true },            // T11 feed-forward (expansion)
        Op::Gemm { w: 5, x: 11, shift: 4, relu: false },           // T12 feed-forward (projection)
        Op::Add { a: 10, b: 12, relu: false },                     // T13 residual
        Op::LayerNorm { x: 13, len: d as u16 },                    // T14
    ];
    (x0, weights, ops)
}

/// (x0, weights, operations) of a 16 -> 24 -> 10 MLP WITH BIASES: Gemm, Bias (ReLU), Gemm, Bias. The last weight is the BIAS TABLE
/// (row 0: layer 1, row 1: layer 2). 16 features x 32 samples.
pub fn demo_bias_mlp(n: usize, seed: u64) -> (Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x0 = vec![0i8; n * n];
    for r in 0..16 {
        for c in 0..32.min(n) {
            x0[r * n + c] = rng.gen_range(0..=64) as i8;
        }
    }
    let mut mk = |rows: usize, cols: usize| {
        let mut w = vec![0i8; n * n];
        for r in 0..rows {
            for c in 0..cols {
                w[r * n + c] = rng.gen_range(-6..=6) as i8;
            }
        }
        w
    };
    let (w0, w1) = (mk(24, 16), mk(10, 24));
    let mut bias = vec![0i8; n * n];
    for i in 0..24 {
        bias[i] = rng.gen_range(-20..=20) as i8;
    }
    for i in 0..10 {
        bias[n + i] = rng.gen_range(-20..=20) as i8;
    }
    // row 2 of the weight table: a quantised tanh activation, table[k] = round(64 tanh((k - 64) / 32)), k = 0..=127
    for k in 0..=127usize {
        bias[2 * n + k] = ((64.0 * (((k as f64) - 64.0) / 32.0).tanh()).round()) as i8;
    }
    let ops = vec![
        Op::Gemm { w: 0, x: 0, shift: 6, relu: false },
        Op::Bias { x: 1, w: 2, row: 0, relu: false },
        Op::Lut { x: 2, w: 2, row: 2 },
        Op::Gemm { w: 1, x: 3, shift: 6, relu: false },
        Op::Bias { x: 4, w: 2, row: 1, relu: false },
    ];
    (x0, vec![w0, w1, bias], ops)
}

/// (x0, weights, operations) of a small network with GENERAL CONVOLUTION (stride 2, no padding): 1 channel 8x8, batch of 2 -> 3x3 conv stride 2
/// (3x3 grid, 4 channels) -> Gemm + Bias -> flattening -> dense classifier.
pub fn demo_convnet(n: usize, seed: u64) -> (Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x0 = vec![0i8; n * n];
    for q in 0..8 * 8 * 2 {
        x0[q] = rng.gen_range(0..=64) as i8;
    }
    let mut mk = |rows: usize, cols: usize| {
        let mut w = vec![0i8; n * n];
        for r in 0..rows {
            for c in 0..cols {
                w[r * n + c] = rng.gen_range(-5..=5) as i8;
            }
        }
        w
    };
    let (w0, w1) = (mk(4, 9), mk(10, 36));
    let mut bias = vec![0i8; n * n];
    for i in 0..4 {
        bias[i] = rng.gen_range(-15..=15) as i8;
    }
    let ops = vec![
        Op::Map { x: 0, map: MapSpec::Conv { c_in: 1, h: 8, w: 8, batch: 2, kh: 3, kw: 3, pad: 0, stride: 2 } }, // T1 : 9 x 18
        Op::Gemm { w: 0, x: 1, shift: 4, relu: false },                                                          // T2 : 4 x 18
        Op::Bias { x: 2, w: 2, row: 0, relu: true },                                                             // T3
        Op::Map { x: 3, map: MapSpec::Flatten { c: 4, h: 3, w: 3, batch: 2 } },                                  // T4 : 36 x 2
        Op::Gemm { w: 1, x: 4, shift: 4, relu: false },                                                          // T5 : 10 x 2
    ];
    (x0, vec![w0, w1, bias], ops)
}

/// (x0, weights, operations) of an MLP with a WIDE INPUT (200 features on an n = 128 chain: 2 feature blocks laid side by side
/// in columns, n/2 = 64 samples per job): Shift to extract block 1, one Gemm per block, Add, classifier.
pub fn demo_wide_mlp(n: usize, seed: u64) -> (Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let half = n / 2;
    let mut x0 = vec![0i8; n * n];
    for r in 0..n {
        for c in 0..half {
            x0[r * n + c] = rng.gen_range(0..=40) as i8; // feature block 0 (rows 0..n)
        }
    }
    for r in 0..72.min(n) {
        for c in half..n {
            x0[r * n + c] = rng.gen_range(0..=40) as i8; // block 1 (rows 0..72)
        }
    }
    let mut mk = |rows: usize, cols: usize| {
        let mut w = vec![0i8; n * n];
        for r in 0..rows {
            for c in 0..cols {
                w[r * n + c] = rng.gen_range(-3..=3) as i8;
            }
        }
        w
    };
    let weights = vec![mk(16, n.min(128)), mk(16, 72), mk(10, 16)];
    let ops = vec![
        Op::Map { x: 0, map: MapSpec::Shift { dr: 0, dc: half as u16 } }, // T1: block 1 brought back to columns 0..
        Op::Gemm { w: 0, x: 0, shift: 6, relu: false },                   // T2
        Op::Gemm { w: 1, x: 1, shift: 6, relu: false },                   // T3
        Op::Add { a: 2, b: 3, relu: true },                               // T4
        Op::Gemm { w: 2, x: 4, shift: 5, relu: false },                   // T5
    ];
    (x0, weights, ops)
}

/// Demonstration programs by name: "cnn" (default), "transformer", "biasmlp", "convnet" or "widemlp".
/// (x0, weights, operations) of a MULTI-HEAD ATTENTION block (4 heads of 8, 32 features, 16 tokens) followed by a LayerNorm
/// WITH gamma (Mul) and beta (Bias): Q, K, V, HeadSpread(Q), scores K^T.Q of all heads side by side, Softmax, V^T, V.A,
/// HeadGather, LayerNorm, Mul, Bias. This is the building block of MiniLM (tools/quantize_minilm.py), in small. Requires n >= 64.
pub fn demo_heads(n: usize, seed: u64) -> (Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let (d, l) = (32usize, 16usize);
    let mut x0 = vec![0i8; n * n];
    for r in 0..d {
        for c in 0..l {
            x0[r * n + c] = rng.gen_range(-20..=20) as i8;
        }
    }
    let mut mk = || {
        let mut w = vec![0i8; n * n];
        for r in 0..d {
            for c in 0..d {
                w[r * n + c] = rng.gen_range(-3..=3) as i8;
            }
        }
        w
    };
    let (wq, wk, wv) = (mk(), mk(), mk());
    let mut table = vec![0i8; n * n];
    for i in 0..d {
        table[i] = rng.gen_range(20..=120) as i8; // gamma (row 0)
        table[n + i] = rng.gen_range(-10..=10) as i8; // beta (row 1)
    }
    let (dh, heads, lh) = (8u16, 4u16, l as u16);
    let ops = vec![
        Op::Gemm { w: 0, x: 0, shift: 3, relu: false },                                    // T1  Q
        Op::Gemm { w: 1, x: 0, shift: 3, relu: false },                                    // T2  K
        Op::Gemm { w: 2, x: 0, shift: 3, relu: false },                                    // T3  V
        Op::Map { x: 1, map: MapSpec::HeadSpread { dh, heads, l: lh } },                   // T4  Q spread per head
        Op::GemmAT { a: 2, x: 4, shift: 4, relu: false },                                  // T5  scores (key, (head, query))
        Op::Softmax { x: 5, len: l as u16 },                                               // T6
        Op::Map { x: 3, map: MapSpec::Transpose },                                         // T7  V^T
        Op::GemmAT { a: 7, x: 6, shift: 6, relu: false },                                  // T8  V.A (all the heads)
        Op::Map { x: 8, map: MapSpec::HeadGather { dh, heads, l: lh } },                   // T9  each row takes back its head
        Op::LayerNorm { x: 9, len: d as u16 },                                             // T10
        Op::Mul { x: 10, w: 3, row: 0, shift: 6 },                                         // T11 gamma
        Op::Bias { x: 11, w: 3, row: 1, relu: false },                                     // T12 beta
    ];
    (x0, vec![wq, wk, wv, table], ops)
}

/// FINE attention (images: hundreds of tokens): same Q, K, V as `demo_heads`, then Softmax (1/64) AND SoftmaxLo (fine part);
/// V.A_high >> 6 + V.A_fine >> 12, then HeadGather, LayerNorm, gamma (Mul), beta (Bias).
pub fn demo_fine_attention(n: usize, seed: u64) -> (Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    let (x0, ws, _) = demo_heads(n, seed);
    let (d, l, dh, heads) = (32u16, 16u16, 8u16, 4u16);
    let ops = vec![
        Op::Gemm { w: 0, x: 0, shift: 3, relu: false },                                    // T1  Q
        Op::Gemm { w: 1, x: 0, shift: 3, relu: false },                                    // T2  K
        Op::Gemm { w: 2, x: 0, shift: 3, relu: false },                                    // T3  V
        Op::Map { x: 1, map: MapSpec::HeadSpread { dh, heads, l } },                       // T4
        Op::GemmAT { a: 2, x: 4, shift: 4, relu: false },                                  // T5  scores
        Op::Softmax { x: 5, len: l },                                                      // T6  probabilities in 1/64
        Op::SoftmaxLo { x: 5, len: l },                                                    // T7  fine part (1/4096)
        Op::Map { x: 3, map: MapSpec::Transpose },                                         // T8  V^T
        Op::GemmAT { a: 8, x: 6, shift: 6, relu: false },                                  // T9  V.A_haut
        Op::GemmAT { a: 8, x: 7, shift: 12, relu: false },                                 // T10 V.A_fin
        Op::Add { a: 9, b: 10, relu: false },                                              // T11
        Op::Map { x: 11, map: MapSpec::HeadGather { dh, heads, l } },                      // T12
        Op::LayerNorm { x: 12, len: d },                                                   // T13
        Op::Mul { x: 13, w: 3, row: 0, shift: 6 },                                         // T14 gamma
        Op::Bias { x: 14, w: 3, row: 1, relu: false },                                     // T15 beta
    ];
    (x0, ws, ops)
}

/// CAUSAL attention (language model): like `demo_fine_attention` but each query only sees the previous keys
/// (CausalSoftmax high + fine), over 12 real tokens in blocks of l = 16 columns per head.
pub fn demo_causal_attention(n: usize, seed: u64) -> (Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    let (x0, ws, _) = demo_heads(n, seed);
    let (d, l, dh, heads) = (32u16, 16u16, 8u16, 4u16);
    let ops = vec![
        Op::Gemm { w: 0, x: 0, shift: 3, relu: false },                                    // T1  Q
        Op::Gemm { w: 1, x: 0, shift: 3, relu: false },                                    // T2  K
        Op::Gemm { w: 2, x: 0, shift: 3, relu: false },                                    // T3  V
        Op::Map { x: 1, map: MapSpec::HeadSpread { dh, heads, l } },                       // T4
        Op::GemmAT { a: 2, x: 4, shift: 4, relu: false },                                  // T5  scores (keys x (head, query))
        Op::CausalSoftmax { x: 5, len: 12, l, fine: false },                               // T6  causal probabilities in 1/64
        Op::CausalSoftmax { x: 5, len: 12, l, fine: true },                                // T7  fine part (1/4096)
        Op::Map { x: 3, map: MapSpec::Transpose },                                         // T8  V^T
        Op::GemmAT { a: 8, x: 6, shift: 6, relu: false },                                  // T9  V.A_haut
        Op::GemmAT { a: 8, x: 7, shift: 12, relu: false },                                 // T10 V.A_fin
        Op::Add { a: 9, b: 10, relu: false },                                              // T11
        Op::Map { x: 11, map: MapSpec::HeadGather { dh, heads, l } },                      // T12
        Op::LayerNorm { x: 12, len: d },                                                   // T13
        Op::Mul { x: 13, w: 3, row: 0, shift: 6 },                                         // T14 gamma
        Op::Bias { x: 14, w: 3, row: 1, relu: false },                                     // T15 beta
    ];
    (x0, ws, ops)
}

/// CAUSAL attention with a COARSER exponent step (`CausalSoftmaxS`, scale 2): the program of `demo_causal_attention` with the two softmax operations replaced.
pub fn demo_causal_attention_scaled(n: usize, seed: u64) -> (Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    let (x0, ws, mut ops) = demo_causal_attention(n, seed);
    for op in ops.iter_mut() {
        if let Op::CausalSoftmax { x, len, l, fine } = *op {
            *op = Op::CausalSoftmaxS { x, len, l, fine, scale: 2 };
        }
    }
    (x0, ws, ops)
}

/// (x0, weights, operations) of a GATED feed-forward block with the operations of the Qwen3 models: 32 features x 16 tokens, projection,
/// RMS normalisation in 2 groups of 16 rows (like a per-head QK-norm), gate and up projections (64 rows), silu table on the gate, element-wise
/// product gate x up (SwiGLU, `MulT`), element-wise product with a weight table (the shape of a rotary position embedding table, `MulW`),
/// down projection, residual addition and a final RMS normalisation of the whole column. Requires n >= 64.
pub fn demo_gated(n: usize, seed: u64) -> (Vec<i8>, Vec<Vec<i8>>, Vec<Op>) {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let (d, l, ff) = (32usize, 16usize, 64usize);
    let mut x0 = vec![0i8; n * n];
    for r in 0..d {
        for c in 0..l {
            x0[r * n + c] = rng.gen_range(-20..=20) as i8;
        }
    }
    let mut mk = |rows: usize, cols: usize| {
        let mut w = vec![0i8; n * n];
        for r in 0..rows {
            for c in 0..cols {
                w[r * n + c] = rng.gen_range(-6..=6) as i8;
            }
        }
        w
    };
    let (w0, w1, w2, w5) = (mk(d, d), mk(ff, d), mk(ff, d), mk(d, ff));
    // table of the silu activation on the gate: row 0 of weight 3, table[k] = round(16 silu((k - 64) / 16)), k = 0..=127
    let mut silu = vec![0i8; n * n];
    for k in 0..=127usize {
        let v = ((k as f64) - 64.0) / 16.0;
        silu[k] = ((16.0 * v / (1.0 + (-v).exp())).round()).clamp(-64.0, 64.0) as i8;
    }
    // weight 4: a table of 64 x 16 entries in [-64, 64] (cos / sin of a rotary embedding: scale 64)
    let mut tab = vec![0i8; n * n];
    for r in 0..ff {
        for c in 0..l {
            let a = (r as f64) * 0.37 + (c as f64) * 0.11;
            tab[r * n + c] = (64.0 * a.cos()).round() as i8;
        }
    }
    let ops = vec![
        Op::Gemm { w: 0, x: 0, shift: 6, relu: false },                        // T1
        Op::RmsNorm { x: 1, len: d as u16, group: 16, shift: 1 },              // T2: two groups of 16 rows
        Op::Gemm { w: 1, x: 2, shift: 6, relu: false },                        // T3 gate
        Op::Gemm { w: 2, x: 2, shift: 6, relu: false },                        // T4 up
        Op::Lut { x: 3, w: 3, row: 0 },                                        // T5 silu(gate)
        Op::MulT { a: 5, b: 4, shift: 6 },                                     // T6 silu(gate) * up
        Op::MulW { x: 6, w: 4, shift: 6 },                                     // T7 times the table
        Op::Gemm { w: 5, x: 7, shift: 6, relu: false },                        // T8 down projection
        Op::Add { a: 1, b: 8, relu: false },                                   // T9 residual
        Op::RmsNorm { x: 9, len: d as u16, group: d as u16, shift: 0 },        // T10 final normalisation
    ];
    (x0, vec![w0, w1, w2, silu, tab, w5], ops)
}

pub fn demo_program(name: &str, n: usize, seed: u64) -> Option<(Vec<i8>, Vec<Vec<i8>>, Vec<Op>)> {
    if let Some(path) = name.strip_prefix("model:") {
        // imported model (see model.rs); `seed` = index of the sample (batch of images) used as input
        let m = crate::model::load(std::path::Path::new(path)).ok()?;
        if m.n != n {
            return None;
        }
        return m.program(seed as usize).ok();
    }
    match name {
        "cnn" => Some(demo_cnn(n, seed)),
        "transformer" => Some(demo_transformer(n, seed)),
        "biasmlp" => Some(demo_bias_mlp(n, seed)),
        "convnet" => Some(demo_convnet(n, seed)),
        "widemlp" => Some(demo_wide_mlp(n, seed)),
        "heads" if n >= 64 => Some(demo_heads(n, seed)),
        "fineattn" if n >= 64 => Some(demo_fine_attention(n, seed)),
        "causalattn" if n >= 64 => Some(demo_causal_attention(n, seed)),
        "gated" if n >= 64 => Some(demo_gated(n, seed)),
        "causalscaled" if n >= 64 => Some(demo_causal_attention_scaled(n, seed)),
        _ => None,
    }
}
