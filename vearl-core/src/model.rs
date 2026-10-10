//! Import of quantised models: a JSON file describes an integer tensor program (operations + int8 weights) and,
//! optionally, batches of test samples. The PROGRAM is what the network executes and verifies; the quantisation
//! (float -> int8, choice of shifts) is done OFF-chain by the conversion tool (`tools/train_digits.py`), which also produces
//! the accuracy report. This module only loads, validates and executes.
use crate::prog::*;
use crate::work::VAL_MAX;
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    /// Input tensor T_0 (n x n int8, base64).
    pub x0: String,
    /// Labels of the samples of the batch (columns 0..batch), for evaluation.
    #[serde(default)]
    pub labels: Vec<u8>,
    /// Program specific to this sample (optional; otherwise the model's): e.g. the length of a text, which fixes the attention softmax and
    /// the mean over the tokens. It is validated against the same weight blocks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ops: Option<Vec<Op>>,
    /// Weight blocks replaced for this sample (index -> base64 block), e.g. the mean vector of a text. The other blocks
    /// are shared: the worker's registry only receives the blocks it does not know.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub weights: BTreeMap<usize, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFile {
    pub name: String,
    pub n: usize,
    pub ops: Vec<Op>,
    /// int8 weights (n x n each, base64), referenced by `Op::Gemm { w }`.
    pub weights: Vec<String>,
    /// Number of classes / useful rows of the output.
    #[serde(default)]
    pub classes: usize,
    /// Batch size per execution (useful columns of the output).
    #[serde(default)]
    pub batch: usize,
    /// Real scale of one int8 unit of the input (informational: used by the conversion tool).
    #[serde(default)]
    pub input_scale: f64,
    #[serde(default)]
    pub samples: Vec<Sample>,
}

pub fn b64_decode(s: &str, expect_len: usize) -> Result<Vec<i8>, String> {
    let bytes = B64.decode(s.trim()).map_err(|e| format!("invalid base64: {e}"))?;
    if bytes.len() != expect_len {
        return Err(format!("length {} != {expect_len} bytes expected", bytes.len()));
    }
    let v: Vec<i8> = bytes.iter().map(|b| *b as i8).collect();
    if v.iter().any(|x| (*x as i32).abs() > VAL_MAX) {
        return Err(format!("value outside of [-{VAL_MAX}, {VAL_MAX}]"));
    }
    Ok(v)
}

pub fn b64_encode(v: &[i8]) -> String {
    B64.encode(v.iter().map(|x| *x as u8).collect::<Vec<u8>>())
}

impl ModelFile {
    pub fn validate(&self) -> Result<(), String> {
        if self.n == 0 || self.n % 128 != 0 || self.n > 1024 {
            return Err("n must be a multiple of 128 (<= 1024) and equal to the n of the target chain".into());
        }
        validate_ops(&self.ops, self.weights.len(), self.n)?;
        for (k, w) in self.weights.iter().enumerate() {
            b64_decode(w, self.n * self.n).map_err(|e| format!("weights {k}: {e}"))?;
        }
        for (k, s) in self.samples.iter().enumerate() {
            b64_decode(&s.x0, self.n * self.n).map_err(|e| format!("sample {k}: {e}"))?;
            if let Some(ops) = &s.ops {
                validate_ops(ops, self.weights.len(), self.n).map_err(|e| format!("sample {k}: program: {e}"))?;
            }
            for (i, w) in &s.weights {
                if *i >= self.weights.len() {
                    return Err(format!("sample {k}: weight block {i} does not exist (the model has {})", self.weights.len()));
                }
                b64_decode(w, self.n * self.n).map_err(|e| format!("sample {k}: weights {i}: {e}"))?;
            }
        }
        Ok(())
    }

    /// Common weight blocks of the model (decoded once).
    pub fn base_weights(&self) -> Result<Vec<Vec<i8>>, String> {
        self.weights.iter().map(|w| b64_decode(w, self.n * self.n)).collect()
    }

    /// (T_0, weights, operations) of sample `idx` from the common blocks already decoded (model already validated by `load`).
    pub fn sample_program(&self, idx: usize, base: &[Vec<i8>]) -> Result<(Vec<i8>, Vec<Vec<i8>>, Vec<Op>), String> {
        let s = self.samples.get(idx).ok_or_else(|| format!("sample {idx} missing (the model contains {})", self.samples.len()))?;
        let x0 = b64_decode(&s.x0, self.n * self.n)?;
        let mut ws = base.to_vec();
        for (i, w) in &s.weights {
            *ws.get_mut(*i).ok_or("weight block does not exist")? = b64_decode(w, self.n * self.n)?;
        }
        Ok((x0, ws, s.ops.clone().unwrap_or_else(|| self.ops.clone())))
    }

    /// (T_0, weights, operations) for sample `idx`.
    pub fn program(&self, idx: usize) -> Result<(Vec<i8>, Vec<Vec<i8>>, Vec<Op>), String> {
        self.validate()?;
        self.sample_program(idx, &self.base_weights()?)
    }

    /// Executes the program on sample `idx`; returns the scores (classes x batch) of the last output.
    pub fn logits(&self, idx: usize) -> Result<Vec<Vec<i8>>, String> {
        let (x0, ws, ops) = self.program(idx)?;
        let (tens, _) = run_program(&x0, &ws, &ops, self.n);
        let out = tens.last().unwrap();
        let (rows, cols) = (self.classes.max(1), self.batch.max(1));
        Ok((0..cols).map(|c| (0..rows).map(|r| out[r * self.n + c]).collect()).collect())
    }
}

pub fn load(path: &Path) -> Result<ModelFile, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // LOCAL client file (never received from the network): common-sense bound, wide enough for ~50 weight blocks and a series of
    // a few hundred samples (1.3 MB each in base64 at n = 1024)
    if txt.len() > 1 << 30 {
        return Err("model file too large".into());
    }
    let m: ModelFile = serde_json::from_str(&txt).map_err(|e| format!("unreadable model: {e}"))?;
    m.validate()?;
    Ok(m)
}

/// Input of the `dump` mode (conformance): a complete program with its input.
#[derive(Deserialize)]
pub struct DumpInput {
    pub n: usize,
    pub ops: Vec<Op>,
    pub weights: Vec<String>,
    pub x0: String,
}

/// Executes a program and returns ALL the tensors T_0..T_L (base64): used to compare bit for bit with an external engine.
pub fn dump(input: &DumpInput) -> Result<Vec<String>, String> {
    validate_ops(&input.ops, input.weights.len(), input.n)?;
    let x0 = b64_decode(&input.x0, input.n * input.n)?;
    let ws = input.weights.iter().map(|w| b64_decode(w, input.n * input.n)).collect::<Result<Vec<_>, _>>()?;
    let (tens, _) = run_program(&x0, &ws, &input.ops, input.n);
    Ok(tens.iter().map(|t| b64_encode(t)).collect())
}
