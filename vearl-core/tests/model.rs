//! Model import: Flatten, file format, validation, execution.
use vearl_core::model::*;
use vearl_core::prog::*;

const N: usize = 128;

#[test]
fn flatten_maps_channels_and_positions_to_feature_rows() {
    // 3 channels, 2x2 images, batch 2: row i = c*4 + y*2 + x; column = sample
    let op = Op::Map { x: 0, map: MapSpec::Flatten { c: 3, h: 2, w: 2, batch: 2 } };
    // input: T[c][(y*2+x)*2 + b] = unique value per (c,y,x,b)
    let mut t = vec![0i8; N * N];
    let val = |c: usize, y: usize, x: usize, b: usize| (c * 16 + y * 8 + x * 4 + b + 1) as i8;
    for c in 0..3 {
        for y in 0..2 {
            for x in 0..2 {
                for b in 0..2 {
                    t[c * N + (y * 2 + x) * 2 + b] = val(c, y, x, b);
                }
            }
        }
    }
    let (_, out) = eval_op(&op, &[t], &[], N);
    for c in 0..3 {
        for y in 0..2 {
            for x in 0..2 {
                for b in 0..2 {
                    assert_eq!(out[(c * 4 + y * 2 + x) * N + b], val(c, y, x, b), "({c},{y},{x},{b})");
                }
            }
        }
    }
    assert_eq!(out[12 * N], 0, "beyond c*h*w rows: zero");
    assert_eq!(out[2], 0, "beyond the batch: zero");
    // validation: dimensions outside the tensor
    assert!(validate_ops(&[Op::Gemm { w: 0, x: 0, shift: 3, relu: false }, Op::Map { x: 1, map: MapSpec::Flatten { c: 40, h: 4, w: 4, batch: 2 } }], 1, N).is_err(), "40*16 > n");
    assert!(validate_ops(&[Op::Gemm { w: 0, x: 0, shift: 3, relu: false }, op.clone()], 1, N).is_ok(), "Flatten valid on an existing tensor");
}

fn tiny_model() -> ModelFile {
    let w: Vec<i8> = (0..N * N).map(|i| ((i % 7) as i8) - 3).collect();
    let x0: Vec<i8> = (0..N * N).map(|i| ((i % 5) as i8) * 4).collect();
    ModelFile {
        name: "tiny".into(),
        n: N,
        ops: vec![Op::Gemm { w: 0, x: 0, shift: 6, relu: true }],
        weights: vec![b64_encode(&w)],
        classes: 4,
        batch: 2,
        input_scale: 0.25,
        samples: vec![Sample { x0: b64_encode(&x0), labels: vec![1, 2], ops: None, weights: Default::default() }],
    }
}

#[test]
fn model_file_roundtrip_execution_and_validation() {
    let m = tiny_model();
    let json = serde_json::to_string(&m).unwrap();
    let dir = std::env::temp_dir().join(format!("vearl-model-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.json");
    std::fs::write(&path, &json).unwrap();
    let back = load(&path).unwrap();
    assert_eq!(back.name, "tiny");
    // execution: identical to the direct evaluation of the program
    let (x0, ws, ops) = back.program(0).unwrap();
    let (tens, _) = run_program(&x0, &ws, &ops, N);
    let logits = back.logits(0).unwrap();
    assert_eq!(logits.len(), 2);
    for b in 0..2 {
        for r in 0..4 {
            assert_eq!(logits[b][r], tens[1][r * N + b]);
        }
    }
    // the program loaded by the name "model:" is the same
    let (x0b, _, _) = demo_program(&format!("model:{}", path.display()), N, 0).unwrap();
    assert_eq!(x0b, x0);
    assert!(demo_program(&format!("model:{}", path.display()), N, 5).is_none(), "sample missing");

    // invalid files
    let mut bad = tiny_model();
    bad.weights[0] = b64_encode(&vec![100i8; N * N]);
    assert!(bad.validate().is_err(), "weights outside [-64,64]");
    let mut bad = tiny_model();
    bad.weights[0] = "!!not base64!!".into();
    assert!(bad.validate().is_err());
    let mut bad = tiny_model();
    bad.weights[0] = b64_encode(&vec![1i8; N * N - 1]);
    assert!(bad.validate().is_err(), "incorrect length");
    let mut bad = tiny_model();
    bad.ops = vec![Op::Gemm { w: 3, x: 0, shift: 6, relu: true }];
    assert!(bad.validate().is_err(), "nonexistent weight");
    let mut bad = tiny_model();
    bad.n = 256;
    assert!(bad.validate().is_err(), "unsupported n");
    let mut bad = tiny_model();
    bad.samples[0].x0 = b64_encode(&vec![64i8; N * N - 5]);
    assert!(bad.validate().is_err());
    assert!(load(&dir.join("absent.json")).is_err());
    std::fs::write(dir.join("junk.json"), "{ not json").unwrap();
    assert!(load(&dir.join("junk.json")).is_err());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn samples_may_carry_their_own_program_and_weight_blocks() {
    // a sample replaces the program (shift) and the weight block 0; the others keep those of the model
    let mut m = tiny_model();
    let w2: Vec<i8> = (0..N * N).map(|i| ((i % 3) as i8) - 1).collect();
    let mut s = m.samples[0].clone();
    s.ops = Some(vec![Op::Gemm { w: 0, x: 0, shift: 4, relu: false }]);
    s.weights.insert(0, b64_encode(&w2));
    m.samples.push(s);
    let json = serde_json::to_string(&m).unwrap();
    assert!(!json.contains("\"ops\":null"), "optional fields absent from the JSON of ordinary samples");
    let back: ModelFile = serde_json::from_str(&json).unwrap();
    back.validate().unwrap();
    let (x0, ws, ops) = back.program(1).unwrap();
    assert_eq!(ops, vec![Op::Gemm { w: 0, x: 0, shift: 4, relu: false }]);
    assert_eq!(ws[0], w2);
    let (tens, _) = run_program(&x0, &ws, &ops, N);
    let want = run_program(&x0, &[w2.clone()], &[Op::Gemm { w: 0, x: 0, shift: 4, relu: false }], N).0;
    assert_eq!(tens.last(), want.last());
    // sample 0 is unchanged; sample_program from the common blocks = program
    let base = back.base_weights().unwrap();
    assert_eq!(back.sample_program(0, &base).unwrap(), back.program(0).unwrap());
    assert_eq!(back.sample_program(1, &base).unwrap(), back.program(1).unwrap());
    assert_eq!(back.program(0).unwrap().2, m.ops);
    // validation of the replacements
    let mut bad = back.clone();
    bad.samples[1].weights.insert(5, b64_encode(&w2));
    assert!(bad.validate().is_err(), "replaced block nonexistent");
    let mut bad = back.clone();
    bad.samples[1].weights.insert(0, b64_encode(&vec![90i8; N * N]));
    assert!(bad.validate().is_err(), "replaced block outside [-64,64]");
    let mut bad = back.clone();
    bad.samples[1].ops = Some(vec![Op::Gemm { w: 2, x: 0, shift: 4, relu: false }]);
    assert!(bad.validate().is_err(), "invalid sample program");
}

#[test]
fn op_transfer_box_roundtrips_and_rejects_bad_dimensions() {
    use vearl_core::net::{box_pack, box_unpack, nz_box};
    // non-zero corner 5 x 9 (plus an isolated cell at (7, 2)): the box is 8 x 9, and the reconstruction is exact
    let mut m = vec![0i32; N * N];
    for i in 0..5 {
        for j in 0..9 {
            m[i * N + j] = (i * 31 + j) as i32 - 70;
        }
    }
    m[7 * N + 2] = -3;
    assert_eq!(nz_box(&m, N), (8, 9));
    let (d, v) = box_pack(&m, N);
    assert_eq!((d, v.len()), ((8, 9), 72));
    assert_eq!(box_unpack(d, &v, N).unwrap(), m);
    // all zero -> empty box; full matrix -> n x n
    let z = vec![0i8; N * N];
    assert_eq!(box_pack(&z, N).0, (0, 0));
    assert_eq!(box_unpack((0, 0), &[] as &[i8], N).unwrap(), z);
    let full: Vec<i8> = (0..N * N).map(|i| (i % 5) as i8 + 1).collect();
    let (d, v) = box_pack(&full, N);
    assert_eq!(d, (N as u32, N as u32));
    assert_eq!(box_unpack(d, &v, N).unwrap(), full);
    // dimensions from a malicious peer: refused without panicking
    assert!(box_unpack((N as u32 + 1, 1), &vec![0i8; N + 1], N).is_none());
    assert!(box_unpack((u32::MAX, u32::MAX), &[] as &[i8], N).is_none());
    assert!(box_unpack((2, 2), &[1i8, 2, 3], N).is_none());
}

#[test]
fn dump_returns_every_tensor_for_conformance_checks() {
    let m = tiny_model();
    let input = DumpInput { n: N, ops: m.ops.clone(), weights: m.weights.clone(), x0: m.samples[0].x0.clone() };
    let t = dump(&input).unwrap();
    assert_eq!(t.len(), 2, "T_0 and the output of the single operation");
    assert_eq!(b64_decode(&t[0], N * N).unwrap(), b64_decode(&m.samples[0].x0, N * N).unwrap());
    let bad = DumpInput { n: N, ops: vec![], weights: vec![], x0: m.samples[0].x0.clone() };
    assert!(dump(&bad).is_err(), "empty program refused");
}

#[test]
fn shipped_digits_cnn_classifies_real_images() {
    let m = load(std::path::Path::new("models/digits-cnn.json")).unwrap();
    assert_eq!((m.n, m.classes, m.batch, m.ops.len(), m.weights.len()), (128, 10, 2, 8, 3));
    let (mut ok, mut total) = (0, 0);
    for (i, s) in m.samples.iter().enumerate() {
        for (b, col) in m.logits(i).unwrap().iter().enumerate() {
            let pred = col.iter().enumerate().max_by_key(|(j, v)| (**v, std::cmp::Reverse(*j))).unwrap().0;
            ok += (pred == s.labels[b] as usize) as usize;
            total += 1;
        }
    }
    assert_eq!(total, 24);
    assert!(ok >= 19, "the int8 CNN must classify the great majority of the digits correctly ({ok}/{total})");
}

#[test]
fn shipped_mnist_mlp_classifies_1024_real_images_on_a_1024_wide_program() {
    // int8 MLP 784-512-256-10 (docs/11): a program of 3 GEMMs 1024x1024, one image per column
    let m = load(std::path::Path::new("models/mnist-mlp.json")).unwrap();
    assert_eq!((m.n, m.classes, m.batch, m.ops.len(), m.weights.len()), (1024, 10, 1024, 3, 3));
    let logits = m.logits(0).unwrap();
    let labels = &m.samples[0].labels;
    let ok = logits.iter().zip(labels).filter(|(col, l)| col.iter().enumerate().max_by_key(|(j, v)| (**v, std::cmp::Reverse(*j))).unwrap().0 == **l as usize).count();
    assert_eq!(logits.len(), 1024);
    assert!(ok >= 1000, "the int8 MLP must classify >= 97.6 % of the 1024 images ({ok}/1024)");
}

#[test]
fn shipped_digits_mlp_with_biases_and_folded_batchnorm_classifies_real_images() {
    // PyTorch MLP 64-48-10 with biases AND BatchNorm1d (folded): Gemm, Bias, Gemm, Bias; converted by tools/quantize.py (selftest-torch)
    let m = load(std::path::Path::new("models/digits-mlp-bias.json")).unwrap();
    assert_eq!((m.n, m.classes, m.batch, m.ops.len(), m.weights.len()), (128, 10, 128, 4, 3));
    assert_eq!(m.ops.iter().filter(|o| matches!(o, vearl_core::prog::Op::Bias { .. })).count(), 2);
    let logits = m.logits(0).unwrap();
    let labels = &m.samples[0].labels;
    let ok = logits.iter().zip(labels).filter(|(col, l)| col.iter().enumerate().max_by_key(|(j, v)| (**v, std::cmp::Reverse(*j))).unwrap().0 == **l as usize).count();
    assert!(ok >= 122, "the int8 MLP with biases must classify >= 95 % of the 128 digits ({ok}/128)");
}

#[test]
fn shipped_digits_mlp_with_gelu_table_classifies_real_images() {
    // PyTorch MLP 64-48-10 with biases, folded BatchNorm1d and GELU activation: Gemm, Bias, Lut (table of 128 entries), Gemm, Bias
    let m = load(std::path::Path::new("models/digits-mlp-gelu.json")).unwrap();
    assert_eq!((m.n, m.classes, m.batch, m.ops.len(), m.weights.len()), (128, 10, 128, 5, 3));
    assert_eq!(m.ops.iter().filter(|o| matches!(o, vearl_core::prog::Op::Lut { .. })).count(), 1);
    let logits = m.logits(0).unwrap();
    let labels = &m.samples[0].labels;
    let ok = logits.iter().zip(labels).filter(|(col, l)| col.iter().enumerate().max_by_key(|(j, v)| (**v, std::cmp::Reverse(*j))).unwrap().0 == **l as usize).count();
    assert!(ok >= 122, "the int8 GELU MLP must classify >= 95 % of the 128 digits ({ok}/128)");
}

#[test]
fn shipped_digits_cnn_with_maxpool_and_folded_batchnorm_classifies_real_images() {
    // PyTorch CNN (conv + folded BatchNorm2d + ReLU + max-pooling, conv, max-pooling, dense): Im2col, Gemm, Bias, Pool Max, ..., Flatten, Gemm, Bias
    let m = load(std::path::Path::new("models/digits-cnn-maxpool.json")).unwrap();
    assert_eq!((m.n, m.classes, m.batch), (128, 10, 2));
    let max_pools = m.ops.iter().filter(|o| matches!(o, vearl_core::prog::Op::Pool { pool: vearl_core::prog::PoolSpec::Max { .. }, .. })).count();
    assert_eq!(max_pools, 2);
    let (mut ok, mut total) = (0, 0);
    for (i, s) in m.samples.iter().enumerate() {
        for (b, col) in m.logits(i).unwrap().iter().enumerate() {
            let pred = col.iter().enumerate().max_by_key(|(j, v)| (**v, std::cmp::Reverse(*j))).unwrap().0;
            ok += (pred == s.labels[b] as usize) as usize;
            total += 1;
        }
    }
    assert_eq!(total, 128);
    assert!(ok >= 122, "the int8 CNN with max-pooling must classify >= 95 % of the digits ({ok}/{total})");
}
