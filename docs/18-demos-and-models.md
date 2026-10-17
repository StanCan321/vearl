# 18 - Demos, models and tools

Everything here is run from `vearl-core/` and is described in more detail in the document it points to. The demos start real nodes on your
machine (CPU unless a command says GPU) and check every result against the independent Python implementations.

## Quick demos

**Usefulness demo** (handwritten digits classified by an int8 CNN run by a GPU worker, every operation verified before payment): `bash tools/demo_inference.sh`. **Endurance test** (4 real nodes, hard stops, a cheating worker, jobs of every type): `python3 tools/soak.py 300`. **Partition then merge** of two chains of real nodes: `python3 tools/fork_test.py`.

**Python SDK** to delegate a computation from a script (`tools/vearl_client.py`: transfers, a matrix product entrusted to a worker, Freivalds verification, payment or a fraud proof built in Python): `python3 tools/client_demo.py` (an honest worker is paid, a cheating worker is sanctioned, on real nodes).

**Level 2** (Monte Carlo, tiled 3D rendering, verified by sampling): `vearl-cli l2-submit NODE KEY 5 render 128 128 2 7` then `vearl-cli l2-finish NODE WORKER KEY JOBID 12 image.ppm`; `vearl-l2 render 256 256 2 out.ppm` locally. **Render2** is a cinematic 3D renderer (soft shadows, reflections, ambient occlusion, glow, fog, free camera) that runs on the GPU and gives bit-identical tiles on the CPU: `vearl-l2 render2 1920 1088 4 2 28 1.4 out.ppm gpu` renders a full-HD frame (`docs/10` §4 bis). **Render3** is a Monte Carlo path tracer (global illumination, glass, metals, depth of field, procedural textures, constructive solid geometry), also bit-identical on CPU and GPU: `vearl-l2 render3 watch 1920 1088 16 16 out.ppm gpu` (`docs/10` §4 ter); a noise-free image needs hundreds of samples per pixel, so it takes minutes, not seconds.

**Real inference**: `python3 tools/mnist_demo.py` classifies 1024 MNIST images (int8 MLP, 98.2 %) on an n = 1024 chain with a GPU worker, verifies every operation, pays, then reuses the published embeddings (`docs/11`). A **784→4096→4096→10** MLP (39 operations, layers split into blocks) also runs and is verified (97.85 % on 1024 images, 11 s on the worker side). **Bulk embeddings**: `python3 tools/embed_demo.py /tmp/embed-agnews.json 20` (20,480 texts, output verified bit by bit, then similarity search). **Linear algebra**: `python3 tools/la_demo.py 8 8 8` (an 8192³ product = 1.1 TOP, verified end to end in ~11 s); `docs/12`.

**Bring your own model**:
- `python3 tools/quantize.py mlp weights.npz calib.npy --out model.json` converts a float MLP into a verifiable program.
- `python3 tools/quantize.py torch state.pt calib.npy --out model.json` imports a PyTorch MLP **with biases and BatchNorm1d** (folded) from a `state_dict`.
- `python3 tools/quantize_cnn.py torch …` imports a **PyTorch CNN** (convolutions, BatchNorm, max-pooling; `tools/cnn_demo.py`).
- `python3 tools/import_onnx.py model.onnx calib.npy` imports a **sequential ONNX** model (`docs/12`).

**Verified models** (each runs on the chain and is checked bit for bit against the independent engine; details and limits in `docs/12`):
- **A small trained transformer** (attention, LayerNorm, integer softmax, residuals, FFN): `python3 tools/transformer_demo.py models/tiny-transformer-int8.json` (256/256 embeddings identical to the independent engine).
- **ResNet-20 CIFAR-10** (residual blocks, quantization-aware training) on GPU: `python3 tools/resnet_demo.py` (`docs/12` §4 quater).
- **Full MiniLM-L6** (all-MiniLM-L6-v2: 6 layers, 12 heads, 195 operations including 61 GEMMs, quantization-aware fine-tuning, cosine 0.92 with the float model): embeddings verified and paid on GPU at ≈ 1 s per text, byte-identical to the independent engine (`python3 tools/minilm_demo.py`, `docs/12` §4 quinquies).
- **High-resolution vision**: DeiT-Tiny (224 × 224 photos, 1,000 classes) as two chained jobs, with a new fine attention operation (`SoftmaxLo`): 1.76 s per verified photo, outputs identical to the independent engine, about 8 points below the float model on unseen photos (`python3 tools/vit_demo.py`, `docs/12` §4 sexies).
- **BGE-base embeddings** (110 M parameters, 5 × MiniLM) as three chained jobs: 96.7 % of the original model's quality on STS-B, 3.3 s per verified text (`python3 tools/bert_demo.py`, `docs/12` §4 septies).
- **A language model that writes, word by word, on the chain**: TinyStories-33M, with a new causal attention operation (`CausalSoftmax`); two verified jobs per word, 8 stories written in parallel at 18 s per word, same next word as the original model in 92.9 % of cases (perplexity 2.42 against 2.37) (`python3 tools/gpt_demo.py`, `docs/12` §4 octies).
- **Qwen3-Embedding-0.6B (0.44 B weights, the first model above half a billion parameters)**: ten chained jobs per text, each verified operation by operation. **Ask for a job** with the model file (a release asset, `qwen3-embedding-0.6b-int8.json.xz`, checked against `SHA256SUMS.txt`) and a worker that runs a release with the four Qwen3 operations (rc17 or later; the worker's registry holds the 528 weight blocks, cap 1,024): `python3 tools/qwen3_client.py MODEL.json.xz NODE WORKER KEY "the market fell" "the team won"` prints the 1,024-dimension embeddings (the text goes through `vearl-cli infer`, the cost is escrowed per GEMM and paid once every operation is verified); `python3 tools/qwen3_demo.py MODEL.json.xz` starts a local chain and a worker, embeds 8 sentences (80 jobs, bit for bit equal to the independent engine) and shows their nearest neighbours. Needs `pip install torch transformers numpy`. Quality and limits (cosine 0.77 with the float model, topic retrieval 85 % against 90 %, a different embedding space from the standard Qwen3 because of the prepended token): `docs/12` §4 nonies.

Running a testnet (genesis, wallet, node, faucet, explorer): `docs/06`. **Explorer website** (`site/explorer/`, same design as the site, live blocks, jobs, miners, accounts, search): `docs/06` §4 quater. GPU miner: `node --gpu --miner-wallet FILE …`. A model imported as a network job: `vearl-cli prog-submit … model:models/digits-cnn.json`.

## What a second implementation verifies

The independent Python implementations live in `vearl-core/tools/` and were written separately from the Rust code (same author, see `docs/09`, A18).

The work layer; the Native chain with transfers; the compute market (12 transaction types); **tensor programs** (CNN, general convolution, transformer, MLP with biases and tables, 6 fraud types, data availability; a program defined, computed, mined and proven wrong by the Python code is accepted by the Rust code); **tiled products** (two fraud types, data availability, 15,300 identical decisions in differential fuzzing); and a **Python light client** that connects over Noise XX to a real node and re-validates the whole chain. Blocks mined by the Python code are accepted by the Rust code. The vectors are regenerated and cross-checked by `python3 tools/audit_all.py` (17 independent checks).
