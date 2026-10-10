#!/bin/bash
# End-to-end demonstrations (real chain, GPU worker, each result verified before payment). From vearl-core, after:
#   cargo build --release --features gpu --bins
set -e
echo "== 1. inference MNIST (MLP 784-512-256-10, 1024 images par job)";       python3 tools/mnist_demo.py models/mnist-mlp.json
echo "== 2. CNN with max-pooling and BatchNorm (8x8 digits, n = 128 chain)";    python3 tools/cnn_demo.py
echo "== 3. hand-written text embeddings (client)";                            python3 tools/embed_client_demo.py models/embed-agnews.json
echo "== 4. 4096^3 matrix product (137 GOP) verified end to end";    python3 tools/la_demo.py 4 4 4
echo "== 5. worker that withholds the data: Fetch then SlashUnserved";         python3 tools/da_demo.py
echo "== 6. audit file: independent validators and differential fuzzes"; python3 tools/audit_all.py --quick
