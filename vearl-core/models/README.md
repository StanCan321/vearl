# Models

Small demonstration models used by the tools and the tests. Where they come from and the terms that apply to the data they were trained
on: `THIRD-PARTY-NOTICES.md` at the root of the repository (section 1).

## Large generated models are not stored here

Four files are too big to keep in the repository (10 to 35 MB each) and are fully reproducible. Regenerate them with the commands below
(they need `pip install torch numpy` and, for the embedding models, `sentence-transformers datasets`):

| file | command (from `vearl-core/`) | documented in |
|---|---|---|
| `mnist-mlp-4096.json` | `python3 tools/train_mnist.py models/mnist-mlp-4096.json --hidden 4096,4096 --epochs 6 --batch 256 --lr 5e-4` | `docs/11` |
| `embed-agnews.json` (+ `.featurizer.json`) | `python3 tools/train_embed.py models/embed-agnews.json` | `docs/12` |
| `embed-agnews-v4096.json` | `python3 tools/train_embed.py models/embed-agnews-v4096.json` with the options given in `docs/12` | `docs/12` |
| `transformer2-float.npz` | `python3 tools/train_tiny_transformer.py models/transformer2-float.npz` (options in `docs/12` section 6); only its int8 version `transformer2-int8.json` is needed to run it | `docs/12` |

The models that stay in this folder are small, and they are all the tests and quick demonstrations need.
