# Third-party material

The Vearl source code and documentation are under the Apache License 2.0 (`LICENSE`). This file lists the third-party material that the
repository includes, derives from, or downloads when you run its tools, with the terms that apply to it. Please check the upstream terms
yourself before reusing any of it: this list is a good-faith summary, not legal advice, and licenses of upstream projects can change.

## 1. Models stored in `vearl-core/models/`

These small models were trained, distilled or quantized by the project to demonstrate verifiable inference. Their weights are provided for
testing and research. They are derived from the data sets and models below, and **the terms of the original data apply to them in addition
to this repository's license**.

| files | what it is | derived from |
|---|---|---|
| `mnist-mlp*.json` | int8 MLPs for handwritten digits | MNIST (LeCun, Cortes, Burges) |
| `digits-*.json` | small int8 networks for 8 x 8 digits | the UCI "Optical Recognition of Handwritten Digits" set, as shipped with scikit-learn (CC BY 4.0) |
| `resnet20-cifar-float.pt` | ResNet-20 float weights (the int8 version is generated, not stored) | CIFAR-10 (Krizhevsky) |
| `embed-agnews*.json`, `*.featurizer.json` | text-embedding models and their word lists | AG News corpus (see the note below) |
| `tiny-transformer-*`, `transformer2-*` | tiny transformers distilled from `all-MiniLM-L6-v2` on AG News texts | AG News, `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) |

Note on AG News: the corpus is a collection of news articles gathered for research. Its distributors state that it is for non-commercial
research use. Models trained on it should be treated the same way: do not assume they can be used commercially.

## 2. Models and data downloaded by the tools (not stored in the repository)

The scripts in `vearl-core/tools/` download these when you run them. Nothing below is redistributed here; the converted (quantized) files
that the tools produce are derived works of the originals and keep their terms.

| used for | source | license of the source |
|---|---|---|
| MiniLM sentence embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 |
| BGE embeddings | `BAAI/bge-base-en-v1.5` | MIT |
| photo classification | `facebook/deit-tiny-patch16-224` (Meta) | Apache-2.0 |
| text generation | `roneneldan/TinyStories-33M` and the TinyStories data | model MIT; data CDLA-Sharing-1.0 |
| sentence similarity benchmark | STS-B (GLUE) | research data set, see its page |
| perplexity measurement | WikiText-2 | CC BY-SA 3.0 |
| image classes | ImageNet-1k class names | see the ImageNet terms |

## 3. Website assets (`site/`)

- **Fonts** (Instrument Sans, Instrument Serif, IBM Plex Mono): SIL Open Font License 1.1. The files and their notice are fetched by
  `site/tools/fetch_fonts.py`.
- **Photographs** shown by the compute page (`site/assets/photos/`): each one is released under CC0 or the public domain, verified file by
  file by `site/tools/fetch_cc0.py`; the authors and sources are listed on the credits page (`site/credits.html`, `site/tools/web-photos/credits.json`).
  Images from ImageNet, Imagenette or COCO must never be added.
- **Brand files** (`brand/`): the Vearl name, logo and mark are not covered by the Apache License (section 6 of the license: no trademark grant).

## 4. Software dependencies

The Rust crates listed in `vearl-core/Cargo.lock` keep their own licenses (mostly MIT and Apache-2.0). Before a release, check them with
`cargo install cargo-deny` and `cargo deny check licenses`. The Python tools use numpy, torch, transformers, scikit-learn and similar
open-source libraries, installed by the user. The GPU kernels use the NVIDIA CUDA runtime, linked statically in the Linux builds under the
terms of the NVIDIA CUDA Toolkit EULA.

The wallet window (`vearl-wallet-gui/`) additionally uses the egui / eframe GUI libraries (MIT or Apache-2.0), `qrcode` (MIT or Apache-2.0),
`png` (MIT or Apache-2.0) and, on Windows, `winresource` (MIT), with their own dependencies listed in `vearl-wallet-gui/Cargo.lock`. It embeds
the three fonts of the website (Instrument Sans, Instrument Serif, IBM Plex Mono, SIL Open Font License 1.1: `vearl-wallet-gui/assets/fonts/`,
converted from the website files) and the Vearl mark (`brand/`, not covered by the Apache License).

## 5. The optional Quantus add-on

The release builds can be extended with a separate, optional archive that contains the official Quantus miner (Apache-2.0, built
unmodified from <https://github.com/Quantus-Network/quantus-miner>) and the NVIDIA NVRTC libraries it loads (redistributable under the CUDA
Toolkit EULA, attachment A). Their notices are inside that archive (`licenses/`). Vearl is not affiliated with, endorsed by or paid by
Quantus Network.
