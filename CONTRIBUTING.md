# Contributing

Thank you for your interest. Vearl is a testnet project: the most useful contributions are bug reports with a way to reproduce them,
tests, review of the specification (`docs/07`, `docs/08`), and independent re-implementations that check the frozen vectors.

## Build and test

```bash
cd vearl-core
cargo build --release              # CPU build; add --features gpu on a machine with CUDA (docs/15)
cargo test --release               # about 200 tests: protocol, market, network, fuzzing, frozen vectors
```

On a CUDA machine also run `cargo test --release --features gpu --test gpu` (the GPU must agree bit for bit with the CPU).

## Rules of the project

- **Everything is in English**: code, comments, messages printed by the programs, documentation, commit messages.
- **Consensus changes are special.** Anything that changes a hash, a serialization or a validity rule changes the network. Do not do it
  in a patch without a discussion first; the frozen vectors in `vearl-core/tests/vectors/` and the independent Python implementations in
  `vearl-core/tools/` (`spec_ref.py`, `chain_ref.py`, `market_ref.py`, `tiled_ref.py`, `prog_ref.py`, `l2_ref.py`) must keep agreeing.
- **Tests come with the change.** A bug fix needs a test that fails before and passes after.
- **No secrets, no personal data.** Never commit wallets, keys, passwords, tokens or the 24-word phrases, even for a test network.
- Keep the Rust code formatted and warning-free where reasonable (`cargo clippy`).
- Claims in documents are labelled `[PROVEN]`, `[HYPOTHESIS]`, `[RESEARCH]` or `[AUDIT ACTION]` (see the README): keep that discipline.

## Licensing of contributions

By submitting a contribution you agree that it is licensed under the Apache License 2.0 (see `LICENSE`, section 5), and that you have the
right to submit it. Third-party material must be listed in `THIRD-PARTY-NOTICES.md` with its license.
