# 20 - Vearl Wallet (the wallet window)

`vearl-wallet-gui` is the Vearl wallet with a window, for Windows and Linux. It is a separate crate (`vearl-wallet-gui/`) that uses the
wallet code of `vearl-core` (`wallet.rs`, `net.rs`, `regmail.rs`), so it does exactly what the command-line wallet `vearl-wallet` does:
a wallet created in one opens in the other, and the signatures and transactions are the same bytes. It is testnet software and has not been
audited.

## What it does

| page | what you can do |
|---|---|
| Overview | balance (standard and post-quantum account), address with a copy button, the latest transactions |
| Send | pay an address; the amount, the fee, a *Max* button, an optional post-quantum signature; a summary and your password before anything is sent |
| Receive | your address as text and QR code; the post-quantum address once unlocked |
| Activity | your transactions with their confirmations, from the block explorer |
| Rewards | status of your registration (read from the public list), and the registration itself: sign, check the three lines, send (docs/19) |
| Tools | sign a message with your key, verify a signature |
| Settings | node and explorer, the 24 recovery words, password change, where the wallet file is |

At the start it also creates a wallet (name, password, the 24 words shown once, then a check of three of them before anything is written) or
restores one from its 24 words.

## Where the files are

- Wallets created by the window: `%APPDATA%\Vearl\wallets\NAME.json` on Windows, `~/.local/share/vearl/wallets/NAME.json` on Linux.
- The `wallet.json` made by `vearl-miner` or `vearl-wallet new`, when it sits in the folder of the program, is listed automatically. Any other
  wallet file can be opened by its path (*Open a wallet file*).
- Settings (node, explorer, wallets opened by path): `settings.json` in the same folder as the wallets folder (`%APPDATA%\Vearl`).
- The file format is the one of `vearl-core/src/wallet.rs`: Argon2id (64 MiB, 3 passes) and ChaCha20-Poly1305 around the entropy of the
  24 words. Nothing is ever stored in clear.

## How the keys are handled

- Opening a wallet needs no password: the address is public and stored in the file, so the balance can be shown without decrypting anything.
- The password is asked **for each action that needs a key**: sending, signing the registration, signing a message, showing the 24 words,
  showing the post-quantum address, changing the password. The keys are derived in a background thread, used, and dropped; the window never
  keeps them between two actions. The password is cleared after use.
- A payment shows its summary (who, how much, fee, signature type) in the same dialog that asks the password. The node must accept the
  transaction before the window says *sent*.
- The 24 words are shown on screen only when asked (and never copied to the clipboard); they are erased from memory when the dialog closes.
- Limits (same as `wallet.rs`): a compromised computer (keylogger, malware reading memory while an action runs) defeats any software wallet,
  and the account derivation is not SLIP-10, so the 24 words only restore Vearl wallets.

## What it talks to

- **A node** (`HOST:PORT`, encrypted Noise channel, the same as `vearl-wallet balance` and `send`): balance, nonce, sending the transaction,
  the block height shown at the bottom left. The nodes are the seed nodes of `network.json` (next to the program) unless a node is set in the
  settings. The first one that answers is used.
- **The block explorer** (`https://vearl.org` by default, HTTPS with the Mozilla trust anchors): the activity list. If it is not reachable the
  rest of the window works.
- **The registration e-mail** (`mail.vearl.org:465`, TLS) only when you press *Send the registration*, and
  **`https://vearl.org/rewards/registered.json`** to show whether your address is registered.

Nothing else is contacted. There is no telemetry and no account.

## Build from source

```
cd vearl-wallet-gui
cargo build --release --locked
cargo test --release --locked
```

Linux needs no development package to build; at run time it needs a graphical session (X11 or Wayland) with OpenGL (Mesa is enough) and
`libxkbcommon`. On a server without a desktop use the command-line `vearl-wallet`.

`cargo test` also contains an end-to-end test of the window logic against a real local node (it creates a wallet with the same commands as the
buttons, pays with both signature types, registers, shows the words, changes the password). It runs when `VEARL_NODE_BIN` points to the `node`
program of a release (`genesis.json` next to it) and is skipped otherwise.

For design work, `cargo build --release --features devshots` builds a variant that fills every screen with demo data and writes a picture of
each one (`VEARL_GUI_SHOTS=FOLDER`). That feature is never part of a release.
