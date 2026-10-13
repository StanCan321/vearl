# 20 - Vearl Wallet (the wallet window)

`vearl-wallet-gui` is the Vearl wallet with a window, for Windows and Linux. It is a separate crate (`vearl-wallet-gui/`) that uses the
wallet code of `vearl-core` (`wallet.rs`, `net.rs`, `regmail.rs`), so it does exactly what the command-line wallet `vearl-wallet` does:
a wallet created in one opens in the other, and the signatures and transactions are the same bytes. It is testnet software and has not been
audited.

## What it does

| page | what you can do |
|---|---|
| Overview | balance (standard and post-quantum account, counting up to its new value), address with a copy button, the latest transactions |
| Send | pay an address; the amount, the fee, a *Max* button, an optional post-quantum signature; a summary and your password before anything is sent |
| Receive | your address as text and QR code; the post-quantum address once unlocked |
| Activity | your transactions with their confirmations, from the block explorer |
| Rewards | status of your registration (read from the public list), your rank and points (read from the public leaderboard), and the registration itself: sign, check the three lines, send (docs/19) |
| Tools | sign a message with your key, verify a signature |
| Settings | node and explorer, lock delay, the 24 recovery words, password change, where the wallet file is |

## Creating a wallet

The creation is a short ceremony in six steps, shown in a rail on the left:

1. **Name and password.** At least 10 characters, not the name of the wallet, not one of the most common passwords. The wallet file format
   itself accepts 8; the window asks for more.
2. **Randomness.** The 24 words are 256 bits of entropy. They always come from the system generator (`OsRng`); the movements of the pointer
   over the window are hashed with it (BLAKE3) as an extra, so the result is never less unpredictable than the system generator alone. A
   "skip" link uses the system generator only.
3. **Recovery words.** The 24 words are hidden until you press *Show*; they decode one after the other, hide again by themselves after
   90 seconds, and the window is hidden from screenshots and recordings while they are shown (Windows).
4. **Verification.** Three words, chosen from the phrase, must be typed from the paper. Nothing is written to disk before they are right.
5. **Securing.** The key is derived from the password (Argon2id, 64 MiB, 3 passes), the phrase is encrypted (ChaCha20-Poly1305), the file is
   written, read back and unlocked once. A wallet that cannot be opened is never announced. (The checklist is a real sequence; the
   animation is paced so that it can be read, and never shorter than the real work.)
6. **Ready.** The address, and what protects the file.

*Restore* is the same, from the 24 words (checked against the word list as you type). Opening needs no password: the address is in the
file, so the balance can be shown without decrypting anything.

## Where the files are

- Wallets created by the window: `%APPDATA%\Vearl\wallets\NAME.json` on Windows, `~/.local/share/vearl/wallets/NAME.json` on Linux.
  The environment variable `VEARL_WALLET_HOME` moves everything (wallets and settings) to a folder of your choice.
- The `wallet.json` made by `vearl-miner` or `vearl-wallet new`, when it sits in the folder of the program, is listed automatically. Any other
  wallet file can be opened by its path or by dropping it on the start screen.
- Settings (node, explorer, lock delay, motion, wallets opened by path): `settings.json` next to the wallets folder (`%APPDATA%\Vearl`;
  mode 0600 on Linux).
- The file format is the one of `vearl-core/src/wallet.rs`: Argon2id (64 MiB, 3 passes) and ChaCha20-Poly1305 around the entropy of the
  24 words. Nothing is ever stored in clear.

## How the keys are handled

- The password is asked **for each action that needs a key**: sending, signing the registration, signing a message, showing the 24 words,
  showing the post-quantum address, changing the password. The keys are derived in a background thread, used, and dropped; the window never
  keeps them between two actions.
- After three wrong passwords in a row the dialog makes you wait (2, 4, 8... seconds, up to 32).
- A payment shows its summary (who, in groups of four characters, how much, fee, signature type) in the same dialog that asks the password,
  and flags a **new recipient** (an address this wallet has not paid before). The node must accept the transaction before the window says
  *sent*.
- The window **refuses to sign for another network**: the node says which genesis it is on (the Noise handshake announces it), and it is
  compared with the `genesis.json` next to the program. A node on another network is shown as such and no payment is signed for it.
- Secrets typed in the window (passwords, the 24 words of a restore, the new password) are cleared from the window's memory as soon as they
  have been used, the text fields' undo history is wiped, and the 24 words shown in a dialog are erased when it closes. The `bip39` phrase type
  zeroizes itself on drop. These are good-faith measures, not a guarantee: the operating system or the graphics layer may keep copies.
- The wallet goes back to the start screen after 10 minutes without a click or a key (Settings: never, 5, 10 or 30 minutes).
- Windows: the window can hide itself from screenshots and recordings while the 24 words are on screen (`SetWindowDisplayAffinity`,
  Settings), and the title bar takes the colours of the wallet. Linux: the process is made non-dumpable (`PR_SET_DUMPABLE`, no core files).
- The 24 words are never copied to the clipboard.
- Keyboard: Ctrl+L locks the wallet at once, Enter sends a form or a dialog, Esc closes a dialog.
- Limits (same as `wallet.rs`): a compromised computer (keylogger, malware reading memory while an action runs) defeats any software wallet,
  and the account derivation is not SLIP-10, so the 24 words only restore Vearl wallets.

## What it talks to

- **A node** (`HOST:PORT`, encrypted Noise channel, the same as `vearl-wallet balance` and `send`): balance, nonce, sending the transaction,
  the block height shown at the bottom left. The nodes are the seed nodes of `network.json` (next to the program) unless a node is set in the
  settings. The first one that answers is used.
- **The block explorer** (`https://vearl.org` by default, HTTPS with the Mozilla trust anchors): the activity list. If it is not reachable the
  rest of the window works.
- **The registration e-mail** (`mail.vearl.org:465`, TLS) only when you press *Send the registration*, and
  **`https://vearl.org/rewards/registered.json`** and **`/rewards/leaderboard.json`** to show whether your address is registered and how it
  ranks.

Nothing else is contacted. There is no telemetry and no account.

## Motion

The window is animated (screens open one block after the other, buttons and fields glow, the balance counts up, the Vearl mark of the start
screen is a sphere of dots that turns its light and swells towards the pointer). It is drawn only while something moves, and at most 66
times a second. *Settings > Reduce motion* turns every animation off.

## Recordings

`vearl-wallet-gui --demo` shows made-up figures (a wallet with a balance, activity, a standing in the leaderboard), contacts no network and
sends nothing; a wallet created in this mode gets a fixed phrase, so that a recording can be repeated, and every file goes to a temporary
folder. A "Demo data" tag is shown. With `VEARL_GUI_OPEN=1` it opens the first demo wallet straight away. The phrase of the demo wallet
belongs to no real wallet: never use it.

## Build from source

```
cd vearl-wallet-gui
cargo build --release --locked
cargo test --release --locked
```

Linux needs no development package to build; at run time it needs a graphical session (X11 or Wayland) with OpenGL (Mesa is enough) and
`libxkbcommon`. On a server without a desktop use the command-line `vearl-wallet`.

`cargo test` runs the unit tests, a headless drawing of every screen at seven window sizes and seven moments of the animations (no panic, something
painted), and, when `VEARL_NODE_BIN` points to the `node` program of a release, an end-to-end test of the window logic against a real local
node (it creates a wallet with the same commands as the buttons, pays with both signature types, checks the network identity, registers, shows
the words, changes the password). With `VEARL_WALLET_BIN` (the command-line `vearl-wallet`) a test checks that wallet files are interchangeable.
Those two are skipped otherwise.

For design work, `cargo build --release --features devshots` builds a variant that fills every screen with demo data and writes a picture of
each one, some of them at a chosen moment of their animation (`VEARL_GUI_SHOTS=FOLDER`, `VEARL_GUI_SIZE=1440x900`). That feature is never part
of a release.
