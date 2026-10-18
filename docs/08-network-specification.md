# 08 — Specification of the P2P network and chain selection (v1)

**Status: testnet, not audited.** Complements `docs/07` (work layer, chain, market). This text allows a second implementer to **talk to a VEARL node** (connect, authenticate, synchronise) and fixes the **chain-selection rule** that all nodes must apply.

## 1. Transport

- TCP. One **initiator** (outbound) peer and one **responder** (inbound).
- Encrypted and authenticated channel: **Noise_XX_25519_ChaChaPoly_BLAKE2s** (standard library; no home-made primitive). XX scheme: both static X25519 keys are transmitted and authenticated during the handshake; a peer's identity is its static key (an operator can **pin** it: `host:port@hex_key`).
- **Framing of handshake messages**: each Noise message is preceded by its length on **2 big-endian bytes**. Order: `→ e`; `← e, ee, s, es`; `→ s, se` (empty payload).
- **Transport framing**: an application message `M` (bytes) is turned into `plain = length(M) on 4 little-endian bytes || M`, cut into chunks of at most **65,519 bytes**; each chunk is encrypted (Noise, 16-byte tag) and preceded by the length of the ciphertext on **2 big-endian bytes**. On reception one decrypts, concatenates, and extracts the complete messages. Maximum size of an application message: 16 MiB (beyond: connection closed).
- An application message is a `Msg` encoded in **bincode 1.x** (fixed little-endian integers, `u32` enumeration tag, `Vec` = `u64` length + elements, `Option` = byte 0/1, `String` = `u64` length + UTF-8), see `docs/07` §2.

## 2. Messages

Tags (declaration order): `Hello` 0, `GetBlocks` 1, `Blocks` 2, `NewBlock` 3, `NewTx` 4, `GetPeers` 5, `Peers` 6, `GetStatus` 7, `Status` 8, `GetAccount` 9, `Account` 10, `Ack` 11, then the **off-chain** messages of the compute market: `PutJob` 12 … `OpData` 35 (deposit and reading of job data at a worker: `docs/05`, not specified here).

| message | fields | role |
|---|---|---|
| `Hello` | `node_id u64`, `genesis [32]`, `height u64`, `work u128 (16 B LE)`, `listen Option<String>` | first message **from each side**; announces the genesis, the height and the cumulative work |
| `GetBlocks` | `from u64` | requests blocks with index `≥ from`, **64 at most** |
| `Blocks` | `Vec<Block>` | answer (≤ 64 blocks); an empty batch = end |
| `NewBlock` | `Block` | broadcast of a new block |
| `NewTx` | `SignedTx` | broadcast of a transaction |
| `GetPeers` / `Peers` | — / `Vec<String>` | address exchange (32 max) |
| `PutProgMore` (tag 40) | `x0_root`, `ws Vec<Vec<i8>>` | continuation of the weights of a program whose input and first weights `PutProg` already deposited; the weights are appended in order (≤ 32 in total), a message staying ≤ 16 MiB |
| `GetOpsBox` (tag 55) | `job [32]`, `ops Vec<u8>` | outputs and `Z` of **several** operations of a program in one round trip (replaces one `GetOp` request per operation) |
| `OpsBox` (tag 56) | `Vec<(x_dims (u32, u32), x Vec<i8>, z_dims Option<(u32, u32)>, z Vec<i32>)>` | a **prefix** of the requested operations (at least one, at most 8 MiB); each `n × n` matrix is reduced to its **bounding box**: the top-left `rows × columns` corner outside which everything is zero. The client rebuilds the matrices (dimensions ≤ `n` checked) then checks their roots against those published on-chain, exactly as with `OpData`: it is a transport encoding, not a consensus rule. A worker that does not know this message is queried with `GetOp`. |

`Block = Header || TileProof || Vec<SignedTx>`; `TileProof = ti u32, tj u32, a_rows Vec<Vec<i8>>, a_paths Vec<Vec<[32]>>, b_cols Vec<Vec<i8>>, b_paths Vec<Vec<[32]>>` (16 rows of `A`, 16 columns of `B`, with their Merkle paths: `docs/07` §6).

**Genesis hash** (announced in `Hello`, must be identical on both sides, otherwise the peer is disconnected):
`H("vearl/genesis/v1", bincode((params, alloc, timestamp, target, timing, retarget)))` with `params = 4×u64 (n, tile, rank, fold_k)`, `alloc = Vec<([32], u64)>`, `timestamp u64`, `target u64`, `timing = 2×u64 (challenge_secs, serve_secs)`, `retarget` = `u32` tag (Ema 0, Lwma 1, Fixed 2, Asert 3). The JSON genesis file expresses addresses in Bech32m (`vrl1…`, data = 32-byte public key) and amounts in decimal VRL (8 decimals).

## 3. Application handshake and admission

1. After Noise, **each side sends `Hello` first**. A `Hello` with a different genesis closes the connection. An identity (static key) that is already connected is refused.
2. A client that does **not** send a `Hello` can only issue simple requests (account state, status, data deposit, submission of a transaction); it receives neither the chain nor the gossip.
3. **Admission limits** (basic anti-eclipse, not a Sybil defence): at most 24 inbound PEERS (nodes that say Hello), 2 per IP and 4 per /16 subnet (excluding loopback; `--max-inbound`, `--max-per-ip`, `--max-per-subnet` change them, a public seed raises them); wallets and command-line tools are not peers and are bounded only by the number of simultaneous connections (256 in total, 16 per IP; `--max-conns`); a peer that does not fit receives the list of known addresses before it is closed; at most 6 outbound peers, never two in the same /16; 64 simultaneous handshakes.
4. **Rate**: token bucket per peer, 100 messages/s, burst 200; exceeding = disconnection. **Behaviour**: an invalid message (invalid block, repeated transaction with an invalid signature/nonce) earns a score; an invalid block or 100 points = **disconnection and IP ban**.

## 3 bis. Mempool (local rule, non-consensus)

The mempool is not a consensus rule: two nodes can implement it differently without splitting. The reference node applies:

- **Admission** of a transaction: valid signature; `nonce` = account nonce + number of its transactions already pending (a sender's nonces form a sequence); `fee ≤ balance`; **non-zero balance** (an empty account cannot pay anything: otherwise disposable keys would fill the mempool); at most **16** pending transactions per sender (**64** in later versions: `docs/06` §4 ter), **5,000** in total; duplicates ignored. A rejection of the signature/nonce/balance type counts for 10 points of misbehaviour (100 = disconnection and IP ban).
- **Selection** for a block: in arrival order, each transaction is tried on a copy of the state at the time of the block (height, timestamp); those that fail are skipped.
- **Eviction of stuck transactions** (found by the endurance test `tools/soak.py`): if a sender's first pending transaction can no longer go through — either its nonce is not the expected one (**gap**: the missing transaction was lost in a reorganization or a restart) for more than **10 s**, or it fails on application (for example a bid that arrived after the end of the auction) for more than **30 s** — all of that sender's pending transactions are evicted. Without this rule, a single blockage freezes all of the sender's submissions and, since a worker does not resend a transaction it believes pending, **it stops working**.
- Purge after each tip change: those whose nonce is consumed are removed. After a reorganization, the transactions of the abandoned branch that the new chain did not include are **put back into the mempool** (in order, subject to the admission rules: a nonce already consumed by the new chain is refused).
- An `Ack ok` returned to a client means **"accepted into this node's mempool"**, not "included in a block": a fraud proof only has an effect if it is included before the end of the challenge window; the `vearl-cli *-finish` tool therefore waits for the final state of the job.

## 3 ter. Worker data

The data deposited by clients at a worker (`PutJob`, `PutMlp`, `PutTiled`, `PutProg`, `PutProgMore`) is **persisted** in the data directory (`putdata.bin`, 512 MiB at most) and reloaded at start-up; a worker that restarts resumes its jobs (layers and operations already published on-chain are recomputed, the result of a published job is recomputed so that the requested data can be served).

## 4. Synchronisation

- A peer whose announced `work` is **strictly greater** than ours is synchronised: we send `GetBlocks{from = our height}`.
- If the first block received **connects** (`height = our height`, `prev = our tip`): we validate and apply each block (`docs/07` §10.3, §11); as long as the batch is full (64), we ask again from the new height.
- Otherwise (**fork**): the original rule downloaded **the peer's whole chain from 0** in batches of 64 (at most 200,000 blocks), **replayed it from genesis in a fresh state** and validated it entirely. (Later versions locate the common ancestor by doubling backward and reorganize atomically from state snapshots: `docs/13` §2.)
- `NewBlock`: if `height = our height` and `prev = our tip`, validation and broadcast; if `height ≥ our height` otherwise, the peer is treated as having more work and we synchronise; a block already seen (hash) is ignored.
- Clock: a block whose timestamp exceeds the local clock by more than `MAX_FUTURE = 60` s is refused **at reception** (local, non-deterministic rule; the chain replayed later accepts it if it is valid).

## 5. Chain selection (consensus rule of the nodes)

Each block weighs `⌊2⁶⁴ / (target + 1)⌋`; the **cumulative work** is the sum over the chain. A candidate chain **replaces** the current chain **if and only if its cumulative work is strictly greater**; **on a tie we keep the current chain** (first seen). The candidate is validated entirely (block by block, from genesis) before replacement: a peer cannot make a node adopt an invalid chain. After a replacement: the chain file is rewritten (temporary file + atomic rename), the mempool is purged of transactions whose nonce is consumed, the mining in progress is cancelled. (The optional MESS braking of deep reorganizations is in `docs/13` §3.)

Known limit: an attacker who controls several subnets can eclipse a node (`docs/02`).

## 6. Verification by an independent implementation (measured)

`tools/light_client.py`, written from this text and `docs/07` (without reading the node's code), was run against a **real `node` process** (`tools/interop_p2p.sh`):

- **Noise XX** handshake with a Rust node (the Python library `noiseprotocol` against `snow`); the genesis hash computed in Python from the JSON file (Bech32m addresses decoded in Python) is **accepted** by the node;
- `Hello`s exchanged, paginated `GetBlocks`; **85 to 91 blocks** received and decoded (headers, tile proofs, transfer transactions signed by the Rust wallet);
- **full re-validation** with the independent validator: work recomputed for each block, Ed25519 signatures, nonces, balances, rewards, retargeting, and consistency of the rows/columns of each `TileProof` with the native matrices.

Limits: the light client does not speak the gossip (`NewBlock`/`NewTx` received are ignored); it only decodes Native blocks + transfers (the other work classes and transactions raise an explicit error); reorganizations and admission by addresses are not tested by this client; same author for the code and the text `[AUDIT ACTION]`. (Since then the client also checks headers v2 and account proofs: `docs/14`.)
