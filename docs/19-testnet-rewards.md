# 19 - Testnet contributors program

> **Status: a plan of the team, not a promise.** This document describes how the team intends to thank the people who help secure the
> public testnet, by allocating some tokens of the mainnet to them when it launches. Testnet tokens themselves have no value, are not for
> sale and cannot be converted. Nothing here is an offer, a contract, a guarantee of any amount, or investment advice, and the team may
> change or stop the program. **This draft has not been reviewed by a lawyer yet: it must be before it is announced as final.**

## 1. What is rewarded

Real, verifiable contribution to the testnet: blocks that your address mined on the public chain, counted from the blocks themselves.
Mining Quantus with the optional add-on does not count; running the miner without finding blocks is not measured.

## 2. The reward pool

The team sets aside a fixed share of the mainnet supply for the testnet contributors. Planned size: **5 % of the mainnet supply**, announced as a
percentage before the testnet ends. The mainnet is expected soon after the start of the testnet, so the durations in this document (early period, active days) are counted in days; if the schedule changes, the team announces the new values before they apply. The pool is shared between the eligible addresses in proportion to their
points (section 3). The size of the pool does not depend on how many people take part.

## 3. How points are counted

Every number is computed from the blocks of the chain, by `vearl-core/tools/testnet_rewards.py`. Parameters (they are printed in every
result):

| rule | value |
|---|---|
| a valid block mined to your address | 1 point |
| a block that carried **paid work** (a job from a client) | 1.5 points |
| early period: the first 3 days after the start | points ×1.25 |
| scoring periods | days, for the daily leaderboard; the final score is the sum |

The leaderboard (<https://vearl.org/leaderboard/>) is refreshed regularly. It is built from the public files `leaderboard.json`,
`leaderboard.csv` and `blocks.json` (the blocks used, with their SHA-256): you can download the blocks and re-run the same script to get
the same result, bit for bit.

Why blocks: a block carries the address of the miner who found it, and nobody can find a block without really doing the work, so the number of
blocks is proportional to the real computing power and cannot be inflated off-chain. A person who splits one machine over several addresses
gets the same total points, so there is nothing to gain by creating many addresses.

Not counted in this first version: a bonus for completed jobs (a worker can create jobs for itself with free testnet tokens, so a bonus would need
a safe way to ignore self-dealing). It may be added later; the rules in force are always the ones in this file.

## 4. Who is eligible

All of the following:

1. **Activity:** at least **3 distinct active days** (24-hour windows since the start of the testnet in which you mined at least one block).
2. **Minimum share:** at least **0.005 %** of all the points, to avoid dust.
3. **Registered address:** you registered the address, with a signature that proves that you own it (section 5), before the deadline
   announced by the team.
4. **Honest behavior:** no fraud (a worker proven to have published a wrong result, a manipulated block, a forged registration), and no attack on
   the network or on other participants outside the rules of `SECURITY.md`. Responsible reports of weaknesses are welcome and are rewarded
   separately.
5. **One participant per person.** You may use several machines, but not several identities to receive more than your share. Addresses that
   look like one person's are reviewed by hand.
6. **Legal conditions:** you are of legal age in your country, you are not in a country or on a list that the team has to exclude by law
   (the list is published before the registration opens), and you accept these terms.

The team may exclude an address that breaks these rules, and publishes the address and the reason.

## 5. How to register

Registration is done by e-mail, but you do not have to write the e-mail yourself: the wallet sends it. With the wallet window (`vearl-wallet-gui`, docs/20) it is the **Rewards** page: type your contact, press *Sign the registration*, check the three lines, press *Send the registration*. The command line below does the same.

1. Mine with an address of your wallet (the one the miner created, or `vearl-wallet new`).
2. In the folder of the wallet, run (replace the contact with your Discord name or an e-mail address):

   ```
   vearl-wallet register --file wallet.json --contact "YOUR_DISCORD_NAME_OR_EMAIL"
   ```

   The wallet asks your password, signs the registration message
   `Vearl testnet rewards registration. Address: ADDRESS. Contact: CONTACT. I accept the terms in docs/19.`, **shows you the three lines**
   (address, message, signature) and asks you to confirm before it sends **one** e-mail from `registrar@vearl.org` to `register@vearl.org`.
   Nothing secret is sent: a signature proves that you own the address without revealing any key. Nobody will ever ask for your 24 words, your
   password or your wallet file.
3. A script on the server reads that mailbox every two minutes, checks the signature and publishes the list of registered addresses (addresses
   only, never the contacts) at <https://vearl.org/rewards/registered.json>. Your address should appear there within a few minutes. The
   leaderboard marks only registered addresses as eligible.
4. If sending fails (some networks block outgoing mail), the wallet says why and shows the three lines: send them yourself in an e-mail to
   `register@vearl.org` (subject: Vearl testnet registration). `vearl-wallet register --file wallet.json --contact "..." --print-only` prints
   them without sending anything. The old way still works too: `vearl-wallet sign --file wallet.json --message "..."`.
5. Anyone can check a registration with `vearl-wallet verify --address ADDRESS --message "..." --signature HEX`, and the team does it in bulk with
   `tools/process_registration_mail.py` (the script that reads the mailbox) and `tools/verify_registration.py`.

How it is protected: `registrar@vearl.org` is a send-only account whose password is public (it is in `vearl-core/src/regmail.rs`, because the
wallet needs it). The mail server lets it write to `register@vearl.org` only, on a dedicated port, with a small size and a rate limit, and it has
no mailbox. Nobody can register an address that they do not own: the script accepts a registration only if the signature is valid, whoever sent
the mail. A registration sent twice counts once; a contact that registers several addresses is reviewed by hand (one participant per person).

## 6. Snapshot, disputes and distribution

- A **final snapshot** is taken when the testnet ends: the team publishes the blocks, the script, the full leaderboard and the list of excluded
  addresses with reasons. There are **14 days** to contest a result before it becomes final.
- The top of the ranking is reviewed by hand before anything is paid.
- **Distribution at the mainnet launch**, as a plan: a first part (planned 25 %) at the launch, the rest (planned 75 %) spread over 6 to 12
  months, which rewards people who stay and avoids an immediate sell-off. The mainnet does not yet have an on-chain vesting mechanism: until it
  does, the later parts are released by the team in tranches, and the team will say which mechanism is used before the first payment.
- The leaderboard data is kept outside the chain, because the testnet can be reset; the published snapshot is what counts.

## 7. Personal data

Only what is needed: your wallet address, the contact you give at registration (kept private, never published), and what is public on the chain. The mail server sees the IP address that sends the registration, like any web server, and keeps its usual logs. The team does not collect
identity documents for this program, and may ask for more information later only if the law requires it. You can ask the team to delete your
contact details at any time (the on-chain data stays).

## 8. Changes and contact

The rules may change; changes are announced before they apply and never reduce points already earned for blocks mined before the
announcement. Questions and corrections: contact@vearl.org. Security reports: security@vearl.org.
