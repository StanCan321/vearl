# 19 - Testnet contributors program

> **Status: a plan of the team, not a promise.** This document describes how the team intends to thank the people who help secure the
> public testnet, by allocating some tokens of the mainnet to them when it launches. Testnet tokens themselves have no value, are not for
> sale and cannot be converted. Nothing here is an offer, a contract, a guarantee of any amount, or investment advice, and the team may
> change or stop the program. **This draft has not been reviewed by a lawyer yet: it must be before it is announced as final.**

## 1. What is rewarded

Real, verifiable contribution to the testnet: blocks that your address mined on the public chain, counted from the blocks themselves.
Mining Quantus with the optional add-on does not count; running the miner without finding blocks is not measured.

## 2. The reward pool

The team sets aside a fixed share of the mainnet supply for the testnet contributors. Intended size: **5 % of the supply** (never more than
10 %), announced as a percentage before the testnet ends. The pool is shared between the eligible addresses in proportion to their
points (section 3). The size of the pool does not depend on how many people take part.

## 3. How points are counted

Every number is computed from the blocks of the chain, by `vearl-core/tools/testnet_rewards.py`. Parameters (they are printed in every
result):

| rule | value |
|---|---|
| a valid block mined to your address | 1 point |
| a block that carried **paid work** (a job from a client) | 1.5 points |
| early period: the first 2 weeks after the start | points ×1.25 |
| scoring periods | weeks, for the weekly leaderboard; the final score is the sum |

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

1. **Activity:** at least **7 distinct active days** (24-hour windows since the start of the testnet in which you mined at least one block).
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

1. Create your wallet with `vearl-wallet new` (or use the one the miner created) and mine with that address.
2. Sign the registration message with your wallet, replacing ADDRESS and CONTACT (your Discord name or e-mail):

   ```
   vearl-wallet sign --file wallet.json --message "Vearl testnet rewards registration. Address: ADDRESS. Contact: CONTACT. I accept the terms in docs/19."
   ```

3. Send the three lines it prints (address, message, signature) through the channel announced by the team. Nobody will ever ask for your 24
   words, your password or your wallet file. A signature proves ownership without revealing any secret.
4. Anyone can check a registration with `vearl-wallet verify --address ADDRESS --message "..." --signature HEX`, and the team does it in bulk with
   `tools/verify_registration.py`.

## 6. Snapshot, disputes and distribution

- A **final snapshot** is taken when the testnet ends: the team publishes the blocks, the script, the full leaderboard and the list of excluded
  addresses with reasons. There are **14 days** to contest a result before it becomes final.
- The top of the ranking is reviewed by hand before anything is paid.
- **Distribution at the mainnet launch**, as a plan: a first part (planned 25 %) at the launch, the rest (planned 75 %) spread over 6 to 12
  months, which rewards people who stay and avoids an immediate sell-off. The mainnet does not yet have an on-chain vesting mechanism: until it
  does, the later parts are released by the team in tranches, and the team will say which mechanism is used before the first payment.
- The leaderboard data is kept outside the chain, because the testnet can be reset; the published snapshot is what counts.

## 7. Personal data

Only what is needed: your wallet address, the contact you give at registration, and what is public on the chain. The team does not collect
identity documents for this program, and may ask for more information later only if the law requires it. You can ask the team to delete your
contact details at any time (the on-chain data stays).

## 8. Changes and contact

The rules may change; changes are announced before they apply and never reduce points already earned for blocks mined before the
announcement. Questions and corrections: contact@vearl.org. Security reports: security@vearl.org.
