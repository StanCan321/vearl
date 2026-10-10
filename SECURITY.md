# Security policy

Vearl is a **testnet**: its tokens have no value, the code has **not been audited** by a third party, and the network can be reset
(`docs/09` is the threat model and the list of what is covered, partial or open). Even so, we want to hear about weaknesses early.

## Reporting a vulnerability

Please write to **security@vearl.org** with:

- what you found and which component it affects (consensus, networking, wallet, miner, gateway, website...),
- how to reproduce it (a test, a script, or exact steps),
- what you think the impact is.

Please do not open a public issue for a vulnerability, and do not test against nodes or services that are not yours without asking.
We will acknowledge your message within a few days, tell you what we decided, and credit you in the release notes if you wish.
There is no bug bounty at this stage.

## What matters most

- anything that lets someone mint tokens, double-spend, accept an invalid block or an invalid proof, or split the network;
- anything that lets a worker be paid for a wrong result, or lets a client avoid paying for a correct one;
- wallet and key handling (the 24-word phrase, the keystore, password handling);
- remote crashes or resource exhaustion of a node, the pool server or the gateway;
- anything in the website or the job gateway that exposes visitors' data.

## Out of scope

Known limits are written in `docs/09` and `docs/13` (for example selfish mining and the absence of a finality gadget); reports that only
restate them are welcome as discussion, not as vulnerabilities. The optional Quantus add-on starts the official Quantus miner as a separate
program: report problems of that program to its own project.
