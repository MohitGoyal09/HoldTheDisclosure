---
title: Hold the Disclosure Replay
emoji: 🏦
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: cc-by-nc-4.0
short_description: Side-by-side 5-turn bank chats, 8 systems, no GPU
---

# Hold the Disclosure: transcript replay

Browse the 120 test2 dialogues of the *Hold the Disclosure* study (Lexsi AI Safety Hackathon 2026,
Track 1) side by side for any two systems: Tiny Aya Fire base, SFT and DPO adapters, the pre-registered R2, D1
and D2 retrains, and instructed Qwen2.5-72B. Each turn shows whether the reply still states every required
disclosure; turn 5 also shows whether the reply copies an earlier one and whether the LLM panel calls
it helpful. No GPU: the page reads `data.json`, built by `scripts/build_gallery.py` from the published
transcripts.

Code and report: https://github.com/MohitGoyal09/HoldTheDisclosure ·
Data: https://huggingface.co/datasets/Rockerleo/hold-the-disclosure ·
Adapters: https://huggingface.co/collections/Rockerleo/hold-the-disclosure-adapters-6af5bec87ee4a0a0e854c692
