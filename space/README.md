---
title: Hold the Disclosure Replay
emoji: 🏦
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: cc-by-nc-4.0
short_description: 5-turn bank-chat transcripts, base vs SFT vs DPO vs Qwen
---

# Hold the Disclosure: transcript replay

Browse the 120 test2 dialogues of the *Hold the Disclosure* study (Lexsi AI Safety Hackathon 2026,
Track 1) side by side for two systems. Each turn shows whether the reply still states every required
disclosure; turn 5 also shows whether the reply copies an earlier one and whether the LLM panel calls
it helpful. No GPU: the page reads `data.json`, built by `scripts/build_gallery.py` from the published
transcripts.

Code and report: https://github.com/MohitGoyal09/HoldTheDisclosure ·
Data: https://huggingface.co/datasets/Rockerleo/hold-the-disclosure
