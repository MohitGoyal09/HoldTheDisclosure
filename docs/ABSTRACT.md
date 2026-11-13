# Abstract

A bank's support chatbot must state required risk disclosures (lock-in, exit load, market risk,
interest charges) even when the customer pushes back. We test whether small multilingual models
keep them over five turns of scripted pushback in English, Devanagari Hindi and romanized
Hinglish, and whether fine-tuning fixes the failure. The product facts are always in the system
prompt; our main prompt does not tell the model to state them, so we measure uninstructed
retention, with an instructed ("strict") baseline. DR@5 is the share of dialogues whose turn-5
reply states every required disclosure, scored by a rule checker.

Uninstructed, the failure is large. Tiny Aya Fire (3.35B) states the disclosures at turn 1 in
every English and Hindi test dialogue but keeps them at turn 5 in 0% and 15% (13.3% pooled).
Llama-3.2-3B keeps 0% and Qwen2.5-72B 3.3%. Instruction helps unevenly: the strict prompt lifts
Qwen2.5-72B to 41.7% on test but not on held-out products (28.3%), and lifts Tiny Aya on new
pushback lines (40.0%) but not on the first test split (11.7%).

LoRA supervised fine-tuning and DPO (through AlignTune) on gated synthetic replies raise DR@5. On 120 new, native-reviewed pushback dialogues that no system was tuned on, all five trained systems beat base (32.5 to 52.5% against 16.7%; all Holm p < 0.01,
pre-registered). The gain holds over training runs, a second model family, held-out products and a
shuffled turn order.

A post hoc check found that most passing turn-5 replies of trained systems copy an earlier
reply of the same model (29% of SFT dialogues on the new lines pass with a verbatim copy;
instructed Qwen2.5-72B almost never copies). We then pre-registered a test of what copying
costs. A three-family LLM panel rates copied passes helpful far less often than
fresh ones in the same dialogues (34.5% against 66.4%, p = 0.0001). Counting only turn-5 replies
that keep the disclosure and help (HD@5), SFT still beats base by 23.3 points (27.5% against
4.2%, p = 0.0002) and is not detectably different from instructed Qwen2.5-72B (30.8%, p = 0.7). About a third of SFT's
DR@5 gain is repetition that does not answer the customer. A post hoc rule appending the
missing disclosure sentence to every base reply does better still (HD@5 55.0%; 47.5% by meaning-based judges). SFT also copies on control products (51 of
60 dialogues, base 13); its training replies were already near-repeats (median similarity 0.86).

Other costs: no trained system adds a warning to products that need none; SFT answers control
questions less often than base (not significant after correction); one
DPO mix often answers Hinglish customers in Devanagari.

Lessons: a per-turn compliance metric must also check that the
reply answers the customer and be compared with a simple rule, and multi-turn training replies
must vary across turns. Limitations:
one fictional bank, synthetic scripted dialogues, 20 to 40 independent scenarios per comparison,
helpfulness from model labels, and a post hoc repetition finding, then tested as
pre-registered. Code, data and adapters are released.
