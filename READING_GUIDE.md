# WalkVLM-LR — Reading Guide (Code + Paper, From Scratch)

A step-by-step plan for understanding **WalkVLM-LR** as a complete system: what the paper proposes, how the two trained models (the EAD classifier and the GRPO-fine-tuned VLM) are wired together, and where every component lives in the code. Read top to bottom — each stage assumes the previous one.

The goal at the end: you should be able to look at any file in this repo and explain what it does, what it consumes, and what it produces, and you should be able to map each block in the paper's architecture diagram (Figure 2) to a concrete piece of code.

---

## 0. Orient yourself first (10 minutes)

Before reading anything technical, build a one-paragraph mental model of the project.

1. Open [README.md](README.md) — the repo-level README. Read just the **Overview** section.
2. Open [WalkVLM-LR/README.md](WalkVLM-LR/README.md) — slightly more detailed.
3. Open the architecture figure: [figures/framework.jpg](figures/framework.jpg).
4. Open the GRPO figure: [figures/GRPO.jpg](figures/GRPO.jpg).

**One-paragraph summary to internalize:** WalkVLM-LR is a walking-assistance system for blind/low-vision (BLV) users. It takes the last N video frames and decides (a) **whether** to speak, and (b) **what** to say. A small classifier (**EAD**) shares the VLM's visual encoder and predicts a danger level A/B/C per frame; only on "danger" does the **VLM** (Qwen2-VL-2B fine-tuned with GRPO) generate a short, keyword-dense navigation reminder. The two contributions are: **less output redundancy** (via 4 custom GRPO rewards) and **less temporal redundancy** (via EAD gating instead of running the LLM every frame).

---

## 1. Read the paper for the mental model (60–90 minutes)

File: [WalkVLM-LR.pdf](WalkVLM-LR.pdf), 15 pages.

Read **in this order**, and as you go, write down the named entities you encounter (e.g. "Simplicity Reward", "EAD", "SAIM", "WAD dataset"). You will need this glossary while reading the code.

### 1.1 Pass 1 — skim for vocabulary (15 min)

- **Abstract + Introduction (p.1–2)**: identify the two problems (*output redundancy*, *temporal redundancy*) and the two solutions (*4 GRPO rewards*, *EAD*).
- **Figure 1**: notice the axes — output length vs. keyword density. This is the central pitch.
- **Figure 2 (p.3)**: spend real time here. Trace the data flow: frames → visual encoder → (EAD branch) and (LLM branch). The EAD reuses the visual encoder. This is the most important figure in the paper.
- **Figure 3 (p.4)**: how GRPO uses 4 rewards inside a group-relative advantage. Don't worry about the equations yet.

### 1.2 Pass 2 — Methods section, slowly (30–45 min)

Read pages 3–5 ("Methods") equation by equation. Goal: know what each reward measures and what its formula looks like.

| Concept | Where in paper | What it does | Maps to (code) |
|---|---|---|---|
| Simplicity Reward (Eq. 1) | p.3 | Penalizes deviation from ideal output length L₀ | `simplicity_reward` in [grpo_query_gene.py](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py) |
| Fluency Reward (Eq. 2–4) | p.4 | GPT-2 perplexity + n-gram diversity | `fluency_reward` (same file) |
| Accuracy Reward (Eq. 5–7) | p.4 | CLIP cosine sim + mean token accuracy | `semantic_similarity_reward` (same file) — note: paper calls it Accuracy, code calls it `sematic` |
| Keywords Reward (Eq. 8) | p.4 | CLIP-synonym frequency of annotation keywords | `keywords_reward` (same file) |
| EAD architecture | p.5 | MultiScaleConv → SAIM (Transformer) → MLP → A/B/C | [EAD.py](WalkVLM-LR/EAD.py): `MultiScaleConv`, `VisionDangerClassification` |
| Problem formulation | p.3 | At time tₙ, use [f_{tₙ-N}…f_{tₙ}] to predict danger | EAD takes 3 history frames (see Appendix A1, `n=3`) |

### 1.3 Pass 3 — Experiments + Ablation (15–20 min)

- **Table 1 (p.6)**: WalkVLM-LR at 2B beats 72B baselines on ROUGE / KeyDens / GPT Score.
- **Table 2**: TRF metric — temporal F1 for whether-to-speak decisions.
- **Table 5**: ablation per reward (drop one, see what suffers). Useful to know which reward governs which quality.
- **Table 6**: ablation per EAD module (MSC / SAIM / MLP). Confirms each block matters.
- **Appendix A1 (p.11)**: exact training hyperparameters — match these against [run_grpo_query_gene.sh](WalkVLM-LR/vlm_grpo_template/run_grpo_query_gene.sh).
- **Appendix B1 (p.12)**: WAD dataset = 12k videos / 120k images, 3 danger grades, 6 reminder types.
- **Appendix C2 (p.14)**: EAD training details — vision encoder frozen, cross-entropy + focal loss, 4 epochs.
- **Appendix D1 (p.14–15)**: failure cases — read these. They reveal the model's real-world limits (proximate-obstacle blindness, directional misjudgment).

After this pass you should be able to answer, without looking at code:
1. What two redundancies does WalkVLM-LR reduce?
2. What four rewards drive GRPO, and what does each measure?
3. What is EAD, why does it share the visual encoder, and what does it output?
4. What's the TRF metric and why is it the right one for temporal redundancy?

---

## 2. Map the repo (10 min)

Skim the layout. Don't read content yet — just know where things live.

```
.
├── README.md                       # top-level overview
├── WalkVLM-LR.pdf                  # the paper
├── figures/                        # framework.jpg, GRPO.jpg, visual.jpg
├── wad_dataset/                    # dataset download script + tar URL
└── WalkVLM-LR/
    ├── README.md                   # project usage
    ├── EAD.py                      # the EAD model (training-time prototype)
    ├── train_EAD.py                # distributed trainer for EAD
    ├── inference.py                # end-to-end: EAD gate → VLM reminder
    ├── test.py                     # evaluation (ROUGE / KeyDens / GPT Score)
    ├── checkpoint/                 # download_checkpoint.sh (Qwen2-VL, CLIP, GPT-2)
    └── vlm_grpo_template/          # GRPO training rig (fork of open-r1)
        ├── run_grpo_query_gene.sh  # launch script
        ├── configs/                # accelerate/DeepSpeed yaml
        └── src/open_r1/
            ├── grpo_query_gene.py  # ★ THE training entry point with 4 rewards
            ├── grpo.py             # generic open-r1 GRPO (math-style)
            ├── sft.py              # supervised baseline trainer
            ├── generate.py         # vLLM batch generation
            ├── evaluate.py         # eval helpers
            └── trainer/
                ├── grpo_trainer.py # ★ Qwen2VLGRPOTrainer (loss, generation, KL)
                └── InternVL2.py    # InternVL2 image preprocessing (optional)
```

Two files do the heavy lifting and deserve the most attention later:
- [grpo_query_gene.py](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py) — defines the 4 rewards, builds the dataset, kicks off training.
- [grpo_trainer.py](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py) — implements GRPO's group-relative advantage + per-token KL loss for a Qwen2-VL backbone.

---

## 3. Read the code in dependency order (3–4 hours)

The trick is to follow the *inference* path first (simplest, smallest) and only then the *training* paths (EAD trainer, then GRPO trainer). That way you've already seen every model class before you watch it being optimized.

### Step 3.1 — End-to-end inference (read first)

**File:** [WalkVLM-LR/inference.py](WalkVLM-LR/inference.py)

This is ~200 lines and shows the complete deployed pipeline. Read it once top-to-bottom.

What to track:
- [inference.py:57–63](WalkVLM-LR/inference.py:57): loads the GRPO-fine-tuned `Qwen2VLForConditionalGeneration` + its processor.
- [inference.py:65–73](WalkVLM-LR/inference.py:65): loads the trained EAD (`VisionDangerClassification`) from `checkpoint.pth`.
- [inference.py:125–179](WalkVLM-LR/inference.py:125): `infer_batch` — feeds 3 frames through `model.module.visual` (the **shared visual encoder**), pools the features, calls EAD, returns A/B/C labels.
- [inference.py:181–188](WalkVLM-LR/inference.py:181): `infer_with_check` — **the gating logic**. If any of the 3 frames is graded `C` (high danger), trigger the VLM to generate a reminder; otherwise return `"No reminder needed"`.
- [inference.py:184](WalkVLM-LR/inference.py:184): the prompt template — 5 rules for the reminder (key elements, directions like clock positions, object specifics, action, safety/clarity). This same prompt appears in `test.py` and inside the training dataset builder.

**Key insight to verify yourself:** look at [inference.py:161–168](WalkVLM-LR/inference.py:161). The line `visual_model = model.module.visual` is the literal implementation of "EAD shares the visual encoder with the VLM" from the paper — the same Qwen2-VL ViT produces features for both the LLM and the EAD classifier. There is no second encoder.

### Step 3.2 — The EAD model

**File:** [WalkVLM-LR/EAD.py](WalkVLM-LR/EAD.py) (103 lines)

Reads as three blocks that mirror Figure 2 (bottom-right of the framework figure):

1. `MultiScaleConv` ([EAD.py:5–23](WalkVLM-LR/EAD.py:5)) — three parallel 1-D convs (kernels 3/5/7) over the sequence of patch tokens, concatenated. This is the **MSC** module in Table 6.
2. `VisionDangerClassification` ([EAD.py:25–48](WalkVLM-LR/EAD.py:25)) — MSC → `MultiheadAttention` (this is **SAIM**, the Scene Awareness Inference Module in the paper) → 3-layer MLP → 3-class logits.
3. Losses ([EAD.py:50–88](WalkVLM-LR/EAD.py:50)) — `FocalLoss` + `LabelSmoothingLoss`, summed with cross-entropy. Focal loss handles the class imbalance noted in Appendix C2 (most frames are low-danger A).

Cross-reference now:
- Map MSC → paper §"Reducing Temporal Redundancy with EAD" — "capturing target information at different scales using a multi-scale convolutional network."
- Map SAIM → "Scene Awareness Inference Module, composed of several layers of stacked transformers." (Code uses a single `MultiheadAttention` layer — simpler than the paper text implies, but functionally equivalent.)
- The `input_dim=1536` ([EAD.py:90](WalkVLM-LR/EAD.py:90)) matches the Qwen2-VL ViT hidden size — confirming the shared-encoder design.

### Step 3.3 — Training EAD

**File:** [WalkVLM-LR/train_EAD.py](WalkVLM-LR/train_EAD.py) (210 lines)

This is a standalone supervised trainer; it expects pre-extracted visual features cached as JSON per video folder.

What to track:
- [train_EAD.py:15–65](WalkVLM-LR/train_EAD.py:15): `VideoDataset` — walks a directory, loads `{folder}_features.json` per video, expects each entry to have `visual_features` + a `label` string whose 2nd token is "A"/"B"/"C".
- [train_EAD.py:67–105](WalkVLM-LR/train_EAD.py:67): two alternative classification heads (`MLPModel`, `VisualDangerClassificationHead`) — note these are **not used by `main()`**; `main` instantiates `VisionDangerClassification` from `EAD.py`. (Likely dead/experimental code.)
- [train_EAD.py:159–189](WalkVLM-LR/train_EAD.py:159): vanilla DDP loop. CE + Focal + LabelSmoothing summed.
- [train_EAD.py:191–209](WalkVLM-LR/train_EAD.py:191): `main()` — uses the EAD model and wraps it in `DistributedDataParallel`. **`data_dir` is empty** ([line 194](WalkVLM-LR/train_EAD.py:194)) — you must fill this in.

**Important to notice**: this trainer assumes someone *already ran the Qwen2-VL visual encoder over your video frames and saved features to disk*. The repo does not ship that feature-extraction script; you'd build it yourself using the same `model.module.visual(...)` call seen in [inference.py:166](WalkVLM-LR/inference.py:166).

### Step 3.4 — The four GRPO rewards (the heart of the paper)

**File:** [WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py) (637 lines)

Read in this order; each section corresponds to one equation from the paper.

| Read | Lines | Paper equation | Reward registry key |
|---|---|---|---|
| `simplicity_reward` | [506–528](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:506) | Eq. 1 (L₀=20 in code) | `simplicity` |
| `compute_perplexity` + `fluency_reward` | [171–239](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:171) | Eq. 2–4 | `fluency` |
| `semantic_similarity_reward` | [264–306](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:264) | Eq. 5–7 (paper "Accuracy") | `sematic` *(typo preserved)* |
| `calculate_keyword_density` + `keywords_reward` | [308–390](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:308) | Eq. 8 | `keywords` |
| `reward_funcs_registry` | [534–543](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:534) | — | dispatch table |

Tips while reading:
- Each reward returns a list of floats, one per generated completion. Higher is better. They're all roughly in `[0, 1]`.
- `fluency_reward` is a **composite**: it combines GPT-2 perplexity (lower = better → mapped through sigmoid) with weighted 1-gram + 2-gram diversity (higher = better). The final formula `diversity / (diversity + λ·gpt_reward)` ([line 233](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:233)) doesn't exactly match Eq. 4 of the paper — note this as a place where code drifted from text.
- `semantic_similarity_reward` ([line 302](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:302)) uses weights `a=0.75` for token accuracy and `1-a=0.25` for cosine — paper Eq. 7 sums them with equal weight. Another code/paper drift.
- The keyword reward is **slow**: it CLIP-encodes every n-gram against every keyword (O(tokens × keywords) CLIP forward passes). This matters for training throughput.

Also read the dataset construction ([lines 545–603](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:545)):
- Input JSONL with fields `frame_path`, `alter` (reference reminder), `keywords`.
- Per sample, it loads `wad_dataset/src_data/{frame_path}/8.jpg` — note that **only frame 8** is given to the VLM during GRPO training (not a sequence). The temporal reasoning happens in EAD; the VLM just does single-frame conditional generation.
- Solution is wrapped as `<answer>...{alter}...</answer>` — used by `accuracy_reward` (which is *not* in the default reward set; the default uses `sematic`/`keywords`/`fluency`/`simplicity`).

### Step 3.5 — The GRPO trainer

**File:** [WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py) (630 lines)

Read the class top to bottom. The interesting parts:

1. **`__init__` ([lines 150–340](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:150))** — model loading branches on the model name (Qwen2-VL vs. Aria vs. InternVL2). Reference model is created via `create_reference_model` or loaded fresh under DeepSpeed Zero-3. Left-padding tokenizer for generation.
2. **`_get_per_token_logps` ([350–363](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:350))** — computes log-probs per token. Used for both the policy and the reference, to form KL.
3. **`compute_loss` ([371–563](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:371))** — *this is the GRPO algorithm itself*:
   - Build prompt tensors (text + image patches).
   - Generate `num_generations` completions per prompt (a "group").
   - Mask after EOS.
   - Run reward functions over the completions ([482–514](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:482)). All 4 are summed with equal weight `[1,1,1,1]` ([line 516](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:516)).
   - **Group-relative advantage** ([522–528](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:522)): mean and std within each group of `G` samples; standardize to get advantages.
   - **Per-token loss** ([540–542](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:540)): policy-gradient term (`exp(logp - logp.detach()) * advantage`) minus `β · KL`.
4. **Trajectory logging ([530–538](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:530))** — dumps prompts/responses/rewards per step under `trajectories/`. Useful for debugging reward shaping.

Cross-reference Figure 3 of the paper while you read `compute_loss`. The "intra-group relative advantage" arrow is precisely the standardization in [lines 522–528](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:522).

### Step 3.6 — The launch script

**File:** [WalkVLM-LR/vlm_grpo_template/run_grpo_query_gene.sh](WalkVLM-LR/vlm_grpo_template/run_grpo_query_gene.sh)

Compare each flag to Appendix A1 of the paper:
- 8 GPUs (`CUDA_VISIBLE_DEVICES=0..7`) + `configs/zero2.yaml` → DeepSpeed Zero-2.
- `max_prompt_length 1024`, `max_completion_length 700`, `per_device_train_batch_size 1`, `gradient_accumulation_steps 2`, `bf16`, `flash_attention_2`, `max_pixels 401408`, `num_train_epochs 1` — every value matches the paper.
- Several string flags are blank in the shipped file (`--output_dir`, `--model_name_or_path`, `--run_name`, `--dataset_prefix`, `--dataset_path`) — you must fill these.

### Step 3.7 — Evaluation pipeline

**File:** [WalkVLM-LR/test.py](WalkVLM-LR/test.py) (210 lines)

This file implements three of the four paper metrics:
- **ROUGE-1/2/L** ([lines 113–115](WalkVLM-LR/test.py:113)) via `rouge_scorer`.
- **Keyword Density** ([lines 55–84](WalkVLM-LR/test.py:55)) — same CLIP-similarity logic as the keywords reward but used purely for evaluation.
- **GPT Score** ([line 190](WalkVLM-LR/test.py:190)) — calls `evaluate_image` from `GPTScore` (not in this repo; you must wire up a GPT-4 API and provide `appid`, `appkey`, `source`).

TRF (temporal redundancy F1) is **not** in `test.py` — that one is computed externally from EAD predictions vs. ground-truth danger labels.

---

## 4. Tie code blocks back to the paper diagram (30 min)

Open [figures/framework.jpg](figures/framework.jpg) side-by-side with your editor. For each labeled block, point to a file/line.

| Block in Figure 2 | Code location |
|---|---|
| Visual Encoder (ViT) | `model.module.visual` from Qwen2-VL ([inference.py:161](WalkVLM-LR/inference.py:161)) |
| EAD: MultiScaleConv | [EAD.py:5–23](WalkVLM-LR/EAD.py:5) |
| EAD: SAIM (Transformer/Attention) | [EAD.py:31](WalkVLM-LR/EAD.py:31) (`MultiheadAttention`) |
| EAD: MLP → danger A/B/C | [EAD.py:33–46](WalkVLM-LR/EAD.py:33) |
| EAD gate ("trigger LLM?") | [inference.py:181–188](WalkVLM-LR/inference.py:181) (`'C' in predictions`) |
| LLM (Qwen2-VL decoder) | `model.generate(...)` in [inference.py:104](WalkVLM-LR/inference.py:104) |
| GRPO group + advantage | [grpo_trainer.py:522–528](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:522) |
| GRPO KL to reference | [grpo_trainer.py:471](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:471) |
| Reward: Simplicity | [grpo_query_gene.py:506](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:506) |
| Reward: Fluency | [grpo_query_gene.py:196](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:196) |
| Reward: Accuracy (semantic) | [grpo_query_gene.py:264](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:264) |
| Reward: Keywords | [grpo_query_gene.py:371](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:371) |

When you can do this from memory, you understand the system.

---

## 5. Verify your understanding with concrete questions

If you can answer all of these without re-reading, you're done.

**Architecture**
1. Why is sharing the visual encoder between EAD and the VLM efficient? (Hint: paper §"Reducing Temporal Redundancy", and [inference.py:161](WalkVLM-LR/inference.py:161).)
2. How many history frames does EAD consume? Where does that number come from in code, and where in the paper?
3. What is the input dimension of EAD's MLP, and why exactly that number?

**GRPO**
4. What does "group-relative" mean? Why subtract group mean / divide by group std instead of using a learned value function?
5. Where does the KL term enter the loss, and what is β? (Find `self.beta` in `grpo_trainer.py`.)
6. The default reward set in the training script does **not** include `accuracy` or `format`. Why? What does that imply about the `<think>...</think><answer>...</answer>` formatting?

**Rewards**
7. Walk through `fluency_reward` for the string "go forward go forward go forward". Predict whether the reward is high or low and why.
8. The paper's Eq. 4 differs from the code at [grpo_query_gene.py:233](WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py:233). What is the difference?
9. Why is `keywords_reward` likely the training bottleneck? Suggest one cheap optimization.

**Inference / deployment**
10. What is the user-facing latency cost when EAD predicts A (low danger) for all 3 frames? What about when it predicts C?
11. What hard-coded path-format does `load_images_from_folder` expect in [inference.py:22–29](WalkVLM-LR/inference.py:22)?

**Paper claims vs. code**
12. The paper says SAIM is "several layers of stacked transformers." How many does the code actually use? Is that a documentation gap or a real difference?
13. The paper's Accuracy Reward sums cosine similarity and mean token accuracy with equal weight (Eq. 7). What does the code use? Where would you change it?

---

## 6. Suggested follow-ups (optional)

Once the above is solid, here are good next reads ordered by depth:

1. **DeepSeekMath GRPO paper** (Shao et al., 2024, arXiv:2402.03300) — the algorithm WalkVLM-LR fine-tunes with. The citation lives in [grpo_trainer.py:608](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:608).
2. **Qwen2-VL technical report** (Wang et al., 2024, arXiv:2409.12191) — what's actually inside `model.module.visual` and how the language model conditions on patch tokens.
3. **WalkVLM (the predecessor)** (Yuan et al., 2024, arXiv:2412.20903) — the WAD dataset originates here, and the temporal-aware prediction module (TAP) is what EAD improves on.
4. **TRL `GRPOTrainer`** — the upstream class that `Qwen2VLGRPOTrainer` mirrors. Diffing them shows exactly what was added for multimodal inputs.

---

## 7. Known rough edges in this codebase (worth knowing before you run anything)

- Many path strings are blank placeholders: `checkpoint_path` in [inference.py:58](WalkVLM-LR/inference.py:58), `data_dir` in [train_EAD.py:194](WalkVLM-LR/train_EAD.py:194), nearly every `--*` value in [run_grpo_query_gene.sh](WalkVLM-LR/vlm_grpo_template/run_grpo_query_gene.sh), the model paths in [test.py:89, 94](WalkVLM-LR/test.py:89), CLIP path at [test.py:38](WalkVLM-LR/test.py:38). Expect to fill these in.
- `test.py` imports `from GPTScore import evaluate_image` — that module is not in the repo; bring your own GPT-4 wrapper.
- `train_EAD.py` defines `MLPModel` and `VisualDangerClassificationHead` that `main()` does not use — likely experimental leftovers.
- Reward weights are hard-coded to `[1,1,1,1]` in [grpo_trainer.py:516](WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py:516); the paper doesn't tune them.
- The training script computes `accuracy` only as a fallback registry entry (`accuracy_reward`) — the *paper's* "Accuracy Reward" is implemented under the name `semantic_similarity_reward` / registry key `sematic`. The string-matching `accuracy_reward` is for RAVEN-style symbolic answers, not walking guidance.
- EAD feature-extraction step (running Qwen2-VL's ViT over WAD videos and dumping `{video}_features.json`) is referenced by `train_EAD.py` but not provided. You write it.

These aren't bugs; they're the typical "research code" gaps. Knowing about them up front saves hours.

---

## Time budget summary

| Stage | Time |
|---|---|
| 0. Orient | 10 min |
| 1. Paper, three passes | 60–90 min |
| 2. Repo map | 10 min |
| 3. Code in dependency order | 3–4 hours |
| 4. Diagram-to-code mapping | 30 min |
| 5. Self-check questions | 30 min |
| **Total** | **~5–7 hours** for solid understanding |
