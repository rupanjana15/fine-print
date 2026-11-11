# Concepts guide

Plain-language explanations of every idea this project uses, in the order you meet them.
Each section says **what it is**, **why we need it here**, and **where it lives in the repo**.

---

## Part 1 — Language models

### LLM, parameters, "3B"
A large language model predicts the next token (word piece) given the previous ones. Its
**parameters** (weights) are the numbers learned during training; "3B" = ~3 billion of them.
More parameters → more knowledge and skill, but more memory and slower.
At 16-bit precision each parameter takes 2 bytes, so a 3B model ≈ **6 GB**; at 4-bit ≈ **1.7–2 GB**.

### Tokens
Text is split into tokens (~¾ of an English word each). Models have a maximum number of tokens they
can read at once (context length). A 200-token chunk ≈ 150 words ≈ one or two paragraphs.

### Base vs instruct model
A **base** model just continues text. An **instruct** model was further trained to follow instructions
in a chat format. We start from an instruct model so it already understands "here's a task, answer it".

### Chat template
Instruct models expect conversations in a specific format with special tokens marking who is speaking
(system / user / assistant). Training data must use the **same template** the model was built with,
or it learns garbage. Our JSONL stores `messages` and the tokenizer applies the template.
*Repo:* `src/fineprint/data/format.py` (v0.2), training notebook (v0.4).

---

## Part 2 — Fine-tuning

### Fine-tuning
Continuing to train a pre-trained model on a small, task-specific dataset so it gets good at *one*
thing. We teach it: legal chunk in → JSON flags out.

### Supervised fine-tuning (SFT)
Fine-tuning on (input, correct output) pairs. The model is shown the input and learns to produce the
output, one token at a time. It's what we do: input = chunk, output = the JSON written from ToS;DR labels.

### Loss and completion-only loss
**Loss** measures how wrong the model's predictions are; training adjusts weights to reduce it.
By default the loss covers every token, including the prompt. **Completion-only loss** masks the
system and user tokens so only the assistant's answer counts. Then the model learns to *answer*,
not to memorise legal text. *Repo:* `configs/train.yaml` → `completion_only_loss: true`.

### Epoch, learning rate, batch size
- **Epoch:** one pass over all training data. We use ~2; more risks memorising (overfitting).
- **Learning rate:** how big each weight update is. Too high → unstable; too low → learns nothing.
  2e-4 is a standard LoRA starting point.
- **Batch size:** examples per update. "Effective batch 16" = 4 per step × 4 **gradient accumulation**
  steps (adds up gradients over several small batches to simulate a bigger one on limited memory).

### Overfitting
The model memorises training examples instead of learning the general pattern: great on training data,
worse on new data. Detected by watching **validation** loss/metrics; prevented by fewer epochs, more
data, and a proper split.

### LoRA (Low-Rank Adaptation)
Instead of changing the huge weight matrices, LoRA adds two small matrices (A and B) beside each one
and trains only those. Their product is a low-"rank" update to the original weights. With rank 16 on a
3B model, you train ~1% of the parameters. Result: far less memory, and the adapter file is tiny (tens of MB).
- **Rank (r):** size of the adapter; higher = more capacity, more memory.
- **Alpha (α):** scales the adapter's effect (we use α = 2r).
- **Target modules:** which layers get adapters; we use all linear layers (attention + MLP).

### Quantization
Storing weights with fewer bits (16 → 8 → 4). 4-bit cuts memory ~4× with a small quality loss.

### QLoRA
LoRA on top of a **4-bit quantized**, frozen base model. The base takes ~2 GB instead of ~6 GB, which
is why a 3B model trains on a free 16 GB T4 GPU. *Decision:* D04.

### Unsloth / TRL / PEFT
- **TRL** (Hugging Face): provides `SFTTrainer`, the training loop for supervised fine-tuning.
- **PEFT**: the library implementing LoRA adapters.
- **Unsloth**: optimised kernels that make LoRA/QLoRA ~2× faster with less memory, which is ideal for free GPUs.

### Merging the adapter
After training you can add the LoRA update into the base weights permanently ("merge"), producing one
normal model file, which is needed before converting to GGUF.

---

## Part 3 — Data

### Labels, positives, negatives
- A **label** is the correct answer for an example.
- **Positive:** a chunk that should be flagged. **Negative:** a harmless chunk → `{"flags": []}`.
- **Hard negative:** a negative that *looks* like a positive (e.g. a clause DocBot thought was about
  data selling, but a human said no). These teach fine distinctions. *Decision:* D07, D13.

### Taxonomy
The set of categories the model can output. We merge hundreds of ToS;DR cases into ~20–25 so each
category has enough examples. *Repo:* `configs/taxonomy.yaml`. *Decision:* D12.

### Class imbalance
Some categories have 600 examples, others 20. Models favour common classes. Handled by merging
categories, possibly up/down-sampling, and measuring with **macro**-F1.

### Train / validation / test split
- **Train:** what the model learns from.
- **Validation:** checked during development to choose settings (it indirectly influences the model).
- **Test:** touched only for final numbers, so it stays an honest estimate of real-world performance.

### Data leakage
When information from the test set sneaks into training, scores look better than reality. Here the
risk is boilerplate: the same company reuses near-identical clauses across documents. A **random**
split would put twins in train and test. A **group split by service** prevents it. *Decision:* D14.

### Label noise
Some training labels are wrong. Here the main source is **missing labels**: volunteers don't annotate
every sentence, so a risky clause can end up in a "harmless" chunk. Noise can't be fully removed, so you
**reduce** it (negatives only from fully reviewed services, D26), **measure** it (manually review a sample),
and **account for it** when reading metrics (precision on noisy test data is a lower bound).

### TF-IDF and cosine similarity
**TF-IDF** turns text into a vector of word (and word-pair) weights: a word counts more if it's frequent in
this text but rare across all texts, so "arbitration" matters and "the" doesn't. **Cosine similarity**
compares two such vectors: 1 = same words in the same proportions, 0 = nothing in common. It's a cheap,
model-free way to ask "does this sentence look like a known risky clause?". We use it to throw out
negatives that are probably just unlabelled positives. *Repo:* `max_quote_similarity` in
`build_dataset.py`. *Decision:* D31.

### Chunks, windows and overlap
A long document is cut into **chunks** the model can read (~200 tokens), on sentence boundaries.
Consecutive chunks **overlap** a little (~30 tokens) so a clause on a border appears whole in at least one.
For training positives we build a **window** around a known quote instead, so the labelled clause is
never cut. *Repo:* `src/fineprint/chunk.py` (shared with inference). *Decision:* D25.

### Train/serve skew
When the data a model sees in production differs from its training data (different prompt, different
chunking, different cleaning), performance drops silently. Prevented by sharing the same code for both:
`chunk.py`, `prompt.py`, `clean_text`. *Decisions:* D25, D27.

### Group split with a stable hash
Instead of shuffling, each company name is hashed (MD5) to a number that decides train/val/test. The same
company always gets the same split, even when the dataset is rebuilt with more data, so results stay
comparable over time. *Decision:* D29.

### Rate limiting, caching, backoff
APIs limit requests per time window (HTTP **429** = "too many requests").
- **Throttling:** wait between requests (2 s).
- **Exponential backoff:** after a 429, wait 30 s, then 60 s, 120 s… so you don't hammer the server.
- **Caching:** save every response to disk; re-runs skip what's done, so downloads are **resumable**.
*Repo:* `src/fineprint/data/fetch_tosdr.py`. *Decision:* D08.

### Data licence (CC BY-SA)
Creative Commons Attribution-ShareAlike: you may use and adapt the data, but must credit the source,
and anything you publish derived from it must use the same licence. *Doc:* `DATA.md`.

---

## Part 4 — Evaluation

### Baseline
A simpler system you compare against. A score only means something relative to baselines. *Decision:* D16.

### Zero-shot / few-shot
Using the base model with only instructions (**zero-shot**) or instructions plus a handful of worked
examples in the prompt (**few-shot**), with no training. If few-shot is as good as fine-tuning,
fine-tuning wasn't worth it.

### DeBERTa (encoder classifier)
A smaller model (~140M parameters for "small") that reads text and outputs label probabilities. It can't
write, but it's fast and strong at classification, which makes it the toughest baseline for *detection*.

### Reference row ("always flag")
A deliberately dumb system included to calibrate the metrics. When 65% of test chunks are risky, a system
that flags *everything* gets recall 1.0 and F1 ≈ 0.79 without reading anything. Any real system must
clearly beat that, especially on precision and on how often it flags harmless text. *Decision:* D32.

### Logistic regression, one-vs-rest
A linear classifier: it gives each TF-IDF feature a weight and turns the weighted sum into a probability
(0–1). **One-vs-rest** trains one such classifier per category ("is this about arbitration: yes/no?"), so a
chunk can get several categories at once (**multi-label**). Fast, strong, and easy to inspect.
*Repo:* `src/fineprint/baselines/tfidf.py`. *Decision:* D33.

### Decision threshold, tuned on validation
A classifier outputs probabilities; you still choose the cut-off ("flag if ≥ 0.2"). Lower → more flags
(higher recall, lower precision). We pick the threshold that maximises macro-F1 on **val**, never on test,
because choosing anything by looking at test makes the test score optimistic.

### Greedy decoding
At each step the LLM picks the single most likely next token, with no randomness. The same input always
gives the same output, which makes evaluations repeatable. *Decision:* D34.

### Precision, recall, F1
For "is this chunk risky?":
- **Precision** = of the chunks we flagged, how many really were risky? (low → crying wolf)
- **Recall** = of the truly risky chunks, how many did we flag? (low → missing risks)
- **F1** = harmonic mean of the two; high only if both are high.

### Macro vs micro F1
- **Micro**: pool all decisions, then compute F1, so common classes dominate.
- **Macro**: compute F1 per class, then average, so every class counts equally. We report macro so failing
  on rare-but-important categories (e.g. arbitration) is visible.

### Confusion matrix
A table of true label vs predicted label. Shows *which* mistakes happen (e.g. "blocker" predicted as "bad").

### Hallucination & quote faithfulness
LLMs can produce confident text that isn't supported by the input. Because every flag must include a
verbatim quote, we can check `quote in chunk`: the share that passes is **quote faithfulness**. *Decision:* D11.

### Error analysis
Reading actual mistakes and grouping them by cause (rare class, ambiguous label, chunk cut a clause in
half…). It tells you *what to fix next*, which aggregate numbers can't.

---

## Part 5 — Serving

### fp16 vs 4-bit, GPU batching
**fp16** stores each weight in 16 bits (the model's normal quality, ~6 GB for 3B). The 4-bit GGUF (~2 GB)
is for running on small machines. On a GPU we run fp16 and process ~8 chunks at once (**batching**),
which uses the GPU's parallelism, so it's far faster than one chunk at a time on the Mac. *Decision:* D35.

### Why training needs fp32 (or mixed precision)
fp16 can only hold numbers down to ~6e-8 and has ~3 significant digits. Inference is fine in fp16, but
**training** adds tiny updates to weights and divides by small numbers (AdamW's ε = 1e-8, which becomes 0 in
fp16). Result: NaN after one step. So training keeps weights in **fp32**, or uses **mixed precision** (fp32
master copy, fp16 maths, a loss scaler) — and QLoRA keeps its trainable adapters in higher precision for the
same reason. *Repo:* `baselines/deberta.py`. *Decision:* D36.

### Parallel jobs on two GPUs
Kaggle gives two T4 GPUs. Rather than splitting one model across both (complex), we run **two independent
job queues**, one per GPU (`CUDA_VISIBLE_DEVICES=0` / `=1`). Total time ≈ the longer queue, not the sum.
*Repo:* `notebooks/kaggle_baselines.ipynb`. *Decision:* D35.

### GGUF and llama.cpp
**llama.cpp** is a C/C++ engine for running LLMs efficiently on CPUs and Apple Silicon.
**GGUF** is its model file format. **Q4_K_M** is a 4-bit quantization scheme with a good
quality/size balance (~2 GB for 3B). *Decision:* D18.

### Inference pipeline
Everything that happens at prediction time: clean text → split into chunks → run the model on each →
validate the JSON and quotes → merge duplicate flags across chunks → compute a grade → show in the UI.

### Latency
Time to get an answer. On CPU, a 3B model generates roughly a few to ~20 tokens/second depending on
hardware, so we keep outputs short and skip chunks that are obviously harmless.

---

## Part 6 — How the whole thing works, end to end

1. **Collect** (`fetch_lexglue.py`, `fetch_tosdr.py`): download human-labelled clauses and documents, cached.
2. **Inspect** (`stats.py`): count examples per case/severity, read samples, find data problems.
3. **Build the dataset** (v0.2): clean HTML, keep approved English points, map cases → categories,
   add negatives (including hard negatives), split by service, write chat-format JSONL.
4. **Baselines** (v0.3): keyword rules, base model zero/few-shot, DeBERTa → scored by one harness.
5. **Fine-tune** (v0.4): QLoRA on Kaggle; checkpoints to HF Hub.
6. **Evaluate and iterate** (v0.5): metrics table + error analysis → fix the biggest problem → retrain.
7. **Package** (v0.6): merge adapter → GGUF 4-bit → inference pipeline on the Mac.
8. **Demo and write up** (v0.7–v1.0): Gradio app on HF Spaces, README with results, model card.

Each step reads files written by the previous one, so any step can be re-run on its own.
