# Committee report 4: open-weights LLM and serving stack for grounded yes/no + citation (verified 2026-09-17)

## 0. Budget reality check

Prompt is about 1.2-1.5k tokens (system + ~600-word transcript with segment IDs + question). Output per question is 30-60 tokens if you emit only `{"answer","segments"}` (+ an optional copied quote). Ten questions run as **ten parallel requests sharing one cached prefix** decode in one batch at roughly batch-1 speed. Even a 27B dense model at ~40 tok/s on an H100 finishes all ten in 2-3 s. The budget is only at risk if you let **thinking mode** run (5.5x output tokens for Qwen-3.6-27B per [OptimalThinkingBench-style measurements](https://arxiv.org/pdf/2508.13141); hundreds of seconds TTFT reported). Rule: thinking off, or hard-capped.

## 1. Candidate models (verified Sept 2026)

| Model | Params | License | Weights VRAM | Batch-1 decode (measured / bandwidth-bound est.) | Thinking toggle |
|---|---|---|---|---|---|
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Aug 14 2026) | 27B dense, VL | Apache-2.0 | 56 GB BF16 / 31 GB FP8 / ~15 GB W4 | H100 ~40-50 tok/s (est.); 3090 W4A16 114-124 tok/s only with heavy patches ([syv-ai](https://github.com/syv-ai/qwen38-27b-rtx3090)) | `enable_thinking=False`; `reasoning_effort` low/medium/xhigh (default xhigh) |
| [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (Apr 2026) | 27B dense | Apache-2.0 | as above | as above | `enable_thinking=False` (on by default) |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 35B / 3B active | Apache-2.0 | 70 GB BF16, ~19 GB Q4 | 4090 Q4 llama.cpp 120-196 tok/s (Qwen3-30B-A3B, [runaihome](https://runaihome.com/blog/qwen3-30b-a3b-local-ai-guide-2026/)) | `enable_thinking=False` |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) / 4B (Mar 2026) | 9B dense | Apache-2.0 | 18 GB BF16 | H100 ~150 tok/s (8B class, [Azure](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/performance-of-llama-3-1-8b-ai-inference-using-vllm-on-nd-h100-v5/4448355)); 4090 ~50-60 BF16 | same |
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-E4B) (Apr 2 2026) | 31B dense | Apache-2.0 | 62 GB BF16, ~18 GB INT4 | H100 vLLM ~40 tok/s @c=1, TTFT 279 ms @ISL 128 ([inferencebench](https://inferencebench.io/blog/gemma-4-31b-h100-complete-inference-benchmark/)); 4090 vLLM reported poor ([markaicode](https://markaicode.com/benchmarks/cuda-gemma-4-rtx-4090-latency-benchmark/), low trust) | think token in system prompt |
| Gemma 4 26B-A4B | 26B / ~4B | Apache-2.0 | 52 GB BF16 | 4090 reports ~11 tok/s vLLM vs 150-193 Ollama; contradictory, unverified | same |
| [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | 21B / 3.6B | Apache-2.0 | 16 GB MXFP4 | H100 228 tok/s, L4 62 tok/s @c=1, 10k ctx ([devforth](https://devforth.io/insights/self-hosted-gpt-real-response-time-token-throughput-and-cost-on-l4-l40s-and-h100-for-gpt-oss-20b/)) | `Reasoning: low/medium/high`; always emits some CoT; harmony format mandatory |
| gpt-oss-120b | 117B / 5.1B | Apache-2.0 | 65 GB MXFP4 (80 GB card only) | fast MoE; not measured here | same |
| [GLM-4.7-Flash](https://unsloth.ai/docs/models/tutorials/glm-4.7-flash) (Jan 19 2026) | 31B / 3B | MIT | ~24 GB Q4 / 64 GB BF16 | MTP spec-decode in vLLM | yes |
| [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) (Dec 2025) | 30B / 3.5B, Mamba-hybrid | NVIDIA Open Model License | 60 GB BF16 | NVIDIA claims 2.2-3.3x gpt-oss-20b/Qwen3-30B throughput; no absolute numbers | `enable_thinking=False` |
| Ministral 3 14B (Dec 2025) | 14B dense | Apache-2.0 | 28 GB BF16 | ~90 tok/s H100 (est.) | reasoning variant separate |
| [Phi-4-reasoning-vision-15B](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B) (Mar 2026) | 15B | MIT | 30 GB | - | selective reasoning; 16k ctx. **Phi-5: no HF card found; treat "Phi-5" articles as unverified.** |

Not worth it here: [Mistral Small 4](https://mistral.ai/news/mistral-small-4/) (119B-A6B, Apache, needs 4xH100 per Mistral), [Llama 4 Scout](https://apxml.com/posts/llama-4-system-requirements) (109B, ~61 GB Q4, community license, weak per-parameter), [GLM-5.3 / 5.3-Flash](https://runaihome.com/blog/glm-5-3-open-weights-live-hardware-guide-2026/) (744B / 320B, no single-GPU fit), DeepSeek-R1-0528-Qwen3-8B (reasoning-only; long CoT). Note that Qwen 3.7 was never open-weighted.

**Speed caveat:** batch-1 decode is memory-bandwidth-bound (bandwidth / weight bytes: H100 3.35 TB/s, A100 2.0, 4090 1.0). Claims such as "Llama-3.3-70B FP8 at 120-130 tok/s batch-1 on one H100" ([Spheron](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)) exceed that bound by ~2.5x and should be discounted; ~40 tok/s is realistic for 70B-FP8. Several "benchmark" blogs (markaicode) have retracted their own numbers. Measure on your box with `vllm bench serve`.

## 2. Recommended primary stack

**Model:** Qwen3.6-27B or Qwen3.8-27B, `enable_thinking=False` (FP8 on >=40 GB; AWQ/W4A16 on a 24 GB card). On a 4090 use Qwen3.6-35B-A3B or GLM-4.7-Flash at Q4/FP8 instead. Dense 27B is preferred for near-miss discrimination; MoE-3B-active models are faster but weaker on fine-grained reading.

**Serving:** vLLM (V1, prefix caching on by default; [prefix caching docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)). Put system prompt + transcript first, question last; send ten requests concurrently. The transcript prefill runs once; the ten decodes batch. SGLang is equivalent (RadixAttention, slightly faster constrained decoding); TensorRT-LLM buys ~10% latency for a 30-min engine build, not worth it. llama.cpp is fine for the 24 GB fallback but its prefix sharing across parallel slots is weaker.

**Structured output:** vLLM `structured_outputs` with a JSON schema ([docs](https://docs.vllm.ai/en/latest/features/structured_outputs/)), xgrammar backend: 20-50 ms one-off compile, <40 us/token mask ([XGrammar](https://arxiv.org/pdf/2411.15100)). Constrain `segments` to the integer range actually present. Schema:

```json
{"quote": "<verbatim text of the most relevant segment>",
 "answer": "yes" | "no",
 "segments": [int, ...]}
```

`quote` before `answer` is a cheap, bounded "look before you decide" step that forces the model to re-read the number/duration; it costs ~20 tokens. Empty `segments` when `answer=="no"`.

**Ten-in-one vs ten parallel:** the closest prior work ([multi-question yes/no on call transcripts + utterance indices](https://arxiv.org/html/2509.21732)) shows GPT-4o judgment accuracy 0.90 at 10 questions/prompt, falling to 0.85 at 50, and utterance-selection F1 0.71 to 0.63; instruct-tuned 7-8B models had 31-73% JSON failures when batched. With prefix caching, parallel costs nothing extra and avoids "autoregressive inertia" and numbering errors. Go parallel.

## 3. Prompt design

- **Numbered segments, not timestamps.** Render `[S07] DOCTOR: Take 200 mg twice daily.` and have the model return IDs; map to seconds yourself. This is the LongCite `<C_i>` recipe ([LongCite](https://arxiv.org/html/2409.02897v3)), the candidate-ID selection that beat direct timestamp generation in [topic-to-timestamp alignment](https://arxiv.org/pdf/2606.20890), and the utterance-index scheme in the transcript paper above. Inline seconds add numeric noise that competes with dosages.
- **Rewrite tag questions to declaratives** before prompting: "The lipid profile came back normal, didn't it?" becomes "Claim: the lipid profile came back normal." A 45-model study finds tag suffixes shift yes/no answers by up to +/-32 points and that resistance is grammar-keyed; stating the claim as a bare assertion removes it ([tag-question study](https://arxiv.org/html/2607.23976)).
- Instruct explicitly: "Answer yes only if every detail (number, unit, duration, timing, body part) matches the transcript; ASR may mangle numbers; treat '2 hundred' / '200' / 'two-hundred' as equal, but not 100 vs 200."
- Normalise numbers in the transcript (spell-out to digits) in preprocessing; keep original for quote matching.

## 4. Use of the 390 labelled questions

Use them first for prompt/model selection with a held-out split (39 conversations, ~8-fold by conversation). LoRA is plausible but marginal: the transcript paper's fine-tuned Qwen2.5-7B/Llama-3.1-8B beat GPT-4o on batched yes/no, but its gains were mostly JSON-format and index-format fixes, which constrained decoding gives you for free; [ClaimIQ](https://arxiv.org/abs/2509.11492) (LoRA on numerical claims) reported strong validation but a "notable drop" on test. [PEARC'25](https://dl.acm.org/doi/10.1145/3708035.3736091) finds LoRA competitive at low k and ICL underperforming. If you LoRA, r=8-16, 2-3 epochs, add synthetic hard negatives by perturbing numbers/durations in your own "yes" items; expect a few points, not a step change.

## 5. Cheap fallback (and why it is only a fallback)

Pipeline: BM25 + a small embedding/ColBERT reranker to pick top-3 segments per claim, then `cross-encoder/nli-deberta-v3-large` or [ModernCE-large-nli](https://huggingface.co/dleemiller/ModernCE-large-nli) (395M, 8k ctx) entailment on (segment, claim); yes if max entailment > tau; cite argmax segments. Runs in <1 s on any GPU. **But** NLI cross-encoders are documented as "vulnerable to numerical perturbations" ([CheckThat! 2025 numerical task](https://arxiv.org/html/2507.06195): ModernBERT NLI macro-F1 0.52 test, evidence quality the bottleneck), and NLI-based pipelines only matched LLM judges on entity/date perturbations when paired with a 70B LLM ([answer-perturbation study](https://arxiv.org/html/2609.15561)). Mitigate with a deterministic numeric check: extract numbers/units/durations from claim and candidate segment; if the claim's numbers are absent from the segment, force "no". That hybrid (retriever + NLI + regex numerics) is the credible fallback; retriever + LLM verifier is the primary. Extractive QA (deberta-v3-squad2) is a poor fit: it answers "what dose?" but not "is 200 mg stated?".

## 6. Pitfalls

- Thinking on by default in Qwen3.5+/Gemma 4/Nemotron; forgetting to disable it blows the budget.
- gpt-oss models cannot fully suppress CoT; "low" still reasons and needs the harmony template.
- Prefix cache misses if any byte of the prefix differs (put the question strictly after the transcript; sampling params do not affect the hash).
- vLLM on RTX 4090 (Ada) with Gemma 4 falls back to Triton attention and is slow; Qwen/GLM/gpt-oss paths are better tested.
- Segment IDs must be constrained (enum/regex) or the model will invent IDs; validate the cited segment contains the quoted numbers before trusting a "yes".
- Temporal IoU is 60% of the score: prefer citing the one exact segment; over-citing neighbours dilutes IoU.

**Uncertainty flags:** absolute tok/s for 27B-class on A100/4090 with stock vLLM are estimates (bandwidth bound), not measurements; Gemma 4 26B-A4B consumer-GPU numbers conflict across sources; Phi-5 existence unverified; the numeric-robustness gap between a 27B instruct model and NLI cross-encoders is inferred from adjacent benchmarks, not measured on this task; the 390-question split is the arbiter.
