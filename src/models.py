"""Model registry. w = weight memory (GB); need = minimum GB to hold for an engine (weights + KV + overhead);
margin/misc = activation margin and non-torch overhead in GB (defaults 1.5/0.8; tighter on small idle slices).
bf16 for <=1.5B models; AWQ-INT4 for all >=3B models (GPU memory on the shared host; this also makes the 3B and
7-9B cross-family comparisons precision-matched). qwen2.5-7b-bf16 is a precision control for Qwen2.5-7B."""
MODELS = {
    "qwen2.5-0.5b": dict(hf="Qwen/Qwen2.5-0.5B-Instruct", params=0.5, family="qwen2.5", w=1.0, need=4),
    "qwen2.5-1.5b": dict(hf="Qwen/Qwen2.5-1.5B-Instruct", params=1.5, family="qwen2.5", w=3.1, need=4.9,
                          margin=0.5, misc=0.6),
    "qwen2.5-3b":   dict(hf="Qwen/Qwen2.5-3B-Instruct-AWQ", params=3.1, family="qwen2.5", w=2.2, need=3.9,
                          margin=0.4, misc=0.5),
    "qwen2.5-7b":   dict(hf="Qwen/Qwen2.5-7B-Instruct-AWQ", params=7.6, family="qwen2.5", w=5.6, need=9),
    "qwen2.5-14b":  dict(hf="Qwen/Qwen2.5-14B-Instruct-AWQ", params=14.8, family="qwen2.5", w=10.0, need=14),
    "llama3.2-1b":  dict(hf="unsloth/Llama-3.2-1B-Instruct", params=1.2, family="llama3", w=2.5, need=5.5),
    "llama3.2-3b":  dict(hf="AMead10/Llama-3.2-3B-Instruct-AWQ", params=3.2, family="llama3", w=2.2, need=3.9,
                          margin=0.4, misc=0.5),
    "llama3.1-8b":  dict(hf="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", params=8.0, family="llama3", w=5.8,
                         need=9.5),
    "mistral-7b":   dict(hf="solidrust/Mistral-7B-Instruct-v0.3-AWQ", params=7.2, family="mistral", w=4.2, need=8),
    "gemma2-9b":    dict(hf="hugging-quants/gemma-2-9b-it-AWQ-INT4", params=9.2, family="gemma2", w=6.5, need=10.5),
    # seed replicate: same checkpoint, different sampling seeds (noise floor for curve-shape comparisons)
    "qwen2.5-0.5b-seed2": dict(hf="Qwen/Qwen2.5-0.5B-Instruct", params=0.5, family="qwen2.5", w=1.0, need=3.5,
                               seed=1042, control=True),
    "qwen2.5-7b-bf16": dict(hf="Qwen/Qwen2.5-7B-Instruct", params=7.6, family="qwen2.5", w=15.3, need=19,
                            control=True),
}
