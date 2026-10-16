"""
Ablation study on TinyLlama-1.1B (LLaMA-2 architecture: 22 layers x 32 heads,
grouped-query attention, SwiGLU MLP, RMSNorm, RoPE).

Experiments
  E0  baseline
  E1  single-component sweeps: for every layer, (a) skip the whole layer,
      (b) zero its attention block, (c) zero its MLP
  E2  progressive attention-head pruning (0-90 % of all 704 heads) in three orders:
      random, least-important-first, most-important-first (gradient importance)
  E3  progressive layer removal (0-16 of 22 layers) in four orders:
      least-important-first (from E1), random, top-down, bottom-up
  E4  input-embedding corruption: Gaussian noise and token-embedding masking
  E5  qualitative: greedy generations under increasing ablation

Results are cached in ../results/part2/results.json, so an interrupted run resumes.
    python run_ablation.py
"""
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ablation import Ablator, use_fp32_compute
from tasks import TaskSuite, prepare_data

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "part2")
os.makedirs(RES, exist_ok=True)
RES_FILE = os.path.join(RES, "results.json")
MODEL = "TinyLlama/TinyLlama_v1.1"

# evaluation-set sizes (kept modest so ~150 configurations run in ~2 h on a GTX 1650)
EVAL_SIZES = {"ppl": 24, "blimp": 240, "sst": 200, "sst_shots": 4}
HEAD_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]
LAYER_KS = [0, 1, 2, 3, 4, 6, 8, 11, 16]
N_RANDOM_HEAD_ORDERS, N_RANDOM_LAYER_ORDERS = 2, 2
NOISE = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
MASK = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]


def load_results():
    if os.path.exists(RES_FILE):
        with open(RES_FILE) as f:
            return json.load(f)
    return {}


def save_results(r):
    tmp = RES_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(r, f, indent=1)
    os.replace(tmp, RES_FILE)


def main():
    if os.name == "nt":   # keep Windows from sleeping during the long run (released on exit)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    model.requires_grad_(False)
    use_fp32_compute(model)      # fp16 storage, fp32 arithmetic (much faster on GTX 16xx)
    abl = Ablator(model)
    data = prepare_data(tok)
    suite = TaskSuite(tok, data, max_items=EVAL_SIZES)
    L, H = abl.n_layers, abl.n_heads
    print(f"{MODEL}: {L} layers x {H} heads, head_dim {abl.head_dim}")

    results = load_results()

    def run(key, **ablation):
        if key in results:
            return results[key]
        if "baseline" in results and not any(v for v in ablation.values()):
            results[key] = dict(results["baseline"], spec=ablation)   # nothing ablated = baseline
            save_results(results)
            return results[key]
        t = time.time()
        with abl.apply(**ablation):
            r = suite.evaluate(model)
        r["spec"] = {k: (list(map(list, v)) if k == "heads" else list(v) if isinstance(v, (list, tuple, set)) else v)
                     for k, v in ablation.items()}
        results[key] = r
        save_results(results)
        print(f"{key:40s} ppl={r['ppl']:9.2f} blimp={r['blimp']:.3f} sst={r['sst']:.3f} "
              f"facts={r['facts']:.3f}  ({time.time() - t:.1f}s)", flush=True)
        return r

    # E0
    base = run("baseline")

    # E1 single-layer sweeps
    for l in range(L):
        run(f"E1/skip/{l}", skip_layers=[l])
        run(f"E1/attn/{l}", attn_layers=[l])
        run(f"E1/mlp/{l}", mlp_layers=[l])

    # E2 head pruning
    imp_file = os.path.join(RES, "head_importance.pt")
    if os.path.exists(imp_file):
        imp = torch.load(imp_file)
    else:
        abl.reset()
        imp = suite.head_importance(model, abl)
        torch.save(imp, imp_file)
    all_heads = [(l, h) for l in range(L) for h in range(H)]
    by_imp = sorted(all_heads, key=lambda lh: imp[lh].item())
    orders = {"least_important": by_imp, "most_important": by_imp[::-1]}
    for s in range(N_RANDOM_HEAD_ORDERS):
        rnd = all_heads[:]
        random.Random(s).shuffle(rnd)
        orders[f"random{s}"] = rnd
    for name, order in orders.items():
        for frac in HEAD_FRACS:
            n = int(round(frac * len(all_heads)))
            run(f"E2/{name}/{frac}", heads=order[:n])

    # E3 progressive layer removal
    skip_ppl = [results[f"E1/skip/{l}"]["ppl"] for l in range(L)]
    layer_orders = {
        "least_important": sorted(range(L), key=lambda l: skip_ppl[l]),
        "top_down": list(range(L - 1, -1, -1)),
        "bottom_up": list(range(L)),
    }
    for s in range(N_RANDOM_LAYER_ORDERS):
        rnd = list(range(L))
        random.Random(100 + s).shuffle(rnd)
        layer_orders[f"random{s}"] = rnd
    with open(os.path.join(RES, "layer_orders.json"), "w") as f:
        json.dump(layer_orders, f)
    for name, order in layer_orders.items():
        for k in LAYER_KS:
            run(f"E3/{name}/{k}", skip_layers=order[:k])

    # E4 input embeddings
    for s in NOISE:
        run(f"E4/noise/{s}", embed_noise=s)
    for p in MASK:
        run(f"E4/mask/{p}", embed_mask=p)

    # E5 qualitative generations
    gen_file = os.path.join(RES, "generations.json")
    if not os.path.exists(gen_file):
        prompts = ["The capital of France is", "Once upon a time, there was a little girl who"]
        settings = {
            "intact": {},
            "50% heads (least important)": {"heads": by_imp[:352]},
            "50% heads (random)": {"heads": orders["random0"][:352]},
            "skip 4 least important layers": {"skip_layers": layer_orders["least_important"][:4]},
            "skip layer 0": {"skip_layers": [0]},
            "zero MLP of layer 0": {"mlp_layers": [0]},
            "skip 8 least important layers": {"skip_layers": layer_orders["least_important"][:8]},
            "embedding noise sigma=1": {"embed_noise": 1.0},
        }
        gens = {}
        for name, spec in settings.items():
            gens[name] = {}
            for p in prompts:
                with abl.apply(**spec):
                    ids = tok(p, return_tensors="pt").to("cuda")
                    out = model.generate(**ids, max_new_tokens=30, do_sample=False,
                                         pad_token_id=tok.eos_token_id, repetition_penalty=1.1)
                gens[name][p] = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            print(name, "->", gens[name][prompts[0]])
        with open(gen_file, "w") as f:
            json.dump(gens, f, indent=1)
    print("all experiments done")


if __name__ == "__main__":
    main()
