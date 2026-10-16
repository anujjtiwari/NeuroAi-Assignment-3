"""
Downstream tasks used to measure the effect of ablations.

  1. Next-word prediction  WikiText-2 (test) perplexity on fixed 256-token chunks.
  2. Syntax                BLiMP minimal pairs (4 agreement/morphology paradigms):
                           correct if log P(grammatical) > log P(ungrammatical).
  3. Sentiment             SST-2 (validation), 4-shot prompt; the label is the one
                           of " positive" / " negative" with the higher next-token logit.
  4. Factual completion    hand-written prompts ("The capital of France is") after a
                           fixed 3-fact cloze prefix -> top-1 accuracy of the first
                           answer token.

The data are downloaded once from the Hugging Face Hub and cached in ./data.
"""
import json
import os
import random

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

FACT_PREFIX = "The capital of Peru is Lima. The chemical symbol for iron is Fe. A baby sheep is called a lamb. "

BLIMP_PARADIGMS = [
    "regular_plural_subject_verb_agreement_1",   # subject-verb agreement
    "determiner_noun_agreement_2",               # this/these + noun number
    "anaphor_number_agreement",                  # himself / themselves
    "irregular_past_participle_verbs",           # morphology (was eaten / was ate)
]

FACTS = [
    ("The capital of France is", ["Paris"]), ("The capital of Japan is", ["Tokyo"]),
    ("The capital of Italy is", ["Rome"]), ("The capital of Germany is", ["Berlin"]),
    ("The capital of Spain is", ["Madrid"]), ("The capital of Russia is", ["Moscow"]),
    ("The capital of China is", ["Beijing"]), ("The capital of Egypt is", ["Cairo"]),
    ("The capital of England is", ["London"]), ("The capital of Greece is", ["Athens"]),
    ("The capital of Austria is", ["Vienna"]), ("The capital of Ireland is", ["Dublin"]),
    ("The capital of Norway is", ["Oslo"]), ("The capital of Portugal is", ["Lisbon"]),
    ("The capital of Poland is", ["Warsaw"]), ("The capital of Thailand is", ["Bangkok"]),
    ("The Eiffel Tower is located in the city of", ["Paris"]),
    ("The Great Wall is located in", ["China"]), ("The Taj Mahal is located in", ["India", "Agra"]),
    ("The Statue of Liberty is in New", ["York"]),
    ("The largest planet in the solar system is", ["Jupiter"]),
    ("The planet known as the Red Planet is", ["Mars"]),
    ("The Earth revolves around the", ["Sun", "sun"]),
    ("The tallest mountain in the world is Mount", ["Everest"]),
    ("The longest river in Africa is the", ["Nile"]),
    ("Water is made of hydrogen and", ["oxygen"]),
    ("The chemical symbol for gold is", ["Au"]),
    ("The theory of relativity was developed by Albert", ["Einstein"]),
    ("Romeo and Juliet was written by William", ["Shakespeare"]),
    ("The first president of the United States was George", ["Washington"]),
    ("The Mona Lisa was painted by Leonardo da", ["Vinci"]),
    ("Microsoft was founded by Bill", ["Gates"]),
    ("Apple was co-founded by Steve", ["Jobs", "Wozniak"]),
    ("The Harry Potter books were written by J. K.", ["Rowling"]),
    ("The currency of Japan is the", ["yen", "Yen", "Japanese"]),
    ("The language spoken in Brazil is", ["Portuguese"]),
    ("The official language of Mexico is", ["Spanish"]),
    ("Bees make", ["honey"]), ("Cows give us", ["milk"]),
    ("A baby cat is called a", ["kitten"]), ("A baby dog is called a", ["puppy"]),
    ("The opposite of hot is", ["cold"]), ("The opposite of up is", ["down"]),
    ("The color of the sky on a clear day is", ["blue"]),
    ("Ripe bananas are usually", ["yellow"]), ("Grass is usually", ["green"]),
    ("There are seven days in a", ["week"]), ("There are twelve months in a", ["year"]),
    ("The sun rises in the", ["east", "East"]), ("Christmas is celebrated in the month of", ["December"]),
    ("Ice is frozen", ["water"]), ("The human heart pumps", ["blood"]),
]


# ---------------------------------------------------------------- data prep
def prepare_data(tok, n_ppl_chunks=40, ppl_len=256, n_blimp=100, n_sst=200, seed=0):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "eval_data.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    from datasets import load_dataset
    rng = random.Random(seed)

    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(wt["text"]), add_special_tokens=False).input_ids
    chunks = [ids[i:i + ppl_len] for i in range(0, len(ids) - ppl_len, ppl_len)]
    chunks = [chunks[i] for i in sorted(rng.sample(range(len(chunks)), n_ppl_chunks))]

    blimp = []
    for p in BLIMP_PARADIGMS:
        ds = load_dataset("nyu-mll/blimp", p, split="train")
        for i in rng.sample(range(len(ds)), n_blimp):
            blimp.append({"paradigm": p, "good": ds[i]["sentence_good"], "bad": ds[i]["sentence_bad"]})

    sst = load_dataset("stanfordnlp/sst2")
    train = sst["train"]
    pos = [r["sentence"].strip() for r in train if r["label"] == 1 and 6 < len(r["sentence"].split()) < 14][:2]
    neg = [r["sentence"].strip() for r in train if r["label"] == 0 and 6 < len(r["sentence"].split()) < 14][:2]
    shots = [(pos[0], 1), (neg[0], 0), (neg[1], 0), (pos[1], 1)]
    val = list(sst["validation"])
    val = [val[i] for i in rng.sample(range(len(val)), n_sst)]
    sst_items = [{"sentence": r["sentence"].strip(), "label": r["label"]} for r in val]

    data = {"ppl_chunks": chunks, "blimp": blimp, "sst_shots": shots, "sst": sst_items,
            "facts": [{"prompt": p, "answers": a} for p, a in FACTS]}
    with open(path, "w") as f:
        json.dump(data, f)
    return data


# ---------------------------------------------------------------- helpers
def _pad(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids, mask


@torch.no_grad()
def _forward(model, seqs, pad_id):
    ids, mask = _pad(seqs, pad_id)
    dev = next(model.parameters()).device
    return model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False).logits.float(), ids.to(dev), mask.to(dev)


def _cache_tensors(cache):
    """(key, value) per layer from a Hugging Face cache object (works across versions)."""
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return list(cache)


def _seq_logprob(model, seqs, pad_id, bs=16):
    """Sum of log-probs of tokens 1..n-1 of each sequence (right-padded batches)."""
    out = []
    for i in range(0, len(seqs), bs):
        logits, ids, mask = _forward(model, seqs[i:i + bs], pad_id)
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        out += (lp * mask[:, 1:]).sum(-1).tolist()
    return out


# ---------------------------------------------------------------- tasks
class TaskSuite:
    def __init__(self, tok, data, max_items=None):
        self.tok = tok
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        bos = [tok.bos_token_id]
        m = max_items or {}

        self.ppl_seqs = [bos + c for c in data["ppl_chunks"][:m.get("ppl", None)]]

        blimp = data["blimp"]
        if "blimp" in m:   # keep paradigms balanced when subsampling
            per = m["blimp"] // len(BLIMP_PARADIGMS)
            blimp = [b for p in BLIMP_PARADIGMS for b in [x for x in blimp if x["paradigm"] == p][:per]]
        self.blimp = blimp
        self.blimp_good = [tok(b["good"]).input_ids for b in blimp]
        self.blimp_bad = [tok(b["bad"]).input_ids for b in blimp]

        shots = data["sst_shots"][:m.get("sst_shots", None)]
        prefix = "".join(f"Review: {s}\nSentiment: {'positive' if y else 'negative'}\n\n" for s, y in shots)
        sst = data["sst"][:m.get("sst", None)]
        self.sst_seqs = [tok(prefix + f"Review: {r['sentence']}\nSentiment:").input_ids for r in sst]
        self.sst_labels = torch.tensor([r["label"] for r in sst])
        # the few-shot prefix is shared, so it is run once and its keys/values are reused
        self.sst_prefix = tok(prefix).input_ids
        P = len(self.sst_prefix)
        assert all(s[:P] == self.sst_prefix for s in self.sst_seqs)
        self.sst_items = [s[P:] for s in self.sst_seqs]
        self.pos_id = self._next_token_id("Sentiment:", " positive")
        self.neg_id = self._next_token_id("Sentiment:", " negative")

        self.fact_seqs, self.fact_targets = [], []
        for f in data["facts"]:
            prompt = FACT_PREFIX + f["prompt"]
            self.fact_seqs.append(tok(prompt).input_ids)
            self.fact_targets.append({self._next_token_id(prompt, " " + a) for a in f["answers"]})

    def _next_token_id(self, prompt, cont):
        p = self.tok(prompt).input_ids
        full = self.tok(prompt + cont).input_ids
        assert full[:len(p)] == p, (prompt, cont)
        return full[len(p)]

    # 1. next-word prediction
    def perplexity(self, model, bs=4):
        nll, n = 0.0, 0
        for i in range(0, len(self.ppl_seqs), bs):
            logits, ids, mask = _forward(model, self.ppl_seqs[i:i + bs], self.pad)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="sum")
            nll += loss.item()
            n += ids[:, 1:].numel()
        return float(torch.exp(torch.tensor(nll / n)))

    # 2. syntax
    def blimp_accuracy(self, model):
        g = _seq_logprob(model, self.blimp_good, self.pad)
        b = _seq_logprob(model, self.blimp_bad, self.pad)
        correct = [x > y for x, y in zip(g, b)]
        return sum(correct) / len(correct)

    # 3. sentiment
    @torch.no_grad()
    def sst_accuracy(self, model, bs=16, use_prefix_cache=True):
        if not use_prefix_cache:
            return self._sst_accuracy_full(model, bs=8)
        from transformers import DynamicCache
        dev = next(model.parameters()).device
        out = model(input_ids=torch.tensor([self.sst_prefix], device=dev), use_cache=True)
        kv = _cache_tensors(out.past_key_values)
        P = len(self.sst_prefix)
        preds = []
        for i in range(0, len(self.sst_items), bs):
            items = self.sst_items[i:i + bs]
            ids, mask = _pad(items, self.pad)
            B = len(items)
            cache = DynamicCache()
            for layer, (k, v) in enumerate(kv):
                cache.update(k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous(), layer)
            full_mask = torch.cat([torch.ones(B, P, dtype=torch.long), mask], 1).to(dev)
            logits = model(input_ids=ids.to(dev), attention_mask=full_mask, past_key_values=cache,
                           use_cache=True).logits.float()
            last = torch.tensor([len(s) - 1 for s in items], device=dev)
            ll = logits[torch.arange(B, device=dev), last]
            preds += (ll[:, self.pos_id] > ll[:, self.neg_id]).long().cpu().tolist()
        return (torch.tensor(preds) == self.sst_labels).float().mean().item()

    def _sst_accuracy_full(self, model, bs=8):
        preds = []
        for i in range(0, len(self.sst_seqs), bs):
            seqs = self.sst_seqs[i:i + bs]
            logits, _, _ = _forward(model, seqs, self.pad)
            last = torch.tensor([len(s) - 1 for s in seqs], device=logits.device)
            ll = logits[torch.arange(len(seqs)), last]
            preds += (ll[:, self.pos_id] > ll[:, self.neg_id]).long().cpu().tolist()
        return (torch.tensor(preds) == self.sst_labels).float().mean().item()

    # 4. factual completion
    def fact_accuracy(self, model, bs=16):
        hits = []
        for i in range(0, len(self.fact_seqs), bs):
            seqs = self.fact_seqs[i:i + bs]
            logits, _, _ = _forward(model, seqs, self.pad)
            last = torch.tensor([len(s) - 1 for s in seqs], device=logits.device)
            top = logits[torch.arange(len(seqs)), last].argmax(-1).tolist()
            hits += [t in tg for t, tg in zip(top, self.fact_targets[i:i + bs])]
        return sum(hits) / len(hits)

    def evaluate(self, model, tasks=("ppl", "blimp", "sst", "facts")):
        res = {}
        if "ppl" in tasks:
            res["ppl"] = self.perplexity(model)
        if "blimp" in tasks:
            res["blimp"] = self.blimp_accuracy(model)
        if "sst" in tasks:
            res["sst"] = self.sst_accuracy(model)
        if "facts" in tasks:
            res["facts"] = self.fact_accuracy(model)
        return res

    # gradient-based head importance (Michel et al., 2019): |dL/d mask_h| on LM loss
    def head_importance(self, model, ablator, n_chunks=16):
        mask = ablator.head_mask
        mask.requires_grad_(True)
        imp = torch.zeros_like(mask)
        dev = next(model.parameters()).device
        for seq in self.ppl_seqs[:n_chunks]:
            ids = torch.tensor([seq], device=dev)
            logits = model(input_ids=ids, use_cache=False).logits.float()
            loss = F.cross_entropy(logits[0, :-1], ids[0, 1:])
            g, = torch.autograd.grad(loss * 128.0, mask)     # scaled to avoid fp16 underflow
            imp += g.abs() / 128.0
        mask.requires_grad_(False)
        return imp.cpu()
