"""Instruction-tuned LLM baseline: zero-shot and K-shot in-context learning.

Uses exactly the same K-shot support sets as train.py (seed 1000 + s) as demonstrations,
and writes preds.npz / results.json in the same format, so stats.py and summarize.py
work unchanged.  Reviewers in 2026 will ask for this baseline.

  python llm_baseline.py --target laptop --model Qwen/Qwen2.5-3B-Instruct --shots 0,1,5 \
      --seeds 5 --out results/LLM_qwen3b_laptop
  python llm_baseline.py --target laptop --dry_run          # print one prompt, no model

A 3B model in bf16 fits a 12 GB GPU; for 7B use --load_4bit (needs bitsandbytes).
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data import LABELS, load, sample_support

SYSTEM = ("You are an expert annotator for aspect-based sentiment analysis. Given a sentence and "
          "one aspect term in it, answer with exactly one word: positive, negative or neutral, "
          "describing the sentiment expressed toward that aspect.")


def fmt(x):
    return f'Sentence: "{x["text"]}"\nAspect: "{x["aspect"]}"\nSentiment:'


def build_messages(demos, x):
    msgs = [{"role": "system", "content": SYSTEM}]
    for d in demos:
        msgs.append({"role": "user", "content": fmt(d)})
        msgs.append({"role": "assistant", "content": LABELS[d["label"]]})
    msgs.append({"role": "user", "content": fmt(x)})
    return msgs


def parse(text):
    t = text.strip().lower()
    for i, lab in enumerate(LABELS):
        if t.startswith(lab):
            return i
    hits = [(t.find(lab), i) for i, lab in enumerate(LABELS) if lab in t]
    return min(hits)[1] if hits else 1  # unparsable -> neutral (counted, reported)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--shots", default="0,1,5")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tr, test = load(a.data, a.target, "train"), load(a.data, a.target, "test")
    labels = np.array([x["label"] for x in test])
    shots = [int(k) for k in a.shots.split(",")]

    if a.dry_run:
        demos = sample_support(tr, 1, 1000)
        for m in build_messages(demos, test[0]):
            print(f"[{m['role']}] {m['content']}")
        print("parse checks:", parse("Positive."), parse(" negative"), parse("It is neutral"), parse("??"))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    lm = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()

    os.makedirs(a.out, exist_ok=True)
    res, allp = {}, {"labels": labels}
    for K in shots:
        accs, f1s, P = [], [], []
        for s in range(a.seeds if K > 0 else 1):
            demos = sample_support(tr, K, 1000 + s) if K > 0 else []
            # interleave classes so the last demonstration is not always the same label
            demos = [d for trip in zip(*[demos[c * K:(c + 1) * K] for c in range(3)]) for d in trip] if K else []
            preds, bad = [], 0
            for i in range(0, len(test), a.bs):
                prompts = [tok.apply_chat_template(build_messages(demos, x), tokenize=False,
                                                   add_generation_prompt=True) for x in test[i:i + a.bs]]
                enc = tok(prompts, return_tensors="pt", padding=True).to(lm.device)
                with torch.no_grad():
                    out = lm.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
                for o in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
                    bad += not any(l in o.lower() for l in LABELS)
                    preds.append(parse(o))
            p = np.array(preds)
            accs.append(accuracy_score(labels, p)); f1s.append(f1_score(labels, p, average="macro"))
            P.append(p)
            print(f"K={K} seed={s} acc {100 * accs[-1]:.2f} F1 {100 * f1s[-1]:.2f} unparsable={bad}", flush=True)
        res[K] = {"acc": accs, "f1": f1s, "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                  "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
        allp[f"preds_K{K}"] = np.stack(P)
    cfg = {"source": "none(LLM)", "target": a.target, "pool": a.model.split("/")[-1], "head": "icl",
           "adapt": "none", "score": "-", "r": "-"}
    json.dump({"config": cfg, "shots": res}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    np.savez(os.path.join(a.out, "preds.npz"), **allp)


if __name__ == "__main__":
    main()
