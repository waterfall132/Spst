"""Zero-shot LLM teacher: class probabilities for every (sentence, aspect) of a split.

One forward pass per example; the probability of each polarity is read from the logits of
the first token of the answer (label words "negative" / "neutral" / "positive", lower- and
capitalised forms merged with logsumexp). Labels are NEVER used except to report the
teacher's own test score.

  python llm_label.py --target twitter --splits train,test --model Qwen/Qwen2.5-3B-Instruct \
      --out data/llm/qwen3b_twitter
  -> data/llm/qwen3b_twitter_train.npz, ..._test.npz   (keys, probs[N,3])
  python llm_label.py --target twitter --fake --out data/llm/fake_twitter   # random probs, smoke test
"""
import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data import LABELS, load
from llm_baseline import build_messages


def key_of(x):
    return x["text"] + "\t" + x["aspect"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--target", required=True)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--fake", action="store_true", help="random probabilities (pipeline test only)")
    ap.add_argument("--out", required=True, help="output prefix")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if not a.fake:
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
        label_ids = [sorted({tok.encode(w, add_special_tokens=False)[0] for w in (lab, lab.capitalize())})
                     for lab in LABELS]
        print("label first-token ids:", label_ids)

    for split in a.splits.split(","):
        items = load(a.data, a.target, split)
        if a.fake:
            rng = np.random.default_rng(0)
            probs = rng.dirichlet([1, 1, 1], size=len(items))
        else:
            out = []
            for i in range(0, len(items), a.bs):
                prompts = [tok.apply_chat_template(build_messages([], x), tokenize=False, add_generation_prompt=True)
                           for x in items[i:i + a.bs]]
                enc = tok(prompts, return_tensors="pt", padding=True).to(lm.device)
                with torch.no_grad():
                    logits = lm(**enc).logits[:, -1, :].float()
                lp = torch.log_softmax(logits, -1)
                sc = torch.stack([torch.logsumexp(lp[:, ids], -1) for ids in label_ids], -1)
                out.append(torch.softmax(sc, -1).cpu().numpy())
            probs = np.concatenate(out)
        keys = np.array([key_of(x) for x in items])
        np.savez(f"{a.out}_{split}.npz", keys=keys, probs=probs)
        y = np.array([x["label"] for x in items])
        p = probs.argmax(1)
        print(f"[{a.target}/{split}] teacher acc {100 * accuracy_score(y, p):.2f}  "
              f"macro-F1 {100 * f1_score(y, p, average='macro'):.2f}  "
              f"(pred counts {np.bincount(p, minlength=3).tolist()})  -> {a.out}_{split}.npz")


if __name__ == "__main__":
    main()
