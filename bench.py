"""Deployment cost of the BERT student vs. zero-shot LLMs on one target test set.

Reports parameters, peak GPU memory and inference throughput (examples / s) for
  * the student encoder (bert-base, sentence-pair input, one forward per example);
  * each LLM in --llms (zero-shot prompt, one forward per example = the cheapest possible
    LLM inference: only the first answer token is scored, as in llm_label.py).
Accuracy of the LLMs comes from llm_label.py; the student's from results_confirm.

  python bench.py --target twitter --llms Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct
"""
import argparse
import json
import time

import torch

from data import Collator, load
from llm_baseline import build_messages


def timed(fn, n_items, device):
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.time()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else float("nan")
    return n_items / dt, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--target", default="twitter")
    ap.add_argument("--student", default="bert-base-uncased")
    ap.add_argument("--llms", default="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default="results_cost.json")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = load(a.data, a.target, "test")
    rows = []

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.student)
    enc = AutoModel.from_pretrained(a.student, add_pooling_layer=False).to(dev).eval()
    col = Collator(tok)

    def run_student():
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            for i in range(0, len(test), a.bs):
                b, _ = col(test[i:i + a.bs])
                enc(**{k: v.to(dev) for k, v in b.items() if k != "seg"})

    run_student()  # warm-up
    thr, mem = timed(run_student, len(test), dev)
    rows.append({"model": a.student, "params_M": sum(p.numel() for p in enc.parameters()) / 1e6,
                 "throughput_ex_per_s": thr, "peak_mem_GB": mem})
    print(rows[-1], flush=True)
    del enc
    torch.cuda.empty_cache() if dev.type == "cuda" else None

    for name in [m for m in a.llms.split(",") if m]:
        ltok = AutoTokenizer.from_pretrained(name)
        ltok.padding_side = "left"
        if ltok.pad_token is None:
            ltok.pad_token = ltok.eos_token
        lm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(dev).eval()
        prompts = [ltok.apply_chat_template(build_messages([], x), tokenize=False, add_generation_prompt=True)
                   for x in test]

        def run_llm():
            with torch.no_grad():
                for i in range(0, len(prompts), a.bs):
                    e = ltok(prompts[i:i + a.bs], return_tensors="pt", padding=True).to(dev)
                    lm(**e).logits[:, -1, :]

        run_llm()  # warm-up
        thr, mem = timed(run_llm, len(test), dev)
        rows.append({"model": name, "params_M": sum(p.numel() for p in lm.parameters()) / 1e6,
                     "throughput_ex_per_s": thr, "peak_mem_GB": mem})
        print(rows[-1], flush=True)
        del lm
        torch.cuda.empty_cache() if dev.type == "cuda" else None

    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n| model | params (M) | throughput (ex/s) | peak GPU mem (GB) |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['params_M']:.0f} | {r['throughput_ex_per_s']:.0f} | {r['peak_mem_GB']:.2f} |")


if __name__ == "__main__":
    main()
