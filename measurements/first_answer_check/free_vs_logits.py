"""Check: is the first answer read from option logits the same answer the model writes freely?

For a sample of questions, the model answers the same prompt by free greedy generation.
Reports how often its written letter equals the letter read from the logits, how long
the free answers are, and what form they take. Reads /content/first_answers.json.
"""
import collections, json, random, re, subprocess
subprocess.run("pip install -q flash-linear-attention==0.6.0", shell=True, capture_output=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_ID = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
L = "ABCD"
qs = json.load(open("/content/first_answers.json"))["questions"]
random.Random(5).shuffle(qs); qs = qs[:400]
def qt(q):
    return "Question: " + q["question"] + "\n" + "\n".join(f"({l}) {c}" for l, c in zip(L, q["choices"])) + "\n\nAnswer with the letter of the correct option."
texts = [tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": qt(q)}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in qs]
out_text, n_tok = [None] * len(qs), [0] * len(qs)
order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
with torch.inference_mode():
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        gen = model.generate(**enc, max_new_tokens=160, do_sample=False, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            row = gen[j, enc["input_ids"].shape[1]:]; keep = row[row != tok.pad_token_id]
            n_tok[i] = int(keep.numel()); out_text[i] = tok.decode(keep, skip_special_tokens=True).strip()
def letter(t):
    m = re.search(r"\(([A-D])\)", t) or re.match(r"\s*\**([A-D])\b", t) or re.search(r"answer is[:\s\*]*\(?([A-D])\b", t, re.I)
    return L.index(m.group(1)) if m else None
got = [letter(t) for t in out_text]
parsed = [g is not None for g in got]
agree = [g == q["a1"] for g, q in zip(got, qs) if g is not None]
forms = collections.Counter(re.sub(r"[A-D]", "X", t[:18]) for t in out_text)
res = {"n": len(qs), "parsed": sum(parsed), "letter_equals_logit_read": sum(agree), "share_equal": round(sum(agree) / max(1, len(agree)), 4),
       "tokens_mean": round(sum(n_tok) / len(n_tok), 1), "tokens_median": sorted(n_tok)[len(n_tok) // 2], "hit_160_cap": sum(n >= 160 for n in n_tok),
       "short_answers_under_12_tokens": sum(n < 12 for n in n_tok), "commonest_openings": forms.most_common(6), "samples": out_text[:5]}
json.dump(res, open("/content/free_vs_logits.json", "w"), indent=1)
print("FREECHECK", json.dumps(res)[:1800])
