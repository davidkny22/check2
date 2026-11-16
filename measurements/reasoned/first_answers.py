"""Step 1 of the reasoned-pushback test: Qwen3.5-4B's first answers on corrected MMLU.

Loads MMLU-Redux 2.0 for a list of school-level subjects, keeps only the questions its
annotators marked "ok", and reads the model's first answer from the option logits.
Writes /content/first_answers.json.
"""
import json, subprocess, time

subprocess.run("pip install -q flash-linear-attention==0.6.0", shell=True, capture_output=True)
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SUBJECTS = ["elementary_mathematics", "high_school_mathematics", "college_mathematics", "abstract_algebra",
            "conceptual_physics", "high_school_physics", "college_physics", "astronomy", "high_school_geography",
            "high_school_macroeconomics", "high_school_microeconomics", "high_school_statistics", "philosophy",
            "world_religions", "logical_fallacies", "formal_logic", "prehistory", "global_facts", "econometrics",
            "management", "marketing", "sociology", "high_school_government_and_politics", "public_relations",
            "professional_accounting", "machine_learning", "high_school_computer_science", "college_computer_science"]
MODEL_ID = "Qwen/Qwen3.5-4B"
LETTERS = "ABCD"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
LETTER_IDS = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]

questions, dropped = [], 0
for s in SUBJECTS:
    ds = load_dataset("edinburgh-dawg/mmlu-redux-2.0", s, split="test")
    for i, r in enumerate(ds):
        if r["error_type"] != "ok" or len(r["choices"]) != 4:
            dropped += 1
            continue
        questions.append(dict(qid=f"{s}/{i}", subject=s, question=r["question"], choices=r["choices"], gold=int(r["answer"])))


def question_text(q):
    opts = "\n".join(f"({l}) {c}" for l, c in zip(LETTERS, q["choices"]))
    return f"Question: {q['question']}\n{opts}\n\nAnswer with the letter of the correct option."


texts = [tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."},
                                  {"role": "user", "content": question_text(q)}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=False) + "The answer is ("
         for q in questions]
t0 = time.time()
order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
with torch.inference_mode():
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        p = torch.softmax(model(**enc, logits_to_keep=1).logits[:, -1, LETTER_IDS].float(), -1).cpu().numpy()
        for i, row in zip(idx, p):
            questions[i]["p1"] = [round(float(x), 4) for x in row]
            questions[i]["a1"] = int(row.argmax())
acc = float(np.mean([q["a1"] == q["gold"] for q in questions]))
json.dump({"model": MODEL_ID, "dataset": "edinburgh-dawg/mmlu-redux-2.0", "kept": len(questions), "dropped": dropped,
           "accuracy": round(acc, 4), "questions": questions}, open("/content/first_answers.json", "w"))
print("FIRST", "kept", len(questions), "dropped", dropped, "accuracy", round(acc, 4),
      "wrong", sum(q["a1"] != q["gold"] for q in questions), "seconds", round(time.time() - t0, 1))
