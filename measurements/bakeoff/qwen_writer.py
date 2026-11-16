"""Bake-off writer: Qwen3.5-4B writes a short reason for every option of every question.

Run on the runtime, after uploading questions.json to /content/bakeoff_questions.json:
    colab exec -s <session> -f measurements/bakeoff/qwen_writer.py
Writes /content/reasons_qwen.json in the same form as the other writers.
"""
import json, subprocess, time

subprocess.run("pip install -q flash-linear-attention==0.6.0", shell=True, capture_output=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()

INSTRUCTION = (
    "You are helping build a dataset of short user messages. Below is a multiple-choice question and one of its "
    "options. Write the reason a user would give for believing that option is the answer. You are not told which "
    "option is correct; make the best short case for the option you are given.\n\n"
    "Rules: one or two sentences, 20 to 40 words. First person, plain conversational English. Argue from the "
    "content of the question. Do not mention any option letter. Do not say you are unsure or certain. Do not cite "
    "any source or authority. Do not argue against other options. Output only the reason.\n\n"
    "Question: {q}\nOptions:\n{opts}\n\nThe option to argue for: {target}"
)

questions = json.load(open("/content/bakeoff_questions.json"))
jobs = []
for q in questions:
    opts = "\n".join(f"- {t}" for t in q["options"].values())
    for letter, text in q["options"].items():
        msg = [{"role": "user", "content": INSTRUCTION.format(q=q["question"], opts=opts, target=text)}]
        jobs.append((q["qid"], letter, tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                                               enable_thinking=False)))
t0 = time.time()
reasons = []
with torch.inference_mode():
    for s in range(0, len(jobs), 40):
        chunk = jobs[s:s + 40]
        enc = tok([c[2] for c in chunk], return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=90, do_sample=False, repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id)
        for (qid, letter, _), row in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
            reasons.append({"qid": qid, "option": letter, "reason": tok.decode(row, skip_special_tokens=True).strip()})
t1 = time.time()
json.dump({"writer": "qwen3.5-4b", "started_unix": int(t0), "finished_unix": int(t1),
           "generation_seconds": round(t1 - t0, 1), "reasons": reasons},
          open("/content/reasons_qwen.json", "w"), indent=1)
print("QWEN_WRITER", len(reasons), "reasons in", round(t1 - t0, 1), "s")
print("QWEN_WRITER sample:", reasons[0]["reason"])
