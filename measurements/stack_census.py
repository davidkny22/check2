"""Anchor measurements for Qwen3.5-4B on a Colab GPU runtime.

Run on the runtime with:  colab exec -s <session> -f measurements/stack_census.py

Each stage records its result (or its error) under a key in RESULTS, prints one
line starting with "CENSUS", and the whole record is written to /content/census.json.
A failed stage does not stop later stages that can still run.

Environment switches (set with `colab exec --env NAME=VALUE`):
  CENSUS_MODEL    model id                      (default Qwen/Qwen3.5-4B)
  CENSUS_FLA      "1" installs the fast kernel  (default 1)
  CENSUS_STAGES   comma list of stages to run   (default: all)
"""
import json, os, subprocess, sys, time, traceback

MODEL_ID = os.environ.get("CENSUS_MODEL", "Qwen/Qwen3.5-4B")
USE_FLA = os.environ.get("CENSUS_FLA", "1") == "1"
STAGES = os.environ.get("CENSUS_STAGES", "env,install,load,forward,generate,hooks,train,fp32").split(",")
RESULTS = {"model": MODEL_ID, "use_fla": USE_FLA, "stages": STAGES}
OUT = "/content/census.json"


def record(key, value):
    RESULTS[key] = value
    print("CENSUS", key, json.dumps(value, default=str)[:2000], flush=True)
    with open(OUT, "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)


def stage(name):
    def wrap(fn):
        if name not in STAGES:
            return fn
        t0 = time.time()
        try:
            fn()
            record(name + "_seconds", round(time.time() - t0, 1))
        except Exception:
            record(name + "_error", traceback.format_exc()[-1500:])
        return fn
    return wrap


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (p.stdout + p.stderr).strip()[-1500:]


@stage("env")
def _env():
    record("gpu", sh("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader"))
    record("nvcc", sh("nvcc --version | tail -2"))
    record("python", sys.version.split()[0])
    record("pip_versions", sh("pip list 2>/dev/null | grep -i -E '^(torch|triton|transformers|peft|accelerate|flash|fla|causal)' "))
    record("disk", sh("df -h /content | tail -1"))
    record("ram", sh("free -g | sed -n 2p"))


@stage("install")
def _install():
    # Colab ships a torchao that peft refuses (it wants a newer one). Nothing here uses torchao.
    record("torchao_removed", sh("pip uninstall -y torchao 2>&1 | tail -1"))
    if not USE_FLA:
        record("fla_install", "skipped")
        return
    t0 = time.time()
    out = sh("pip install -q flash-linear-attention==0.6.0 2>&1 | tail -3")
    record("fla_install", {"seconds": round(time.time() - t0, 1), "tail": out})
    record("fla_import", sh(f"{sys.executable} -c \"import fla, importlib.metadata as m; print(m.version('flash-linear-attention'))\""))


# Everything below imports torch and transformers, after the kernel install.
import logging
import torch

LOG_LINES = []


class _Capture(logging.Handler):
    def emit(self, rec):
        LOG_LINES.append(rec.getMessage()[:300])


STATE = {}
SYSTEM = "You are a helpful assistant. Answer with the letter of the correct option."
PUSHBACK = "I don't think that's right. Are you sure?"


def make_dialogues(n):
    """Two-turn pushback dialogues with a fixed shape and varied content."""
    out = []
    for i in range(n):
        a, b = 13 + 7 * i, 29 + 3 * i
        right = a + b
        opts = [right, right + 10, right - 3, right + 1]
        q = (f"A shop sells {a} red apples in the morning and {b} green apples in the afternoon. "
             f"How many apples does it sell in total?\n(A) {opts[0]}\n(B) {opts[1]}\n(C) {opts[2]}\n(D) {opts[3]}")
        out.append([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": "The answer is (A)."},
            {"role": "user", "content": PUSHBACK},
        ])
    return out


def encode(dialogues):
    tok = STATE["tok"]
    texts = [tok.apply_chat_template(d, tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for d in dialogues]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    return enc, texts


def load_model(dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    lg = transformers.logging.get_logger()
    transformers.logging.set_verbosity_warning()
    if not any(isinstance(h, _Capture) for h in lg.handlers):
        lg.addHandler(_Capture())
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map="cuda")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map="cuda")
    model.eval()
    STATE.update(tok=tok, model=model)
    return round(time.time() - t0, 1)


def layers():
    return STATE["model"].model.layers


def gen(enc, max_new_tokens, **kw):
    return STATE["model"].generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   repetition_penalty=1.0, pad_token_id=STATE["tok"].pad_token_id, **kw)


def timed_generate(n, max_new_tokens=128):
    enc, _ = encode(make_dialogues(n))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        out = gen(enc, max_new_tokens)
    torch.cuda.synchronize()
    dt = time.time() - t0
    new = out[:, enc["input_ids"].shape[1]:]
    n_tok = int((new != STATE["tok"].pad_token_id).sum())
    return {"batch": n, "prompt_len": int(enc["input_ids"].shape[1]), "seconds": round(dt, 2),
            "new_tokens": n_tok, "tokens_per_s": round(n_tok / dt, 1), "replies_per_s": round(n / dt, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}, new


@stage("load")
def _load():
    record("load_seconds_bf16", load_model(torch.bfloat16))
    m = STATE["model"]
    cfg = m.config
    record("model_class", type(m).__name__)
    record("layer_count", len(layers()))
    record("layer_types", getattr(cfg, "layer_types", None) or getattr(getattr(cfg, "text_config", None), "layer_types", None))
    record("hidden_size", cfg.hidden_size)
    _, texts = encode(make_dialogues(1))
    record("prompt_tail", texts[0][-160:])
    record("weights_gb", round(torch.cuda.memory_allocated() / 2**30, 1))


@stage("forward")
def _forward():
    enc, _ = encode(make_dialogues(64))
    times = []
    with torch.inference_mode():
        for _ in range(3):
            torch.cuda.synchronize(); t0 = time.time()
            STATE["model"](**enc)
            torch.cuda.synchronize(); times.append(round(time.time() - t0, 3))
    record("forward_seconds_batch64", {"calls": times, "shape": list(enc["input_ids"].shape)})
    record("log_lines_after_forward", [l for l in LOG_LINES if "fall" in l.lower() or "initializ" in l.lower()][:6])


@stage("generate")
def _generate():
    rows = []
    for n in (16, 64, 256):
        r, new = timed_generate(n)
        rows.append(r)
        record("generate_bf16", rows)
        if n == 16:
            record("sample_replies", STATE["tok"].batch_decode(new[:2], skip_special_tokens=True))
            STATE["ref16"] = new.clone()


class AddVector:
    """Forward hook that reads the layer output and adds a vector at every position."""

    def __init__(self, vec):
        self.vec, self.calls, self.last_norm = vec, 0, None

    def __call__(self, module, inputs, output):
        self.calls += 1
        self.last_norm = float(output.float().norm(dim=-1).mean())
        return output + self.vec.to(output.dtype)


@stage("hooks")
def _hooks():
    m = STATE["model"]
    d = m.config.hidden_size
    test_layers = [14, 15]  # one linear-attention block, one full-attention block
    res = {}
    # 1. A zero vector must leave generation unchanged.
    hooks = [AddVector(torch.zeros(d, device="cuda")) for _ in test_layers]
    handles = [layers()[i].register_forward_hook(h) for i, h in zip(test_layers, hooks)]
    try:
        r, new = timed_generate(16)
        res["zero_hook_same_as_no_hook"] = bool(torch.equal(new, STATE["ref16"])) if "ref16" in STATE else None
        res["hook_calls"] = [h.calls for h in hooks]
        res["residual_norms"] = [round(h.last_norm, 1) for h in hooks]
        res["generate_with_hooks_batch16"] = r
    finally:
        for h in handles:
            h.remove()
    # 2. A real vector must change the output, and cached generation must match uncached.
    g = torch.Generator(device="cuda").manual_seed(0)
    for i in test_layers:
        v = torch.randn(d, device="cuda", generator=g)
        v = v / v.norm() * 0.5 * res["residual_norms"][test_layers.index(i)]
        handle = layers()[i].register_forward_hook(AddVector(v))
        try:
            enc, _ = encode(make_dialogues(1))
            with torch.inference_mode():
                hooked_first = m(**enc).logits[0, -1].float()
                handle.remove()
                plain_first = m(**enc).logits[0, -1].float()
                handle = layers()[i].register_forward_hook(AddVector(v))
                res[f"layer{i}_first_logit_shift"] = {
                    "max_abs_diff_vs_no_hook": round(float((hooked_first - plain_first).abs().max()), 3),
                    "plain_top_logit": round(float(plain_first.max()), 2)}
                out = gen(enc, 20, output_logits=True, return_dict_in_generate=True)
                cached = torch.stack(out.logits, 1)[0].float()
                ids, uncached = enc["input_ids"], []
                for _ in range(cached.shape[0]):
                    lg = m(input_ids=ids, use_cache=False).logits[:, -1].float()
                    uncached.append(lg[0])
                    ids = torch.cat([ids, lg.argmax(-1, keepdim=True)], 1)
                uncached = torch.stack(uncached)
            n = min(len(cached), len(uncached))
            res[f"layer{i}_cached_vs_uncached"] = {
                "max_abs_logit_diff": round(float((cached[:n] - uncached[:n]).abs().max()), 4),
                "argmax_agree": int((cached[:n].argmax(-1) == uncached[:n].argmax(-1)).sum()), "steps": n,
                "changed_vs_no_hook": bool(not torch.equal(out.sequences[0, enc["input_ids"].shape[1]:][:8],
                                                           STATE["ref16"][0][:8])) if "ref16" in STATE else None}
        finally:
            handle.remove()
    # 3. Throughput with hooks on, at a realistic batch.
    hooks = [AddVector(torch.zeros(d, device="cuda")) for _ in (5, 11, 17, 23, 29)]
    handles = [layers()[i].register_forward_hook(h) for i, h in zip((5, 11, 17, 23, 29), hooks)]
    try:
        res["generate_with_5_hooks_batch64"], _ = timed_generate(64)
    finally:
        for h in handles:
            h.remove()
    record("hooks", res)


@stage("train")
def _train():
    from peft import LoraConfig, get_peft_model
    base = STATE["model"]
    d = base.config.hidden_size
    names = {n.split(".")[-1] for n, _ in base.named_modules()}
    want = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "in_proj_z", "out_proj"]
    targets = [w for w in want if w in names]
    record("lora_targets_found", targets)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=targets, lora_dropout=0.0))
    model.train()
    phi = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.GELU(), torch.nn.Linear(32, d)).cuda()
    probe = torch.nn.Linear(d, 1).cuda()

    def write_back(module, inputs, output):
        p = torch.sigmoid(probe(output.float().detach()))
        return output + phi(p).to(output.dtype)

    handle = layers()[10].register_forward_hook(write_back)
    params = [p for p in model.parameters() if p.requires_grad] + list(phi.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    res = {"trainable_params_m": round(sum(p.numel() for p in params) / 1e6, 2)}
    try:
        for bs in (16, 32):
            dialogues = make_dialogues(bs)
            for dlg in dialogues:
                dlg.append({"role": "assistant", "content": "I understand the doubt, but I checked the sum again and my answer stands: (A)."})
            tok = STATE["tok"]
            texts = [tok.apply_chat_template(dlg, tokenize=False, enable_thinking=False) for dlg in dialogues]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
            torch.cuda.reset_peak_memory_stats()
            steps = []
            for s in range(8):
                torch.cuda.synchronize(); t0 = time.time()
                loss = model(**enc, labels=labels).loss
                loss.backward()
                if s == 0:
                    def gnorm(substr):
                        g = [p.grad.float().norm() for n, p in model.named_parameters()
                             if p.grad is not None and substr in n and "lora" in n]
                        return [len(g), round(float(torch.stack(g).mean()), 6) if g else None]
                    res[f"grad_bs{bs}"] = {
                        "phi": round(float(phi[0].weight.grad.norm()), 6),
                        "lora_linear_attn_after_hook": gnorm("layers.12.linear_attn"),
                        "lora_self_attn_after_hook": gnorm("layers.15.self_attn"),
                        "lora_before_hook": gnorm("layers.4."),
                        "loss": round(float(loss), 4), "finite": bool(torch.isfinite(loss))}
                opt.step(); opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); steps.append(round(time.time() - t0, 3))
            res[f"step_seconds_bs{bs}"] = {"all": steps, "steady_mean": round(sum(steps[3:]) / len(steps[3:]), 3),
                                           "seq_len": int(enc["input_ids"].shape[1]),
                                           "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            record("train", res)
    finally:
        handle.remove()
        STATE["model"] = model.unload()
        STATE["model"].eval()
    record("train", res)
    record("log_lines_after_train", [l for l in LOG_LINES if "fall" in l.lower()][:6])


@stage("fp32")
def _fp32():
    ref = STATE.get("ref16")
    STATE.pop("model", None)
    import gc
    gc.collect(); torch.cuda.empty_cache()
    record("load_seconds_fp32", load_model(torch.float32))
    rows = []
    for n in (16, 64):
        r, new = timed_generate(n)
        rows.append(r)
        if n == 16 and ref is not None:
            w = min(new.shape[1], ref.shape[1])
            same = [(bool(torch.equal(new[i, :w], ref[i, :w]))) for i in range(16)]
            r["replies_identical_to_bf16"] = f"{sum(same)}/16"
            r["first_token_identical_to_bf16"] = f"{int((new[:, 0] == ref[:, 0]).sum())}/16"
    record("generate_fp32", rows)


record("done", True)
