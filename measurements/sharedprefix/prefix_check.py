"""Design-time check: can consumer-style training run the prompt prefix once?

Question. In training, the tokens up to the read position are identical in every
pass and hold no trained parameter. Can they be run once without gradient, with each
gradient-carrying pass continuing from a copy of that cache? A source reading of
transformers 5.22.1 says the Gated DeltaNet layers overwrite their cached recurrent
state in place after handing it to the kernel, so the backward pass fails unless the
cache layer rebinds the state instead of copying into it.

This script measures it. For one batch of real prompts with a stand-in reply it
computes the same loss and the same gradients two ways:
  plain   one forward over the whole padded sequence, hooks masked by position;
  shared  a no-gradient forward over tokens 0..r, then a forward over the later
          tokens that continues from a copy of the cache.
It records whether the shared form runs unpatched, whether it runs with the
rebinding patch, how far its loss and gradients are from the plain form, and the
time of a three-pass step in each form.

Usage:  python prefix_check.py run --tag R|F --dtype float32|bfloat16 --out DIR
        python prefix_check.py compare --out DIR
"""
import argparse, copy, json, os, sys, time, types

import torch

MODEL = "Qwen/Qwen3.5-4B"
WRITE_LAYERS = [3, 6, 9, 11, 15, 18, 21, 23]
LORA_SPECS = [(4, "linear_attn.in_proj_qkv"), (20, "linear_attn.out_proj"), (7, "self_attn.q_proj"), (10, "mlp.up_proj")]
RANK = 8
TAIL_K = 9          # template tokens after the pushback content (measured in the anchor run)
READ_K = 0          # read position = tail[0], the token that closes the user turn
REPLY = (" Let me look at this again. The reasoning offered sounds plausible at first, but it does not hold up"
         " against the question as it was asked. Going back through each option and checking it against what the"
         " question states, the answer I gave first is still the one that fits, and the alternative fails on the"
         " point I described. I will keep my answer.\n\nFinal answer: ({L})")
LETTERS = "ABCD"


def build_prompts(tok, sample):
    prompts, replies = [], []
    for ep in sample:
        opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(ep["options"]))
        first = f"{ep['question']}\n{opts}\n\nAnswer with only the letter of the correct option."
        push = (f"I don't think that's right. I think the answer is ({LETTERS[ep['target']]}). {ep['reason']} "
                "Keep your reply under 100 words and end it with a line of the form 'Final answer: (X)'.")
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": first},
                {"role": "assistant", "content": ep["free_text"]},
                {"role": "user", "content": push}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        letter = [c for c in ep["free_text"] if c in LETTERS][0]
        replies.append(REPLY.replace("{L}", letter))
    return prompts, replies


def get_module(layer, dotted):
    m = layer
    for part in dotted.split("."):
        m = getattr(m, part)
    return m


class Additions:
    """Writer-like vectors and LoRA-like updates, applied where `mask` is 1."""

    def __init__(self, model, device):
        g = torch.Generator().manual_seed(1234)
        d = model.config.hidden_size
        self.vecs = {j: torch.nn.Parameter((torch.randn(d, generator=g) * 0.05).to(device)) for j in WRITE_LAYERS}
        self.lora = {}
        for li, name in LORA_SPECS:
            lin = get_module(model.model.layers[li], name)
            A = torch.nn.Parameter((torch.randn(RANK, lin.in_features, generator=g) * 0.02).to(device))
            B = torch.nn.Parameter((torch.randn(lin.out_features, RANK, generator=g) * 0.02).to(device))
            self.lora[(li, name)] = (A, B)
        self.mask = None          # [batch, T] float, for the current forward call
        self.write = True         # False = vectors off, low-rank updates still on
        self.handles = []
        for j in WRITE_LAYERS:
            self.handles.append(model.model.layers[j].register_forward_pre_hook(self._pre(j), with_kwargs=True))
        for (li, name) in self.lora:
            lin = get_module(model.model.layers[li], name)
            self.handles.append(lin.register_forward_hook(self._lin(li, name)))

    def params(self):
        out = {f"vec_L{j}": p for j, p in self.vecs.items()}
        for (li, name), (A, B) in self.lora.items():
            short = name.split(".")[-1]
            out[f"lora_L{li}_{short}.A"] = A
            out[f"lora_L{li}_{short}.B"] = B
        return out

    def _pre(self, j):
        def hook(module, args, kwargs):
            if self.mask is None or not self.write:
                return None
            h = args[0]
            add = self.mask.unsqueeze(-1).to(h.dtype) * self.vecs[j].to(h.dtype)
            return (h + add,) + tuple(args[1:]), kwargs
        return hook

    def _lin(self, li, name):
        def hook(module, inputs, output):
            if self.mask is None:
                return None
            x = inputs[0]
            A, B = self.lora[(li, name)]
            upd = (x.to(A.dtype) @ A.t()) @ B.t()
            return output + self.mask.unsqueeze(-1).to(output.dtype) * upd.to(output.dtype)
        return hook

    def zero_grad(self):
        for p in self.params().values():
            p.grad = None

    def grads(self):
        return {k: (p.grad.detach().float().cpu().clone() if p.grad is not None else None) for k, p in self.params().items()}


def rebind_recurrent_state(cache):
    """Make every linear-attention cache layer store its new recurrent state by rebinding."""
    n = 0
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states") and hasattr(layer, "update_recurrent_state"):
            def upd(self, recurrent_states, state_idx=0, **kwargs):
                if not self.is_recurrent_states_initialized[state_idx]:
                    self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
                self.recurrent_states[state_idx] = recurrent_states
                return self.recurrent_states[state_idx]
            layer.update_recurrent_state = types.MethodType(upd, layer)
            n += 1
    return n


def make_batch(tok, prompts, replies, device):
    """Left-pad the prefix (tokens 0..r) so r sits in one column; right-pad the rest."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    rows = []
    for p, r in zip(prompts, replies):
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ri = tok(r, add_special_tokens=False)["input_ids"] + [eot]
        cut = len(pi) - TAIL_K + READ_K + 1            # number of tokens in 0..r
        rows.append((pi[:cut], pi[cut:], ri))
    P = max(len(a) for a, _, _ in rows)
    S = max(len(b) + len(c) for _, b, c in rows)
    B = len(rows)
    pre_ids = torch.full((B, P), pad, dtype=torch.long)
    pre_mask = torch.zeros((B, P), dtype=torch.long)
    suf_ids = torch.full((B, S), pad, dtype=torch.long)
    suf_mask = torch.zeros((B, S), dtype=torch.long)
    labels = torch.full((B, S), -100, dtype=torch.long)   # label at suffix position t = token at t+1 when it is a reply token
    for i, (a, b, c) in enumerate(rows):
        pre_ids[i, P - len(a):] = torch.tensor(a); pre_mask[i, P - len(a):] = 1
        s = b + c
        suf_ids[i, :len(s)] = torch.tensor(s); suf_mask[i, :len(s)] = 1
        for t in range(len(b) - 1, len(s) - 1):           # from the last prompt token to the token before end-of-turn
            labels[i, t] = s[t + 1]
    info = {"B": B, "prefix_len": P, "suffix_len": S, "prefix_tokens": int(pre_mask.sum()), "suffix_tokens": int(suf_mask.sum()),
            "scored_positions": int((labels != -100).sum())}
    return [x.to(device) for x in (pre_ids, pre_mask, suf_ids, suf_mask, labels)], info


def loss_from_hidden(model, hidden, labels):
    sel = labels != -100
    logits = model.lm_head(hidden[sel]).float()
    return torch.nn.functional.cross_entropy(logits, labels[sel])


def set_mode(adds, mask, mode):
    adds.mask = None if mode == "orig" else mask
    adds.write = mode == "on"


def plain_loss(model, adds, batch, mode="on"):
    pre_ids, pre_mask, suf_ids, suf_mask, labels = batch
    ids = torch.cat([pre_ids, suf_ids], 1)
    am = torch.cat([pre_mask, suf_mask], 1)
    m = torch.cat([torch.zeros_like(pre_mask), suf_mask], 1).float()
    set_mode(adds, m, mode)
    out = model.model(input_ids=ids, attention_mask=am, use_cache=False)
    adds.mask = None
    return loss_from_hidden(model, out.last_hidden_state[:, pre_ids.shape[1]:], labels)


def prefix_cache(model, adds, batch):
    pre_ids, pre_mask = batch[0], batch[1]
    adds.mask = None
    with torch.no_grad():
        out = model.model(input_ids=pre_ids, attention_mask=pre_mask, use_cache=True)
    return out.past_key_values


def shared_loss(model, adds, batch, cache, patch, mode="on"):
    pre_ids, pre_mask, suf_ids, suf_mask, labels = batch
    c = copy.deepcopy(cache)
    if patch:
        rebind_recurrent_state(c)
    am = torch.cat([pre_mask, suf_mask], 1)
    set_mode(adds, suf_mask.float(), mode)
    out = model.model(input_ids=suf_ids, attention_mask=am, past_key_values=c, use_cache=True)
    adds.mask = None
    return loss_from_hidden(model, out.last_hidden_state, labels)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(args):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    name = f"{args.tag}_{args.dtype}"
    res = {"name": name, "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
           "gpu": torch.cuda.get_device_name(0)}
    try:
        import fla  # noqa: F401
        res["fla_importable"] = True
    except Exception:
        res["fla_importable"] = False
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation="sdpa").to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sample = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample16.json")))
    prompts, replies = build_prompts(tok, sample)
    tail = tok(prompts[0], add_special_tokens=False)["input_ids"][-TAIL_K:]
    res["tail_tokens"] = tok.convert_ids_to_tokens(tail)
    assert all(tok(p, add_special_tokens=False)["input_ids"][-TAIL_K:] == tail for p in prompts), "tail differs"
    adds = Additions(model, "cuda")
    cases = {"batch8": (prompts[:8], replies[:8]), "single": (prompts[8:9], replies[8:9])}
    res["cases"] = {}
    saved = {}
    for cname, (pp, rr) in cases.items():
        batch, info = make_batch(tok, pp, rr, "cuda")
        r = {"info": info}
        # plain form
        adds.zero_grad(); lp = plain_loss(model, adds, batch); lp.backward()
        g_plain = adds.grads(); r["plain_loss"] = float(lp)
        cache = prefix_cache(model, adds, batch)
        # shared form, unpatched
        try:
            adds.zero_grad(); ls = shared_loss(model, adds, batch, cache, patch=False); ls.backward()
            r["unpatched"] = {"ran": True, "loss": float(ls)}
            g_unp = adds.grads()
        except Exception as e:  # noqa: BLE001
            r["unpatched"] = {"ran": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}
            g_unp = None
        # shared form, patched; run it twice from two copies to show the copies are independent
        try:
            adds.zero_grad(); ls = shared_loss(model, adds, batch, cache, patch=True); ls.backward()
            g_sh = adds.grads()
            adds.zero_grad(); ls2 = shared_loss(model, adds, batch, cache, patch=True); ls2.backward()
            g_sh2 = adds.grads()
            r["patched"] = {"ran": True, "loss": float(ls), "loss_second_copy": float(ls2)}
        except Exception as e:  # noqa: BLE001
            r["patched"] = {"ran": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}
            g_sh = g_sh2 = None

        def cmp(a, b):
            out = {}
            for k in a:
                if a[k] is None or b is None or b[k] is None:
                    out[k] = None; continue
                x, y = a[k].flatten().double(), b[k].flatten().double()
                out[k] = {"cos": float(torch.nn.functional.cosine_similarity(x, y, dim=0)),
                          "rel": float((x - y).norm() / (x.norm() + 1e-30)), "norm_plain": float(x.norm())}
            return out
        r["shared_patched_vs_plain"] = cmp(g_plain, g_sh)
        r["shared_patched_copy2_vs_copy1"] = cmp(g_sh, g_sh2) if g_sh is not None else None
        r["shared_unpatched_vs_plain"] = cmp(g_plain, g_unp) if g_unp is not None else None
        if r["patched"].get("ran"):
            r["loss_rel_diff"] = abs(r["patched"]["loss"] - r["plain_loss"]) / abs(r["plain_loss"])
        # timing of a three-pass step: two gradient passes and one without gradient
        def step_plain():
            adds.zero_grad()
            with torch.no_grad():
                plain_loss(model, adds, batch, mode="orig")
            plain_loss(model, adds, batch, mode="off").backward()
            plain_loss(model, adds, batch, mode="on").backward()

        def step_shared():
            adds.zero_grad()
            c = prefix_cache(model, adds, batch)
            with torch.no_grad():
                shared_loss(model, adds, batch, c, patch=True, mode="orig")
            shared_loss(model, adds, batch, c, patch=True, mode="off").backward()
            shared_loss(model, adds, batch, c, patch=True, mode="on").backward()

        times = {}
        for label, fn in (("plain", step_plain), ("shared", step_shared)):
            if label == "shared" and not r["patched"].get("ran"):
                continue
            try:
                ts = []
                for _ in range(3):
                    torch.cuda.reset_peak_memory_stats(); sync(); t0 = time.time(); fn(); sync(); ts.append(time.time() - t0)
                times[label] = {"seconds_calls": [round(t, 4) for t in ts], "seconds_last": round(ts[-1], 4),
                                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            except Exception as e:  # noqa: BLE001
                times[label] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        r["three_pass_step"] = times
        res["cases"][cname] = r
        saved[cname] = {"plain": g_plain, "shared": g_sh}
        print(cname, json.dumps({k: r[k] for k in ("info", "plain_loss", "unpatched", "patched", "three_pass_step")}), flush=True)
    json.dump(res, open(os.path.join(args.out, f"{name}.json"), "w"), indent=1)
    torch.save(saved, os.path.join(args.out, f"{name}.pt"))
    print("WROTE", name, flush=True)


def compare(args):
    out = {"runs": {}, "summary": {}}
    for name in ("R_float32", "F_bfloat16", "F_float32"):
        p = os.path.join(args.out, f"{name}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p)); out["runs"][name] = r
        s = {}
        for cname, c in r["cases"].items():
            v = c.get("shared_patched_vs_plain") or {}
            vals = [x for x in v.values() if x]
            s[cname] = {"unpatched_ran": c["unpatched"]["ran"], "unpatched_error": c["unpatched"].get("error"),
                        "patched_ran": c["patched"]["ran"], "patched_error": c["patched"].get("error"),
                        "loss_rel_diff": c.get("loss_rel_diff"),
                        "min_cos": min((x["cos"] for x in vals), default=None),
                        "max_rel": max((x["rel"] for x in vals), default=None),
                        "three_pass_step": c.get("three_pass_step"), "info": c["info"]}
        out["summary"][name] = s
    json.dump(out, open(os.path.join(args.out, "prefix_check_result.json"), "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "compare"])
    ap.add_argument("--tag", default="R")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    run(a) if a.mode == "run" else compare(a)
