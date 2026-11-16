#!/usr/bin/env python3
"""Gradient parity of the fla Gated DeltaNet path against the PyTorch path, Qwen/Qwen3.5-4B.

Question: on this GPU, is the bfloat16 gradient that flows back through the Gated DeltaNet
layers correct when transformers uses flash-linear-attention (fla) kernels?

transformers picks the DeltaNet kernel when its Qwen3.5 module is first imported, so each
configuration runs in its own Python process:

    python grad_parity.py run --tag R --dtype float32  --out DIR   # before fla is installed (gold)
    python grad_parity.py run --tag R --dtype bfloat16 --out DIR   # before fla is installed
    pip install flash-linear-attention==0.6.0
    python grad_parity.py run --tag F --dtype bfloat16 --out DIR   # fla path
    python grad_parity.py run --tag F --dtype float32  --out DIR   # fla path
    python grad_parity.py compare --out DIR [--result PATH]

Each `run` writes DIR/<tag>_<dtype>.pt (gradients) and DIR/<tag>_<dtype>.json (metadata).
The caller redirects each run's stdout and stderr to DIR/<tag>_<dtype>.log; `compare` reads
those logs to check for the fallback warning of transformers.

The computation is the same in every run: frozen model, added float32 parameters (hidden-state
vectors and rank-8 updates on in_proj_qkv) applied to the second half of each sequence,
mean next-token cross-entropy over the last 100 positions, one backward pass.

Verdict rule, fixed in advance: the fla bfloat16 path is correct on this GPU if every cosine of
F-bfloat16 against R-float32 is at least 0.99 (all primary parameters, both input cases).
"""

import argparse
import json
import math
import os
import platform
import re
import sys
import time

MODEL_ID = "Qwen/Qwen3.5-4B"
SEED = 20261115
VEC_LAYERS = [3, 6, 9, 15, 23]
LORA_LAYERS = [4, 20]
DIAG_LAYER = 0  # diagnostic probe only; B = 0 so the forward pass is unchanged; not in the verdict
RANK = 8
INIT_SCALE = 0.01
SHORT_BATCH, SHORT_LEN = 4, 400
LONG_LEN = 1500
N_LOSS = 100
THRESHOLD = 0.99
FALLBACK_MARK = "falling back to its reference PyTorch implementation"
RUNS = [("R", "float32"), ("R", "bfloat16"), ("F", "bfloat16"), ("F", "float32")]
GOLD = ("R", "float32")
COMPARED = [("R", "bfloat16"), ("F", "bfloat16"), ("F", "float32")]
# fla Triton kernel names seen in a profile are the direct evidence that fla runs.
FLA_KERNEL_RE = re.compile(
    r"(gated_delta|delta_rule|solve_tril|kkt|recompute_w_u|fwd_kernel_h|bwd_kernel_dhu|"
    r"merge_16x16|l2norm|layer_norm_gated|chunk_local_cumsum|chunk_fwd_kernel_o|chunk_bwd_kernel)",
    re.IGNORECASE,
)

# Public-domain text: the United States Declaration of Independence (1776), followed by
# Abraham Lincoln's Gettysburg Address (1863) and Second Inaugural Address (1865).
TEXT = """When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.

We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness. That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, That whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to abolish it, and to institute new Government, laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their Safety and Happiness. Prudence, indeed, will dictate that Governments long established should not be changed for light and transient causes; and accordingly all experience hath shewn, that mankind are more disposed to suffer, while evils are sufferable, than to right themselves by abolishing the forms to which they are accustomed. But when a long train of abuses and usurpations, pursuing invariably the same Object evinces a design to reduce them under absolute Despotism, it is their right, it is their duty, to throw off such Government, and to provide new Guards for their future security. Such has been the patient sufferance of these Colonies; and such is now the necessity which constrains them to alter their former Systems of Government. The history of the present King of Great Britain is a history of repeated injuries and usurpations, all having in direct object the establishment of an absolute Tyranny over these States. To prove this, let Facts be submitted to a candid world.

He has refused his Assent to Laws, the most wholesome and necessary for the public good.

He has forbidden his Governors to pass Laws of immediate and pressing importance, unless suspended in their operation till his Assent should be obtained; and when so suspended, he has utterly neglected to attend to them.

He has refused to pass other Laws for the accommodation of large districts of people, unless those people would relinquish the right of Representation in the Legislature, a right inestimable to them and formidable to tyrants only.

He has called together legislative bodies at places unusual, uncomfortable, and distant from the depository of their public Records, for the sole purpose of fatiguing them into compliance with his measures.

He has dissolved Representative Houses repeatedly, for opposing with manly firmness his invasions on the rights of the people.

He has refused for a long time, after such dissolutions, to cause others to be elected; whereby the Legislative powers, incapable of Annihilation, have returned to the People at large for their exercise; the State remaining in the mean time exposed to all the dangers of invasion from without, and convulsions within.

He has endeavoured to prevent the population of these States; for that purpose obstructing the Laws for Naturalization of Foreigners; refusing to pass others to encourage their migrations hither, and raising the conditions of new Appropriations of Lands.

He has obstructed the Administration of Justice, by refusing his Assent to Laws for establishing Judiciary powers.

He has made Judges dependent on his Will alone, for the tenure of their offices, and the amount and payment of their salaries.

He has erected a multitude of New Offices, and sent hither swarms of Officers to harrass our people, and eat out their substance.

He has kept among us, in times of peace, Standing Armies without the Consent of our legislatures.

He has affected to render the Military independent of and superior to the Civil power.

He has combined with others to subject us to a jurisdiction foreign to our constitution, and unacknowledged by our laws; giving his Assent to their Acts of pretended Legislation:

For Quartering large bodies of armed troops among us:

For protecting them, by a mock Trial, from punishment for any Murders which they should commit on the Inhabitants of these States:

For cutting off our Trade with all parts of the world:

For imposing Taxes on us without our Consent:

For depriving us in many cases, of the benefits of Trial by Jury:

For transporting us beyond Seas to be tried for pretended offences:

For abolishing the free System of English Laws in a neighbouring Province, establishing therein an Arbitrary government, and enlarging its Boundaries so as to render it at once an example and fit instrument for introducing the same absolute rule into these Colonies:

For taking away our Charters, abolishing our most valuable Laws, and altering fundamentally the Forms of our Governments:

For suspending our own Legislatures, and declaring themselves invested with power to legislate for us in all cases whatsoever.

He has abdicated Government here, by declaring us out of his Protection and waging War against us.

He has plundered our seas, ravaged our Coasts, burnt our towns, and destroyed the lives of our people.

He is at this time transporting large Armies of foreign Mercenaries to compleat the works of death, desolation and tyranny, already begun with circumstances of Cruelty and perfidy scarcely paralleled in the most barbarous ages, and totally unworthy the Head of a civilized nation.

He has constrained our fellow Citizens taken Captive on the high Seas to bear Arms against their Country, to become the executioners of their friends and Brethren, or to fall themselves by their Hands.

He has excited domestic insurrections amongst us, and has endeavoured to bring on the inhabitants of our frontiers, the merciless Indian Savages, whose known rule of warfare, is an undistinguished destruction of all ages, sexes and conditions.

In every stage of these Oppressions We have Petitioned for Redress in the most humble terms: Our repeated Petitions have been answered only by repeated injury. A Prince whose character is thus marked by every act which may define a Tyrant, is unfit to be the ruler of a free people.

Nor have We been wanting in attentions to our Brittish brethren. We have warned them from time to time of attempts by their legislature to extend an unwarrantable jurisdiction over us. We have reminded them of the circumstances of our emigration and settlement here. We have appealed to their native justice and magnanimity, and we have conjured them by the ties of our common kindred to disavow these usurpations, which, would inevitably interrupt our connections and correspondence. They too have been deaf to the voice of justice and of consanguinity. We must, therefore, acquiesce in the necessity, which denounces our Separation, and hold them, as we hold the rest of mankind, Enemies in War, in Peace Friends.

We, therefore, the Representatives of the united States of America, in General Congress, Assembled, appealing to the Supreme Judge of the world for the rectitude of our intentions, do, in the Name, and by Authority of the good People of these Colonies, solemnly publish and declare, That these United Colonies are, and of Right ought to be Free and Independent States; that they are Absolved from all Allegiance to the British Crown, and that all political connection between them and the State of Great Britain, is and ought to be totally dissolved; and that as Free and Independent States, they have full Power to levy War, conclude Peace, contract Alliances, establish Commerce, and to do all other Acts and Things which Independent States may of right do. And for the support of this Declaration, with a firm reliance on the protection of divine Providence, we mutually pledge to each other our Lives, our Fortunes and our sacred Honor.

Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.

Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.

But, in a larger sense, we can not dedicate, we can not consecrate, we can not hallow this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us, that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion, that we here highly resolve that these dead shall not have died in vain, that this nation, under God, shall have a new birth of freedom, and that government of the people, by the people, for the people, shall not perish from the earth.

Fellow-Countrymen: At this second appearing to take the oath of the Presidential office there is less occasion for an extended address than there was at the first. Then a statement somewhat in detail of a course to be pursued seemed fitting and proper. Now, at the expiration of four years, during which public declarations have been constantly called forth on every point and phase of the great contest which still absorbs the attention and engrosses the energies of the nation, little that is new could be presented. The progress of our arms, upon which all else chiefly depends, is as well known to the public as to myself, and it is, I trust, reasonably satisfactory and encouraging to all. With high hope for the future, no prediction in regard to it is ventured.

On the occasion corresponding to this four years ago all thoughts were anxiously directed to an impending civil war. All dreaded it, all sought to avert it. While the inaugural address was being delivered from this place, devoted altogether to saving the Union without war, insurgent agents were in the city seeking to destroy it without war, seeking to dissolve the Union and divide effects by negotiation. Both parties deprecated war, but one of them would make war rather than let the nation survive, and the other would accept war rather than let it perish, and the war came.

One-eighth of the whole population were colored slaves, not distributed generally over the Union, but localized in the southern part of it. These slaves constituted a peculiar and powerful interest. All knew that this interest was somehow the cause of the war. To strengthen, perpetuate, and extend this interest was the object for which the insurgents would rend the Union even by war, while the Government claimed no right to do more than to restrict the territorial enlargement of it. Neither party expected for the war the magnitude or the duration which it has already attained. Neither anticipated that the cause of the conflict might cease with or even before the conflict itself should cease. Each looked for an easier triumph, and a result less fundamental and astounding. Both read the same Bible and pray to the same God, and each invokes His aid against the other. It may seem strange that any men should dare to ask a just God's assistance in wringing their bread from the sweat of other men's faces, but let us judge not, that we be not judged. The prayers of both could not be answered. That of neither has been answered fully. The Almighty has His own purposes.

With malice toward none, with charity for all, with firmness in the right as God gives us to see the right, let us strive on to finish the work we are in, to bind up the nation's wounds, to care for him who shall have borne the battle and for his widow and his orphan, to do all which may achieve and cherish a just and lasting peace among ourselves and with all nations.
"""


# ----------------------------------------------------------------------------------------------
# run: one configuration in this process
# ----------------------------------------------------------------------------------------------

def _versions():
    from importlib import metadata

    out = {"python": platform.python_version()}
    for dist in ["torch", "transformers", "triton", "flash-linear-attention", "fla-core",
                 "causal-conv1d", "accelerate", "safetensors"]:
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def _get_layers(model):
    for path in ("model.layers", "model.language_model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return path, obj
        except AttributeError:
            continue
    raise RuntimeError("no decoder layer list found on the model")


def _second_half_mask(t, like):
    import torch

    m = torch.zeros(1, t, 1, device=like.device, dtype=like.dtype)
    m[:, t // 2:] = 1
    return m


def cmd_run(args):
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    name = f"{args.tag}_{args.dtype}"
    meta = {"tag": args.tag, "dtype": args.dtype, "versions": _versions(),
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cuda_runtime": torch.version.cuda,
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}
    print(f"[{name}] versions {meta['versions']} gpu {meta['gpu']} cc {meta['capability']}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok(TEXT, add_special_tokens=False).input_ids
    meta["text_tokens"] = len(ids)
    need = max(SHORT_BATCH * SHORT_LEN, LONG_LEN)
    if len(ids) < need:
        raise SystemExit(f"text has {len(ids)} tokens, need {need}")
    cases = {
        "short_4x400": torch.tensor([ids[i * SHORT_LEN:(i + 1) * SHORT_LEN] for i in range(SHORT_BATCH)],
                                    device="cuda"),
        "long_1x1500": torch.tensor([ids[:LONG_LEN]], device="cuda"),
    }

    t0 = time.perf_counter()
    model, info = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, attn_implementation="sdpa", device_map="cuda", output_loading_info=True)
    model.eval()
    meta["load_seconds"] = round(time.perf_counter() - t0, 1)
    meta["model_class"] = type(model).__name__
    meta["missing_keys"] = sorted(info.get("missing_keys", []))[:20]
    meta["n_missing_keys"] = len(info.get("missing_keys", []))
    meta["n_unexpected_keys"] = len(info.get("unexpected_keys", []))
    if any(".layers." in k for k in info.get("missing_keys", [])):
        raise SystemExit(f"decoder weights missing after load: {meta['missing_keys']}")
    for p in model.parameters():
        p.requires_grad_(False)

    layers_path, layers = _get_layers(model)
    meta["layers_path"] = layers_path
    meta["n_layers"] = len(layers)
    text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    hidden = text_cfg.hidden_size
    meta["hidden_size"] = hidden
    meta["layer_types"] = list(getattr(text_cfg, "layer_types", []) or [])
    meta["layer_kind"] = {i: ("linear_attn" if hasattr(layers[i], "linear_attn") else "self_attn")
                          for i in sorted(set(VEC_LAYERS + LORA_LAYERS + [DIAG_LAYER]))}
    for i in LORA_LAYERS + [DIAG_LAYER]:
        if not hasattr(layers[i], "linear_attn"):
            raise SystemExit(f"layer {i} is not a Gated DeltaNet layer")

    gdn = layers[LORA_LAYERS[0]].linear_attn
    fn = getattr(gdn, "chunk_gated_delta_rule", None)
    meta["gdn_class"] = type(gdn).__name__
    meta["gdn_chunk_fn"] = {"repr": repr(fn)[:200], "module": getattr(fn, "__module__", None),
                            "qualname": getattr(fn, "__qualname__", None),
                            "has___wrapped__": hasattr(fn, "__wrapped__")}
    meta["fla_imported"] = "fla" in sys.modules
    meta["causal_conv1d_imported"] = "causal_conv1d" in sys.modules
    print(f"[{name}] model {meta['model_class']} layers at {layers_path} ({len(layers)}) hidden {hidden} "
          f"gdn fn {meta['gdn_chunk_fn']} fla imported {meta['fla_imported']}", flush=True)

    # Added parameters, float32, fixed seed, created on the CPU so every run gets the same values.
    gen = torch.Generator().manual_seed(SEED)
    params = {}
    for i in VEC_LAYERS:
        params[f"vec_L{i}"] = torch.zeros(hidden)
    for i in LORA_LAYERS:
        lin = layers[i].linear_attn.in_proj_qkv
        params[f"lora_L{i}.A"] = torch.randn(RANK, lin.in_features, generator=gen) * INIT_SCALE
        params[f"lora_L{i}.B"] = torch.randn(lin.out_features, RANK, generator=gen) * INIT_SCALE
    lin0 = layers[DIAG_LAYER].linear_attn.in_proj_qkv
    diag_A = (torch.randn(RANK, lin0.in_features, generator=gen) * INIT_SCALE).cuda()
    params[f"diag_lora_L{DIAG_LAYER}.B"] = torch.zeros(lin0.out_features, RANK)
    params = {k: v.cuda().requires_grad_(True) for k, v in params.items()}
    meta["params"] = {k: list(v.shape) for k, v in params.items()}
    hits = {"vec": 0, "lora": 0}

    def vec_hook(v):
        def hook(mod, args_, kwargs):
            if args_:
                h = args_[0]
            else:
                h = kwargs["hidden_states"]
            h2 = h + _second_half_mask(h.shape[1], h) * v.to(h.dtype)
            hits["vec"] += 1
            if args_:
                return (h2,) + tuple(args_[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = h2
            return args_, kwargs
        return hook

    def lora_hook(A, B):
        def hook(mod, inp, out):
            x = inp[0]
            delta = F.linear(F.linear(x.to(torch.float32), A), B)
            hits["lora"] += 1
            return out + _second_half_mask(x.shape[1], out) * delta.to(out.dtype)
        return hook

    for i in VEC_LAYERS:
        layers[i].register_forward_pre_hook(vec_hook(params[f"vec_L{i}"]), with_kwargs=True)
    for i in LORA_LAYERS:
        layers[i].linear_attn.in_proj_qkv.register_forward_hook(
            lora_hook(params[f"lora_L{i}.A"], params[f"lora_L{i}.B"]))
    layers[DIAG_LAYER].linear_attn.in_proj_qkv.register_forward_hook(
        lora_hook(diag_A, params[f"diag_lora_L{DIAG_LAYER}.B"]))

    def loss_of(x):
        out = model(input_ids=x, use_cache=False, logits_to_keep=N_LOSS + 1)
        logits = out.logits[:, :-1].float()  # position p predicts token p + 1
        tgt = x[:, -N_LOSS:]
        assert logits.shape[1] == N_LOSS
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))

    def step(x):
        for p in params.values():
            p.grad = None
        torch.cuda.synchronize()
        t = time.perf_counter()
        loss = loss_of(x)
        loss.backward()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        return loss.item(), dt, {k: p.grad.detach().to(torch.float32).cpu().clone() for k, p in params.items()}

    def cos(a, b):
        a, b = a.double().flatten(), b.double().flatten()
        d = a.norm() * b.norm()
        return float((a @ b) / d) if d > 0 else float("nan")

    grads, meta["cases"] = {}, {}
    for cname, x in cases.items():
        torch.cuda.reset_peak_memory_stats()
        hits["vec"] = hits["lora"] = 0
        loss1, dt1, g1 = step(x)  # warm-up (also compiles Triton kernels)
        if hits["vec"] != len(VEC_LAYERS) or hits["lora"] != len(LORA_LAYERS) + 1:
            raise SystemExit(f"hooks fired {hits}, expected vec {len(VEC_LAYERS)} lora {len(LORA_LAYERS) + 1}")
        loss2, dt2, g2 = step(x)  # timed call
        grads[cname] = g2
        meta["cases"][cname] = {
            "shape": list(x.shape), "loss_call1": loss1, "loss_call2": loss2,
            "fwd_bwd_seconds_call1": round(dt1, 4), "fwd_bwd_seconds_call2": round(dt2, 4),
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "call1_vs_call2_cosine": {k: cos(g1[k], g2[k]) for k in g2},
            "grad_norms": {k: float(g2[k].double().norm()) for k in g2},
        }
        print(f"[{name}] {cname} loss {loss2:.6f} fwd+bwd {dt2:.4f}s (warm-up {dt1:.2f}s) "
              f"peak {meta['cases'][cname]['peak_mem_gib']} GiB", flush=True)

    # Profile one more fwd+bwd on the short case: fla Triton kernel names prove the fla path ran.
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step(cases["short_4x400"])
    names = sorted({e.key for e in prof.key_averages()})
    fla_like = [n for n in names if FLA_KERNEL_RE.search(n) and "at::" not in n]
    meta["profiled_kernel_count"] = len(names)
    meta["fla_like_kernels"] = fla_like[:60]
    meta["n_fla_like_kernels"] = len(fla_like)
    meta["fla_imported_after_run"] = "fla" in sys.modules
    print(f"[{name}] fla-like kernels {len(fla_like)}: {fla_like[:12]}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save(grads, os.path.join(args.out, f"{name}.pt"))
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[{name}] saved", flush=True)


# ----------------------------------------------------------------------------------------------
# compare: all four runs against the gold reference
# ----------------------------------------------------------------------------------------------

def cmd_compare(args):
    import torch

    def load(tag, dt):
        name = f"{tag}_{dt}"
        g = torch.load(os.path.join(args.out, f"{name}.pt"))
        with open(os.path.join(args.out, f"{name}.json")) as f:
            m = json.load(f)
        log_path = os.path.join(args.out, f"{name}.log")
        log = open(log_path, errors="replace").read() if os.path.exists(log_path) else ""
        return g, m, log

    data = {f"{t}_{d}": load(t, d) for t, d in RUNS}
    gold = data["_".join(GOLD)][0]

    def stats(a, b):
        a, b = a.double().flatten(), b.double().flatten()
        na, nb = float(a.norm()), float(b.norm())
        c = float((a @ b) / (na * nb)) if na > 0 and nb > 0 else float("nan")
        return {"cosine": c, "norm_ratio": na / nb if nb > 0 else float("nan"),
                "rel_err": float((a - b).norm()) / nb if nb > 0 else float("nan")}

    result = {
        "question": "Is the fla 0.6.0 bfloat16 gradient through the Gated DeltaNet layers of "
                    f"{MODEL_ID} correct on this GPU?",
        "rule": f"fla bfloat16 path correct if every F_bfloat16 cosine vs R_float32 >= {THRESHOLD} "
                "(primary parameters, both input cases; diag_* probes excluded)",
        "setup": {"model": MODEL_ID, "seed": SEED, "vec_layers": VEC_LAYERS, "lora_layers": LORA_LAYERS,
                  "lora_rank": RANK, "init_scale": INIT_SCALE, "diag_probe": f"rank-{RANK} update on layer "
                  f"{DIAG_LAYER} in_proj_qkv with B=0 (forward unchanged), gradient of B only; not in the verdict",
                  "added_to": "second half of each sequence", "loss": f"mean next-token CE, last {N_LOSS} positions",
                  "cases": {"short_4x400": [SHORT_BATCH, SHORT_LEN], "long_1x1500": [1, LONG_LEN]}},
        "runs": {}, "comparisons": {}, "summary": {},
    }

    for key, (g, m, log) in data.items():
        result["runs"][key] = {
            "versions": m["versions"], "gpu": m["gpu"], "capability": m["capability"],
            "cuda_runtime": m["cuda_runtime"], "model_class": m["model_class"],
            "layers_path": m["layers_path"], "layer_kind": m["layer_kind"],
            "layer_types_first8": m["layer_types"][:8],
            "gdn_chunk_fn": m["gdn_chunk_fn"], "fla_imported": m["fla_imported_after_run"],
            "n_fla_like_kernels": m["n_fla_like_kernels"], "fla_like_kernels": m["fla_like_kernels"][:20],
            # transformers also prints this phrase for `causal_conv1d_fn` (causal-conv1d is absent in
            # every run), so only the line that names the DeltaNet kernel counts.
            "fallback_warning_in_log": any(FALLBACK_MARK.lower() in ln.lower() and "chunk_gated_delta_rule" in ln
                                           for ln in log.splitlines()),
            "any_falling_back_in_log": [ln.strip()[:300] for ln in log.splitlines() if "falling back" in ln.lower()][:5],
            "allow_tf32_matmul": m["allow_tf32_matmul"], "n_missing_keys": m["n_missing_keys"],
            "text_tokens": m["text_tokens"],
            "cases": {c: {k: v for k, v in cm.items() if k != "grad_norms"} for c, cm in m["cases"].items()},
            "fwd_bwd_seconds": {c: cm["fwd_bwd_seconds_call2"] for c, cm in m["cases"].items()},
        }

    checks = {
        "R_has_fallback_warning": all(result["runs"][f"R_{d}"]["fallback_warning_in_log"] for d in ("float32", "bfloat16")),
        "F_has_no_fallback_warning": not any(result["runs"][f"F_{d}"]["fallback_warning_in_log"] for d in ("float32", "bfloat16")),
        "R_no_fla_kernels": all(result["runs"][f"R_{d}"]["n_fla_like_kernels"] == 0 for d in ("float32", "bfloat16")),
        "F_has_fla_kernels": all(result["runs"][f"F_{d}"]["n_fla_like_kernels"] > 0 for d in ("float32", "bfloat16")),
    }
    result["path_checks"] = checks

    mins = {}
    for t, d in COMPARED + [("F", "bfloat16_vs_R_bfloat16")]:
        key = f"{t}_{d}"
        if d == "bfloat16_vs_R_bfloat16":
            g, ref = data["F_bfloat16"][0], data["R_bfloat16"][0]
        else:
            g, ref = data[key][0], gold
        result["comparisons"][key] = {}
        mins[key] = {}
        for case in ref:
            per = {p: stats(g[case][p], ref[case][p]) for p in ref[case]}
            result["comparisons"][key][case] = per
            prim = {p: s for p, s in per.items() if not p.startswith("diag_")}
            worst = min(prim, key=lambda p: prim[p]["cosine"])
            mins[key][case] = {"min_cosine": prim[worst]["cosine"], "param": worst,
                               "diag_cosine": {p: s["cosine"] for p, s in per.items() if p.startswith("diag_")}}

    def passes(key):
        vals = [mins[key][c]["min_cosine"] for c in mins[key]]
        return all(v >= THRESHOLD for v in vals if not math.isnan(v)) and not any(math.isnan(v) for v in vals)

    fb = mins["F_bfloat16"]
    worst_case = min(fb, key=lambda c: fb[c]["min_cosine"])
    result["summary"] = {
        "min_cosines": mins,
        "pass_rule": {k: passes(k) for k in ("R_bfloat16", "F_bfloat16", "F_float32")},
        "F_bfloat16_min": {"cosine": fb[worst_case]["min_cosine"], "case": worst_case,
                           "param": fb[worst_case]["param"]},
        "fwd_bwd_seconds": {k: r["fwd_bwd_seconds"] for k, r in result["runs"].items()},
        "verdict": ("CORRECT" if passes("F_bfloat16") else "WRONG")
                   + f": fla bfloat16 gradients on {data['F_bfloat16'][1]['gpu']}",
        "path_checks_all_pass": all(checks.values()),
    }
    out_path = args.result or os.path.join(args.out, "grad_parity_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["summary"], indent=1))
    print(json.dumps(checks, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True, choices=["R", "F"])
    r.add_argument("--dtype", required=True, choices=["float32", "bfloat16"])
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    c.add_argument("--result", default=None)
    args = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
