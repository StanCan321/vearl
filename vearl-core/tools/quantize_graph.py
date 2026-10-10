#!/usr/bin/env python3
"""int8 quantisation of BRANCHED convolutional networks (residual blocks, 1x1 shortcuts) into a verifiable VEARL program (docs/12 section 7).

The protocol's `Add` operations require tensors of the SAME SCALE. Here, all the tensors linked by additions form a GROUP that shares a
single scale (the one that covers the real range of the group at the 99.9th percentile); each convolution that produces a member of the group is set to that
TARGET scale by adjusting the scale of its weights (integer shift, weights >= natural scale: no clipping of the weights). No
rescaling operation, no floating point in the program.

IR: list of nodes {"name", "op": input|conv|linear|pool|flatten|add, "src": [...], ...} in topological order.
    conv   : W (co, ci, kh, kw), b, pad, stride, relu       linear : W (out, in), b, relu
    pool   : mode ("max"|"avg"), k                            flatten, add (relu)
"""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vearl_ref as R
from quantize_cnn import _conv_float, _pool_float


WPCT = float(os.environ.get("QG_WPCT", 100))                              # percentile of |W| that fixes the weight scale (100: the maximum)
BIASCORR = os.environ.get("QG_BIASCORR", "0") == "1"                       # (trial: compensate the mean bias -1/2 step; DEGRADES the measured accuracy, disabled)
PCT = float(os.environ.get("QG_PCT", 99.9))                                # percentile that fixes the range of a tensor (99.9: 0.1 % of the values are clipped)


def p999(x):
    return float(np.percentile(np.abs(x), PCT)) if np.size(x) else 0.0


# ---------------------------------------------------------------- floating point
def float_forward(nodes, x):
    """Real quantities per node: out (after ReLU), and for conv / linear: raw (before bias), pre (before ReLU)."""
    v, stat = {}, {}
    for nd in nodes:
        op, name = nd["op"], nd["name"]
        if op == "input":
            v[name] = x
            continue
        a = v[nd["src"][0]]
        if op == "conv":
            raw = _conv_float(a, nd["W"], np.zeros(nd["W"].shape[0]), nd["pad"], nd["stride"])
            pre = raw + nd["b"][None, :, None, None]
            stat[name] = dict(raw=raw, pre=pre)
            v[name] = np.maximum(pre, 0) if nd["relu"] else pre
        elif op == "linear":
            a2 = a.reshape(len(a), -1)
            raw = a2 @ nd["W"].T
            pre = raw + nd["b"]
            stat[name] = dict(raw=raw, pre=pre)
            v[name] = np.maximum(pre, 0) if nd["relu"] else pre
        elif op == "pool":
            v[name] = _pool_float(a, nd["mode"], nd["k"])
        elif op == "flatten":
            v[name] = a.reshape(len(a), -1)
        elif op == "add":
            s = a + v[nd["src"][1]]
            v[name] = np.maximum(s, 0) if nd["relu"] else s
        else:
            raise ValueError(op)
    return v, stat


# ---------------------------------------------------------------- scale groups
def groups_of(nodes):
    parent = {nd["name"]: nd["name"] for nd in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for nd in nodes:
        if nd["op"] == "add":
            for s in nd["src"]:
                parent[find(s)] = find(nd["name"])
    return {n: find(n) for n in parent}


def build(nodes, vals, stats, n, B, shape, gscale):
    """(ops, weights, info). gscale: scale imposed per group. Also returns, per group, the largest NATURAL scale encountered."""
    grp = groups_of(nodes)
    ops, W, sc, ix, geo, table = [], [], {}, {}, {}, []
    natural = {}
    clipped = [0]
    c0, h0, w0 = shape

    def add_weight(m2d, sw):
        blk = np.zeros((n, n), np.int64)
        m = np.clip(np.round(m2d / sw), -63, 63).astype(np.int64)
        blk[:m.shape[0], :m.shape[1]] = m
        W.append(blk)
        return len(W) - 1

    plan = {}
    err = {}                                                                 # mean error (in steps of the tensor scale): -1/2 after an arithmetic shift
    tb_index = [None]                                                       # bias table block (created at the end, reserved index)
    bias_rows = []

    def gemm_block(nd, x_idx, s_in, W2d, bvec, relu, group_target):
        sw_nat = np.percentile(np.abs(W2d), WPCT) / 63.0
        st = stats[nd["name"]]
        s_need = max(p999(st["raw"]), p999(st["pre"])) / 63
        if group_target is not None:
            sh = math.floor(math.log2(group_target / (s_in * sw_nat)))
            if sh < 0:
                sh = 0
                natural[grp[nd["name"]]] = max(natural.get(grp[nd["name"]], 0), s_in * sw_nat)
                sw_used, s_out = sw_nat, s_in * sw_nat
            else:
                sw_used, s_out = group_target / (s_in * 2.0 ** sh), group_target
        else:
            sh = max(0, math.ceil(math.log2(s_need / (s_in * sw_nat)))) if s_need > s_in * sw_nat else 0
            sw_used, s_out = sw_nat, s_in * sw_nat * 2.0 ** sh
        if BIASCORR:                                                        # the mean error of the input goes through the layer: we subtract it in the bias
            bvec = bvec - err.get(nd["src"][0], 0.0) * s_in * W2d.sum(1)
        err[nd["name"]] = -0.5 if sh > 0 else 0.0
        plan[nd["name"]] = dict(sh=sh, sw=sw_used, s_in=s_in, s_out=s_out)
        has_bias = np.abs(bvec).max() > 0
        ops.append({"Gemm": {"w": add_weight(W2d, sw_used), "x": x_idx, "shift": sh, "relu": bool(relu and not has_bias)}})
        cur = len(ops)
        if has_bias:
            b = np.zeros(n)
            b[:len(bvec)] = bvec
            bi = np.round(b / s_out)
            clipped[0] += int((np.abs(bi) > 64).sum())
            bias_rows.append(np.clip(bi, -64, 64).astype(np.int64))
            ops.append({"Bias": {"x": cur, "w": -1, "row": len(bias_rows) - 1, "relu": bool(relu)}})    # w = -1: table block, fixed at the end
            cur = len(ops)
        return cur, s_out

    for nd in nodes:
        op, name = nd["op"], nd["name"]
        if op == "input":
            ix[name], sc[name], geo[name] = 0, 1.0 / 64, (c0, h0, w0)
            continue
        src = nd["src"][0]
        c, h, w = geo[src]
        gt = gscale.get(grp[name])                                          # TARGET scale if the node belongs to a residual group
        if op == "conv":
            co, ci, kh, kw = nd["W"].shape
            pad, stride = nd["pad"], nd["stride"]
            if stride == 1 and pad == kh // 2 and kh == kw:
                ops.append({"Map": {"x": ix[src], "map": {"Im2col": {"c_in": c, "h": h, "w": w, "batch": B, "kh": kh, "kw": kw, "pad": pad}}}})
                oh, ow = h, w
            else:
                ops.append({"Map": {"x": ix[src], "map": {"Conv": {"c_in": c, "h": h, "w": w, "batch": B, "kh": kh, "kw": kw, "pad": pad, "stride": stride}}}})
                oh, ow = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
            cur, s_out = gemm_block(nd, len(ops), sc[src], nd["W"].reshape(co, -1), nd["b"], nd["relu"], gt)
            ix[name], sc[name], geo[name] = cur, s_out, (co, oh, ow)
        elif op == "linear":
            W2d = nd["W"]
            cur, s_out = gemm_block(nd, ix[src], sc[src], W2d, nd["b"], nd["relu"], gt)
            ix[name], sc[name], geo[name] = cur, s_out, (W2d.shape[0], 1, 1)
        elif op == "pool":
            err[name] = err.get(src, 0.0)
            k = nd["k"]
            ops.append({"Pool": {"x": ix[src], "pool": {"Max" if nd["mode"] == "max" else "Avg": {"c": c, "h": h, "w": w, "batch": B, "k": k}}}})
            ix[name], sc[name], geo[name] = len(ops), sc[src], (c, h // k, w // k)
        elif op == "flatten":
            err[name] = err.get(src, 0.0)
            ops.append({"Map": {"x": ix[src], "map": {"Flatten": {"c": c, "h": h, "w": w, "batch": B}}}})
            ix[name], sc[name], geo[name] = len(ops), sc[src], (c * h * w, 1, 1)
        elif op == "add":
            a, b2 = nd["src"]
            err[name] = err.get(a, 0.0) + err.get(b2, 0.0)
            assert abs(sc[a] - sc[b2]) <= 1e-9 * sc[a], f"add {name}: different scales {sc[a]} / {sc[b2]}"
            ops.append({"Add": {"a": ix[a], "b": ix[b2], "relu": bool(nd["relu"])}})
            ix[name], sc[name], geo[name] = len(ops), sc[a], geo[a]
    # bias table: one weight block at the end; the Bias operations are renumbered
    if bias_rows:
        tb = np.zeros((n, n), np.int64)
        for r, row in enumerate(bias_rows):
            tb[r, :len(row)] = row
        W.append(tb)
        for o in ops:
            if "Bias" in o:
                o["Bias"]["w"] = len(W) - 1
    return ops, W, dict(natural=natural, clipped=clipped[0], sc=sc, geo=geo, plan=plan)


def quantize_graph(nodes, calib, n, labels=None, shape=None, verbose=True, max_samples=4, return_info=False):
    shape = shape or calib.shape[1:]
    c0, h0, w0 = shape
    B = n // (h0 * w0)
    if B < 1:
        raise ValueError("an image does not fit in n columns")
    R.N = n
    try:                                                                   # exact integer products by torch (float64): ~100x faster than numpy
        import torch
        R._mm = lambda a, b: torch.round(torch.from_numpy(np.asarray(a, dtype=np.float64)) @ torch.from_numpy(np.asarray(b, dtype=np.float64))).numpy().astype(np.int64)
    except ImportError:
        pass
    sub = calib[:256]
    vals, stats = float_forward(nodes, sub)
    grp = groups_of(nodes)
    # required scale per group: real range of ALL the members
    need = {}
    for nd in nodes:
        if nd["op"] == "input":
            continue
        g = grp[nd["name"]]
        m = p999(vals[nd["name"]])
        if nd["name"] in stats:
            m = max(m, p999(stats[nd["name"]]["raw"]), p999(stats[nd["name"]]["pre"]))
        need[g] = max(need.get(g, 0), m / 63)
    grouped = {grp[nd["name"]] for nd in nodes if nd["op"] == "add"}
    gscale = {g: need[g] for g in grouped}
    for it in range(3):                                                    # natural scales > target: we raise the target and start again
        ops, W, info = build(nodes, vals, stats, n, B, shape, gscale)
        bad = {g: s for g, s in info["natural"].items() if s > gscale.get(g, 0)}
        if not bad:
            break
        for g, s in bad.items():
            gscale[g] = s * 1.0001
    report = {"ops": len(ops), "weights": len(W), "batch": B, "groups": len(grouped), "clipped_biases": info["clipped"]}
    if verbose:
        print(f"program: {len(ops)} operations of which {sum(1 for o in ops if 'Gemm' in o)} Gemm, {len(W)} weight blocks; {len(grouped)} scale groups (residuals); batch of {B} image(s) per job"
              + (f" ; {info['clipped']} saturated biases" if info["clipped"] else ""))

    def inp(imgs):
        t = np.zeros((n, n), np.int64)
        arr = np.round(imgs * 64).astype(np.int64)
        for bb in range(len(imgs)):
            t[:c0, np.arange(h0 * w0) * B + bb] = arr[bb].reshape(c0, -1)
        return t

    last = nodes[-1]["name"]
    classes = geo_rows = info["geo"][last][0]
    if labels is not None:
        pred, m = [], 0
        for i in range(0, len(calib) - len(calib) % B, B):
            T, _ = R.run_program(inp(calib[i:i + B]), W, ops)
            pred += list(T[-1][:classes, :B].argmax(0))
        m = len(pred)
        fv, _ = float_forward(nodes, calib[:m])
        fp = fv[last].reshape(m, -1).argmax(1)
        pred = np.array(pred)
        report.update(float_acc=float((fp == labels[:m]).mean()), int8_acc=float((pred == labels[:m]).mean()), agreement=float((pred == fp).mean()), images=m)
        if verbose:
            print(f"FLOAT {100 * report['float_acc']:.2f} % | INT8 {100 * report['int8_acc']:.2f} % | agreement {100 * report['agreement']:.2f} % | loss {100 * (report['float_acc'] - report['int8_acc']):+.2f} points ({m} images)")
    samples = [{"x0": R.enc(inp(calib[s * B:(s + 1) * B])), "labels": [int(v) for v in labels[s * B:(s + 1) * B]] if labels is not None else []} for s in range(min(max_samples, len(calib) // B))]
    model = {"name": "custom-graph-int8", "n": n, "ops": ops, "weights": [R.enc(w) for w in W], "classes": int(classes), "batch": B, "input_scale": 1 / 64, "samples": samples}
    return (model, report, info) if return_info else (model, report)


# ---------------------------------------------------------------- import ResNet (torch state_dict, module de tools/train_resnet_cifar.py)
def resnet_nodes(sd, nb=3, eps=1e-5):
    g = lambda k: np.asarray(sd[k].detach().cpu().numpy() if hasattr(sd[k], "detach") else sd[k], dtype=np.float64)

    def fold(conv, bn):
        W = g(conv + ".weight")
        gam, bet, mu, var = g(bn + ".weight"), g(bn + ".bias"), g(bn + ".running_mean"), g(bn + ".running_var")
        sc = gam / np.sqrt(var + eps)
        return W * sc[:, None, None, None], bet - mu * sc

    nodes = [dict(name="in", op="input")]
    W, b = fold("conv", "bn")
    nodes.append(dict(name="c0", op="conv", src=["in"], W=W, b=b, pad=1, stride=1, relu=True))
    cur, widths = "c0", [W.shape[0], 2 * W.shape[0], 4 * W.shape[0]]
    cin = W.shape[0]
    for si, cout in enumerate(widths):
        for bi in range(nb):
            i = si * nb + bi
            p = f"blocks.{i}"
            stride = 2 if (bi == 0 and si > 0) else 1
            W1, b1 = fold(p + ".conv1", p + ".bn1")
            W2, b2 = fold(p + ".conv2", p + ".bn2")
            nodes.append(dict(name=f"b{i}c1", op="conv", src=[cur], W=W1, b=b1, pad=1, stride=stride, relu=True))
            nodes.append(dict(name=f"b{i}c2", op="conv", src=[f"b{i}c1"], W=W2, b=b2, pad=1, stride=1, relu=False))
            if (p + ".short.0.weight") in sd:
                Ws, bs = fold(p + ".short.0", p + ".short.1")
                nodes.append(dict(name=f"b{i}s", op="conv", src=[cur], W=Ws, b=bs, pad=0, stride=stride, relu=False))
                skip = f"b{i}s"
            else:
                skip = cur
            nodes.append(dict(name=f"b{i}", op="add", src=[f"b{i}c2", skip], relu=True))
            cur = f"b{i}"
            cin = cout
    nodes.append(dict(name="gap", op="pool", src=[cur], mode="avg", k=8))
    nodes.append(dict(name="flat", op="flatten", src=["gap"]))
    nodes.append(dict(name="fc", op="linear", src=["flat"], W=g("fc.weight"), b=g("fc.bias"), relu=False))
    return nodes


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        print("python3 quantize_graph.py resnet MODEL.pt OUT.json [--n 1024] [--calib 512]")
        return
    import torch
    sd = torch.load(a[1], weights_only=True, map_location="cpu")
    nodes = resnet_nodes(sd)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_resnet_cifar import load_cifar
    _, _, xte, yte = load_cifar()
    n = int(a[a.index("--n") + 1]) if "--n" in a else 1024
    ncal = int(a[a.index("--calib") + 1]) if "--calib" in a else 500
    model, rep = quantize_graph(nodes, xte[:ncal].astype(np.float64), n, yte[:ncal], shape=(3, 32, 32), max_samples=int(a[a.index("--samples") + 1]) if "--samples" in a else 4)
    json.dump(model, open(a[2], "w"))
    json.dump(rep, open(a[2].replace(".json", ".report.json"), "w"), indent=1)
    print("model written:", a[2], os.path.getsize(a[2]) // 1024, "KiB")


if __name__ == "__main__":
    main()
