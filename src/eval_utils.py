"""Shared training / logging / analysis (Member 5).  Put in src/eval_utils.py

Row schema = Matthew's run_resnet.py history rows, so nothing he wrote changes:
  epoch, train_loss, train_time, eval_loss, eval_time,
  accuracy (caltech)  |  micro_precision, micro_recall, micro_f1 (voc),
  cumulative_train_time

Both ResNet and ViT call fit(). It saves one JSON per run:
  results/{model}_{dataset}_{size}_{init}[_{tag}]_s{seed}.json
"""
import json
import os
import time

import numpy as np

RESULTS_DIR = "results"
MODELS = ["resnet", "vit"]
DATASETS = ["caltech", "voc"]
SIZES = ["10mb", "100mb", "1gb"]
INITS = ["baseline", "proposed"]


def metric_name(dataset_name):
    return "accuracy" if dataset_name == "caltech" else "micro_f1"


# ===================================================================
# Training / evaluation (torch)
# ===================================================================
def _sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prep_labels(labels, dataset_name):
    return labels.long() if dataset_name == "caltech" else labels.float()


def train_one_epoch(model, loader, loss_func, optimizer, device,
                    dataset_name, amp=False, scaler=None):
    """Timing covers ONLY the training pass (GPU-synced), not evaluation."""
    import torch
    model.train()
    use_amp = amp and device.type == "cuda"
    total_loss, n = 0.0, 0

    _sync(device)
    t0 = time.perf_counter()
    for images, labels in loader:
        images = images.to(device)
        labels = _prep_labels(labels.to(device), dataset_name)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = loss_func(outputs, labels)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = images.size(0)
        total_loss += loss.detach() * bs      # no per-batch .item() sync
        n += bs
    _sync(device)
    return {"train_loss": float(total_loss) / n,
            "train_time": time.perf_counter() - t0}


def evaluate(model, loader, loss_func, device, dataset_name, amp=False):
    import torch
    model.eval()
    use_amp = amp and device.type == "cuda"
    total_loss, n, correct = 0.0, 0, 0
    tp = fp = fn = 0

    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = _prep_labels(labels.to(device), dataset_name)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = loss_func(outputs, labels)
            bs = images.size(0)
            total_loss += loss.item() * bs
            n += bs
            if dataset_name == "caltech":
                correct += (outputs.argmax(1) == labels).sum().item()
            else:
                pred, tgt = outputs >= 0, labels.bool()
                tp += (pred & tgt).sum().item()
                fp += (pred & ~tgt).sum().item()
                fn += (~pred & tgt).sum().item()
    _sync(device)

    m = {"eval_loss": total_loss / n, "eval_time": time.perf_counter() - t0}
    if dataset_name == "caltech":
        m["accuracy"] = correct / n
    else:
        m["micro_precision"] = tp / (tp + fp) if tp + fp else 0.0
        m["micro_recall"] = tp / (tp + fn) if tp + fn else 0.0
        m["micro_f1"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return m


def fit(model, model_name, dataset_name, size, init_name,
        train_loader, eval_loader, loss_func, optimizer, device,
        epochs, seed=42, amp=False, extra=None, save=True, verbose=True):
    """Train + evaluate every epoch, return history (list of dict), save JSON.

    model_name: "resnet" | "vit"      init_name: "baseline" | "proposed"
    extra: optional dict stored in metadata, e.g. {"init_scale": 0.5} (also
           becomes part of the filename via 'tag' so tuning runs don't collide)
    """
    import torch
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    mname = metric_name(dataset_name)
    hist, cum = [], 0.0
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, loss_func, optimizer, device,
                             dataset_name, amp, scaler)
        ev = evaluate(model, eval_loader, loss_func, device, dataset_name, amp)
        cum += tr["train_time"]
        row = {"epoch": epoch, **tr, **ev, "cumulative_train_time": cum}
        hist.append(row)
        if verbose:
            print(f"[{model_name}|{dataset_name}|{size}|{init_name}] "
                  f"ep {epoch}/{epochs} train {row['train_loss']:.4f} "
                  f"eval {row['eval_loss']:.4f} {mname} {row[mname]:.4f} "
                  f"({row['train_time']:.1f}s)")
    meta = dict(model=model_name, dataset=dataset_name, size=size, init=init_name,
                seed=seed, epochs=epochs, amp=amp,
                optimizer=type(optimizer).__name__,
                lr=optimizer.param_groups[0]["lr"], **(extra or {}))
    if save:
        save_run(hist, meta)
    return hist


# ===================================================================
# Saving / loading
# ===================================================================
def run_path(meta, results_dir=RESULTS_DIR):
    tag = f"_sc{meta['init_scale']}" if "init_scale" in meta else ""
    return os.path.join(
        results_dir,
        f"{meta['model']}_{meta['dataset']}_{meta['size']}_{meta['init']}{tag}_s{meta['seed']}.json")


def save_run(hist, meta, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    with open(run_path(meta, results_dir), "w") as f:
        json.dump({"meta": meta, "history": hist}, f, indent=2)


def load_results(results_dir=RESULTS_DIR):
    """One row per (run, epoch) in a pandas DataFrame."""
    import glob
    import pandas as pd
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        d = json.load(open(p))
        for r in d["history"]:
            rows.append({**d["meta"], **r})
    return pd.DataFrame(rows)


# ===================================================================
# Analysis
# ===================================================================
def _mean_over_seeds(df):
    """Average numeric columns over seeds, per (model,dataset,size,init,epoch).
    Tuning runs (with init_scale) are excluded; analyse those separately."""
    if "init_scale" in df.columns:
        df = df[df["init_scale"].isna()]
    keys = ["model", "dataset", "size", "init", "epoch"]
    num = df.select_dtypes("number").columns.difference(["seed"] + keys)
    return df.groupby(keys)[list(num)].mean().reset_index()


def time_to_target(g, col, target):
    """Cumulative train time at first epoch where col >= target (NaN if never)."""
    hit = g[g[col] >= target].sort_values("epoch")
    return hit["cumulative_train_time"].iloc[0] if len(hit) else np.nan


def summary_table(df):
    import pandas as pd
    d = _mean_over_seeds(df)
    out = []
    for (mo, ds, sz, ini), g in d.groupby(["model", "dataset", "size", "init"]):
        g = g.sort_values("epoch")
        m = metric_name(ds)
        out.append(dict(model=mo, dataset=ds, size=sz, init=ini,
                        final_metric=g[m].iloc[-1], best_metric=g[m].max(),
                        final_eval_loss=g["eval_loss"].iloc[-1],
                        total_train_time=g["cumulative_train_time"].iloc[-1]))
    return pd.DataFrame(out)


def speedup_table(df, frac=0.9):
    """Time for each init to reach frac * (baseline's best metric).
    speedup = baseline_time / proposed_time (>1 means proposed is faster)."""
    import pandas as pd
    d = _mean_over_seeds(df)
    out = []
    for (mo, ds, sz), g in d.groupby(["model", "dataset", "size"]):
        m = metric_name(ds)
        b = g[g.init == "baseline"]
        p = g[g.init == "proposed"]
        if b.empty or p.empty:
            continue
        target = frac * b[m].max()
        tb, tp = time_to_target(b, m, target), time_to_target(p, m, target)
        out.append(dict(model=mo, dataset=ds, size=sz, target=target,
                        baseline_time=tb, proposed_time=tp,
                        speedup=tb / tp if tp and not np.isnan(tp) else np.nan,
                        baseline_final=b.sort_values("epoch")[m].iloc[-1],
                        proposed_final=p.sort_values("epoch")[m].iloc[-1]))
    return pd.DataFrame(out)


# ===================================================================
# Plots
# ===================================================================
def plot_curves(df, model, dataset, y="eval_loss", x="epoch", save=None):
    """Baseline vs proposed; one panel per dataset size.
    y: train_loss | eval_loss | accuracy | micro_f1 ; x: epoch | cumulative_train_time"""
    import matplotlib.pyplot as plt
    d = _mean_over_seeds(df)
    fig, axes = plt.subplots(1, len(SIZES), figsize=(5 * len(SIZES), 3.8), sharey=True)
    for ax, sz in zip(axes, SIZES):
        for ini in INITS:
            g = d[(d.model == model) & (d.dataset == dataset) &
                  (d["size"] == sz) & (d.init == ini)].sort_values("epoch")
            if len(g):
                ax.plot(g[x], g[y], marker="o", label=ini)
        ax.set(title=f"{model} | {dataset} | {sz}", xlabel=x, ylabel=y)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=150)
    return fig


def plot_final_bars(summary, col="final_metric", save=None):
    import matplotlib.pyplot as plt
    s = summary.copy()
    s["cfg"] = s["model"] + "\n" + s["dataset"] + "\n" + s["size"]
    piv = s.pivot(index="cfg", columns="init", values=col)
    ax = piv.plot.bar(figsize=(13, 4), rot=0)
    ax.set_ylabel(col)
    ax.figure.tight_layout()
    if save:
        ax.figure.savefig(save, dpi=150)
    return ax.figure


def plot_speedup(sp, save=None):
    import matplotlib.pyplot as plt
    s = sp.copy()
    s["cfg"] = s["model"] + "\n" + s["dataset"] + "\n" + s["size"]
    ax = s.set_index("cfg")["speedup"].plot.bar(figsize=(13, 4), rot=0)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_ylabel("speedup (baseline time / proposed time)")
    ax.figure.tight_layout()
    if save:
        ax.figure.savefig(save, dpi=150)
    return ax.figure


# ===================================================================
# Mock data so you can build the notebook before real runs finish
# ===================================================================
def make_mock_results(results_dir="results_mock", epochs=10, seed=0):
    rng = np.random.default_rng(seed)
    for mo in MODELS:
        for ds in DATASETS:
            for sz in SIZES:
                for ini in INITS:
                    rate = 0.45 if ini == "proposed" else 0.3
                    top = {"caltech": 0.8, "voc": 0.7}[ds] * {"10mb": .6, "100mb": .85, "1gb": 1}[sz]
                    hist, cum = [], 0.0
                    for e in range(1, epochs + 1):
                        t = 10 * {"10mb": .1, "100mb": 1, "1gb": 8}[sz] * (1 + .05 * rng.standard_normal())
                        cum += t
                        perf = top * (1 - np.exp(-rate * e)) + .01 * rng.standard_normal()
                        row = dict(epoch=e, train_loss=3 * np.exp(-rate * e) + .1, train_time=t,
                                   eval_loss=3 * np.exp(-rate * e) + .3, eval_time=t / 5,
                                   cumulative_train_time=cum)
                        if ds == "caltech":
                            row["accuracy"] = perf
                        else:
                            row.update(micro_f1=perf, micro_precision=perf, micro_recall=perf)
                        hist.append(row)
                    meta = dict(model=mo, dataset=ds, size=sz, init=ini, seed=0, epochs=epochs)
                    save_run(hist, meta, results_dir)
