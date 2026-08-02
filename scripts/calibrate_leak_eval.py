"""Score the rule-based leak detector against the human-style leak ratings.

Measurement only. Run with PYTHONPATH=src.
Writes data/leak_calibration.json.
"""

import itertools
import json
import os
import random

import numpy as np

from calibrate_leak import load, ROOT

CUR = {"verbatim": 1.0, "overlap": 0.6, "elimination": 0.5, "identifying_hits": 1.0}
SIGNALS = ("verbatim", "overlap", "elimination", "identifying_hits")


def human_class(leak: float) -> int:
    """Bin the mean rating. 1.5 and 2.5 (n=14, split two-rater rows) fall in `hints`."""
    if leak < 1.5:
        return 1
    if leak > 2.5:
        return 3
    return 2


def prf(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(pred) if len(pred) else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec,
            "recall": rec, "f1": f1, "accuracy": acc, "balanced_accuracy": (rec + tnr) / 2,
            "mcc": mcc, "n": int(len(pred)),
            "n_pos": int(truth.sum()), "flag_rate": float(pred.mean()) if len(pred) else 0.0}


def auc(score, truth):
    """Rank AUC with ties handled by average rank."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(truth, dtype=bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def or_rule(mat, spec):
    """spec: dict signal -> threshold (absent = signal not used). OR over the members."""
    pred = np.zeros(mat["verbatim"].shape, dtype=bool)
    for k, thr in spec.items():
        pred |= mat[k] >= thr
    return pred


def grids(mat):
    return {
        "verbatim": [1.0],
        "overlap": [round(x, 2) for x in np.arange(0.05, 1.001, 0.05)],
        "elimination": sorted({float(v) for v in mat["elimination"] if v > 0}) or [0.5],
        "identifying_hits": [float(k) for k in range(1, int(mat["identifying_hits"].max()) + 1)],
    }


def best_or_rule(mat, truth, idx=None, max_members=4, objective="f1"):
    """Exhaustive search over OR-combinations of the four signals and their cutoffs."""
    g = grids(mat)
    sub = {k: (v[idx] if idx is not None else v) for k, v in mat.items()}
    y = truth[idx] if idx is not None else truth
    best = None
    for r in range(1, max_members + 1):
        for members in itertools.combinations(SIGNALS, r):
            for combo in itertools.product(*[g[m] for m in members]):
                spec = dict(zip(members, combo))
                m = prf(or_rule(sub, spec), y)
                if best is None or m[objective] > best[1][objective]:
                    best = (spec, m)
    return best


def main():
    rows = load()
    n = len(rows)
    mism = [r["id"] for r in rows if bool(r["recomputed"]) != r["rule_flagged"]]

    mat = {k: np.array([r[k] for r in rows], dtype=float) for k in SIGNALS}
    mat["identifying"] = np.array([r["identifying"] for r in rows], dtype=float)
    flagged = np.array([r["rule_flagged"] for r in rows], dtype=bool)
    leak = np.array([r["leak"] for r in rows], dtype=float)
    cls = np.array([human_class(x) for x in leak])

    out = {
        "meta": {
            "n_rows": n,
            "n_mismatch_recomputed_vs_stored": len(mism),
            "mismatch_ids": mism[:50],
            "rating_distribution": {str(k): int((leak == k).sum())
                                    for k in sorted(set(leak.tolist()))},
            "class_distribution": {"1_no_leak": int((cls == 1).sum()),
                                   "2_hints": int((cls == 2).sum()),
                                   "3_gives_away": int((cls == 3).sum())},
            "n_raters": {"1": int(sum(1 for r in rows if r["n_raters"] == 1)),
                         "2": int(sum(1 for r in rows if r["n_raters"] == 2))},
            "fractional_mean_rows": int(sum(1 for r in rows if r["leak"] not in (1.0, 2.0, 3.0))),
            "current_thresholds": CUR,
            "current_flag_rate": float(flagged.mean()),
        }
    }

    # ---- treatments -------------------------------------------------------
    treat = {
        "A_hints_are_leak": {"mask": np.ones(n, dtype=bool), "truth": cls >= 2},
        "B_hints_are_clean": {"mask": np.ones(n, dtype=bool), "truth": cls == 3},
        "C_strict_drop_hints": {"mask": cls != 2, "truth": cls == 3},
    }
    # sensitivity: drop the 14 fractional means entirely
    frac = np.array([r["leak"] not in (1.0, 2.0, 3.0) for r in rows])
    treat["A_no_fractional"] = {"mask": ~frac, "truth": cls >= 2}
    treat["B_no_fractional"] = {"mask": ~frac, "truth": cls == 3}

    out["treatments"] = {}
    for name, t in treat.items():
        m, y = t["mask"], t["truth"]
        block = {
            "n": int(m.sum()),
            "n_pos": int(y[m].sum()),
            "base_rate": float(y[m].mean()),
            "current_rule": prf(flagged[m], y[m]),
            "baseline_flag_everything": prf(np.ones(int(m.sum()), dtype=bool), y[m]),
            "baseline_flag_nothing": prf(np.zeros(int(m.sum()), dtype=bool), y[m]),
            "signals_alone_at_current_threshold": {},
            "signals_auc": {},
            "best_single_threshold": {},
        }
        for s in SIGNALS:
            block["signals_alone_at_current_threshold"][s] = prf(mat[s][m] >= CUR[s], y[m])
            block["signals_auc"][s] = auc(mat[s][m], y[m])
        block["signals_auc"]["identifying_fraction"] = auc(mat["identifying"][m], y[m])
        g = grids(mat)
        for s in SIGNALS:
            cand = [(prf(mat[s][m] >= thr, y[m]), thr) for thr in g[s]]
            bm, bthr = max(cand, key=lambda x: x[0]["f1"])
            block["best_single_threshold"][s] = {"threshold": bthr, **bm}
        spec, bm = best_or_rule({k: mat[k] for k in SIGNALS}, y, idx=m)
        block["best_or_rule_in_sample"] = {"spec": spec, **bm,
                                           "f1_gain_vs_current": bm["f1"] - block["current_rule"]["f1"]}
        spec_m, bmm = best_or_rule({k: mat[k] for k in SIGNALS}, y, idx=m, objective="mcc")
        block["best_or_rule_by_mcc"] = {"spec": spec_m, **bmm,
                                        "mcc_gain_vs_current": bmm["mcc"] - block["current_rule"]["mcc"]}

        # ablation: current OR rule with one member removed
        sub = {k: mat[k][m] for k in SIGNALS}
        block["ablation_drop_one"] = {}
        for s in SIGNALS:
            spec_ab = {k: CUR[k] for k in SIGNALS if k != s}
            block["ablation_drop_one"]["without_" + s] = prf(or_rule(sub, spec_ab), y[m])
        # k-of-4 family at the current thresholds
        fires = np.stack([sub[s] >= CUR[s] for s in SIGNALS]).sum(axis=0)
        block["k_of_4_at_current_thresholds"] = {
            str(k): prf(fires >= k, y[m]) for k in (1, 2, 3, 4)}

        # bootstrap CI on the paired F1/MCC difference for two fixed candidates
        block["bootstrap_vs_current"] = {}
        for label, spec_c in (("verbatim_or_ident1", {"verbatim": 1.0, "identifying_hits": 1.0}),
                              ("verbatim_or_ident2", {"verbatim": 1.0, "identifying_hits": 2.0})):
            cand = or_rule(sub, spec_c)
            cur_pred = flagged[m]
            yy = y[m]
            rs = np.random.default_rng(0)
            d_f1, d_mcc = [], []
            for _ in range(2000):
                b = rs.integers(0, len(yy), len(yy))
                d_f1.append(prf(cand[b], yy[b])["f1"] - prf(cur_pred[b], yy[b])["f1"])
                d_mcc.append(prf(cand[b], yy[b])["mcc"] - prf(cur_pred[b], yy[b])["mcc"])
            block["bootstrap_vs_current"][label] = {
                "spec": spec_c,
                "point": prf(cand, yy),
                "d_f1_mean": float(np.mean(d_f1)),
                "d_f1_ci95": [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
                "d_mcc_mean": float(np.mean(d_mcc)),
                "d_mcc_ci95": [float(np.percentile(d_mcc, 2.5)), float(np.percentile(d_mcc, 97.5))],
            }
        out["treatments"][name] = block

    # ---- leave-out validation of the sweep -------------------------------
    rng = random.Random(0)
    out["cross_validation"] = {}
    for name in ("A_hints_are_leak", "B_hints_are_clean"):
        y = treat[name]["truth"]
        idx = np.arange(n)
        perm = np.array(sorted(idx, key=lambda _: rng.random()))
        folds = np.array_split(perm, 5)
        cur_f1, swept_f1, specs, all_f1 = [], [], [], []
        for f in folds:
            tr = np.setdiff1d(perm, f)
            spec, _ = best_or_rule({k: mat[k] for k in SIGNALS}, y, idx=tr)
            te = {k: mat[k][f] for k in SIGNALS}
            swept_f1.append(prf(or_rule(te, spec), y[f])["f1"])
            cur_f1.append(prf(flagged[f], y[f])["f1"])
            all_f1.append(prf(np.ones(len(f), dtype=bool), y[f])["f1"])
            specs.append(spec)
        out["cross_validation"][name] = {
            "folds": 5,
            "current_rule_f1_mean": float(np.mean(cur_f1)),
            "swept_rule_f1_mean": float(np.mean(swept_f1)),
            "flag_everything_f1_mean": float(np.mean(all_f1)),
            "gain_mean": float(np.mean(swept_f1) - np.mean(cur_f1)),
            "per_fold_current": [float(x) for x in cur_f1],
            "per_fold_swept": [float(x) for x in swept_f1],
            "per_fold_spec": specs,
        }

    # ---- signal firing counts / overlap ----------------------------------
    fire = {s: (mat[s] >= CUR[s]) for s in SIGNALS}
    out["signal_firing"] = {
        s: {"n_fires": int(fire[s].sum()),
            "rate": float(fire[s].mean()),
            "n_unique_fires": int((fire[s] & ~np.logical_or.reduce(
                [fire[o] for o in SIGNALS if o != s])).sum()),
            "mean_human_rating_when_fires": float(leak[fire[s]].mean()) if fire[s].any() else None}
        for s in SIGNALS
    }
    out["signal_firing"]["none"] = {
        "n_fires": int((~np.logical_or.reduce([fire[s] for s in SIGNALS])).sum()),
        "mean_human_rating_when_fires": float(
            leak[~np.logical_or.reduce([fire[s] for s in SIGNALS])].mean()),
    }
    out["mean_human_rating"] = {
        "flagged": float(leak[flagged].mean()),
        "not_flagged": float(leak[~flagged].mean()),
        "all": float(leak.mean()),
    }

    # label provenance: the ratings are mostly LLM raters, not people
    rm_raw = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "data/rm_dataset.json")))["rows"]}
    has_human = np.array([any(x.startswith("human:")
                              for x in (rm_raw[r["id"]].get("raters") or [])) for r in rows])
    dual = np.array([r["n_raters"] == 2 for r in rows])
    spread = np.array([rm_raw[r["id"]].get("leak_spread", 0) for r in rows], dtype=float)
    out["label_provenance"] = {
        "rows_with_any_human_rater": int(has_human.sum()),
        "rows_llm_rated_only": int((~has_human).sum()),
        "rows_double_rated": int(dual.sum()),
        "double_rated_exact_agreement": float((spread[dual] == 0).mean()) if dual.any() else None,
        "double_rated_spread_counts": {str(int(k)): int((spread[dual] == k).sum())
                                       for k in sorted(set(spread[dual].tolist()))},
        "human_subset_current_rule_A": prf(flagged[has_human], (cls >= 2)[has_human]),
        "human_subset_current_rule_B": prf(flagged[has_human], (cls == 3)[has_human]),
    }

    # the candidate that every CV fold selected, scored like the current rule
    rec_spec = {"verbatim": 1.0, "identifying_hits": 1.0}
    rec_pred = or_rule({k: mat[k] for k in SIGNALS}, rec_spec)
    out["recommended_rule"] = {
        "spec": rec_spec,
        "description": "verbatim OR identifying_hits >= 1; overlap and elimination dropped",
        "flag_rate": float(rec_pred.mean()),
        "A_hints_are_leak": prf(rec_pred, cls >= 2),
        "B_hints_are_clean": prf(rec_pred, cls == 3),
        "C_strict_drop_hints": prf(rec_pred[cls != 2], (cls == 3)[cls != 2]),
    }

    # by subject, current rule vs the recommended one
    out["by_subject"] = {}
    subjects = sorted({r["subject"] for r in rows if r["subject"]})
    for s in subjects:
        m = np.array([r["subject"] == s for r in rows])
        out["by_subject"][s] = {
            "n": int(m.sum()),
            "A": prf(flagged[m], (cls >= 2)[m]),
            "B": prf(flagged[m], (cls == 3)[m]),
            "B_recommended": prf(rec_pred[m], (cls == 3)[m]),
        }

    # ---- reweight to the run's real stratum mix ---------------------------
    # The label set oversamples the rule's decision boundary and adds 447
    # hand-written `good` turns that are not in the run at all. Unweighted
    # precision/recall therefore describe the sample, not the policy.
    key = json.load(open(os.path.join(ROOT, "label_app/data/label_key.json")))
    strat = {t["id"]: t["stratum"] for t in key["turns"]}
    tier = {t["id"]: t["tier"] for t in key["turns"]}
    pop = key["population"]
    pol = np.array([tier[r["id"]] == "policy" for r in rows])
    s_of = np.array([strat[r["id"]] for r in rows])
    samp = {s: int(((s_of == s) & pol).sum()) for s in pop}
    w = np.array([pop[s] / samp[s] if samp.get(s) else 0.0 for s in s_of])
    w = np.where(pol, w, 0.0)

    def wprf(pred, truth):
        pred = np.asarray(pred, dtype=bool)
        truth = np.asarray(truth, dtype=bool)
        tp = float(w[pred & truth].sum())
        fp = float(w[pred & ~truth].sum())
        fn = float(w[~pred & truth].sum())
        tn = float(w[~pred & ~truth].sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {"precision": p, "recall": r,
                "f1": 2 * p * r / (p + r) if p + r else 0.0,
                "accuracy": (tp + tn) / (tp + fp + fn + tn),
                "eff_pos_rate": (tp + fn) / (tp + fp + fn + tn),
                "flag_rate": (tp + fp) / (tp + fp + fn + tn),
                "w_tp": tp, "w_fp": fp, "w_fn": fn, "w_tn": tn}

    out["population_weighted"] = {
        "note": ("policy-tier rows only, each stratum reweighted to label_key.population; "
                 "the 447 hand-written good-tier turns are excluded because they are not "
                 "in the run"),
        "population_counts": pop,
        "sampled_policy_counts": samp,
        "n_policy_rows": int(pol.sum()),
        "current_rule_A": wprf(flagged, cls >= 2),
        "current_rule_B": wprf(flagged, cls == 3),
        "recommended_rule_A": wprf(rec_pred, cls >= 2),
        "recommended_rule_B": wprf(rec_pred, cls == 3),
        "estimated_true_rate_gives_away": wprf(np.ones(n, dtype=bool), cls == 3)["eff_pos_rate"],
        "estimated_true_rate_hints_or_worse": wprf(np.ones(n, dtype=bool), cls >= 2)["eff_pos_rate"],
    }

    # ---- error anatomy ----------------------------------------------------
    import collections
    import re
    fired = lambda r: tuple(sorted(s for s in SIGNALS if r[s] >= CUR[s]))  # noqa: E731
    numeric = lambda r: bool(re.search(r"\d", r["gold"]))                  # noqa: E731
    fp = [r for r in rows if r["rule_flagged"] and r["leak"] <= 1.0]
    fn = [r for r in rows if not r["rule_flagged"] and r["leak"] >= 3.0]
    out["errors"] = {
        "n_false_positive_rated_1": len(fp),
        "n_false_negative_rated_3": len(fn),
        "fp_by_signal_combination": {"+".join(k): v for k, v in
                                     collections.Counter(fired(r) for r in fp).most_common()},
        "fp_numeric_gold": sum(1 for r in fp if numeric(r)),
        "fp_elimination_fired": sum(1 for r in fp if r["elimination"] >= CUR["elimination"]),
        "fp_elimination_fired_numeric_gold": sum(
            1 for r in fp if r["elimination"] >= CUR["elimination"] and numeric(r)),
        "fp_single_identifying_hit": sum(1 for r in fp if r["identifying_hits"] == 1),
        "fp_by_subject": dict(collections.Counter(r["subject"] for r in fp).most_common()),
        "fn_zero_overlap": sum(1 for r in fn if r["overlap"] == 0.0),
        "fn_all_signals_zero": sum(1 for r in fn if r["overlap"] == 0
                                   and r["identifying_hits"] == 0 and r["elimination"] == 0),
        "fn_no_identifying_words": sum(1 for r in fn if r["identifying_n"] == 0),
        "fn_by_subject": dict(collections.Counter(r["subject"] for r in fn).most_common()),
        "elimination_diagnostics": {
            "n_fires": int((mat["elimination"] >= CUR["elimination"]).sum()),
            "share_numeric_gold": float(np.mean([numeric(r) for r in rows
                                                 if r["elimination"] >= CUR["elimination"]])),
            "mean_human_rating_when_fires": float(np.mean(
                [r["leak"] for r in rows if r["elimination"] >= CUR["elimination"]])),
            "corpus_mean_human_rating": float(leak.mean()),
        },
        "examples_false_positive": [{k: r[k] for k in
                                     ("id", "leak", "subject", "grade", "question", "gold",
                                      "distractors", "tutor_turn", "verbatim", "overlap",
                                      "elimination", "identifying_hits", "identifying_n")}
                                    for r in fp],
        "examples_false_negative": [{k: r[k] for k in
                                     ("id", "leak", "subject", "grade", "question", "gold",
                                      "distractors", "tutor_turn", "verbatim", "overlap",
                                      "elimination", "identifying_hits", "identifying_n")}
                                    for r in fn],
    }

    json.dump(out, open(os.path.join(ROOT, "data/leak_calibration.json"), "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("meta", "signal_firing", "mean_human_rating")}, indent=2))
    for name, b in out["treatments"].items():
        print("\n===", name, "n=", b["n"], "pos=", b["n_pos"])
        c = b["current_rule"]
        print(f"  current   P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f} "
              f"acc={c['accuracy']:.3f} mcc={c['mcc']:.3f}  tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']}")
        a = b["baseline_flag_everything"]
        print(f"  flag-all  P={a['precision']:.3f} R={a['recall']:.3f} F1={a['f1']:.3f} "
              f"acc={a['accuracy']:.3f} mcc={a['mcc']:.3f}")
        for s, m in b["signals_alone_at_current_threshold"].items():
            print(f"  {s:18s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                  f"acc={m['accuracy']:.3f} fires={m['tp']+m['fp']} auc={b['signals_auc'][s]:.3f}")
        for s, m in b["best_single_threshold"].items():
            print(f"  best {s:13s} thr={m['threshold']:<5} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} F1={m['f1']:.3f}")
        bo = b["best_or_rule_in_sample"]
        print(f"  BEST-F1 OR {bo['spec']} P={bo['precision']:.3f} R={bo['recall']:.3f} "
              f"F1={bo['f1']:.3f} acc={bo['accuracy']:.3f} mcc={bo['mcc']:.3f} gain={bo['f1_gain_vs_current']:+.3f}")
        bm2 = b["best_or_rule_by_mcc"]
        print(f"  BEST-MCC OR {bm2['spec']} P={bm2['precision']:.3f} R={bm2['recall']:.3f} "
              f"F1={bm2['f1']:.3f} acc={bm2['accuracy']:.3f} mcc={bm2['mcc']:.3f} "
              f"gain={bm2['mcc_gain_vs_current']:+.3f}")
        for k, m2 in b["ablation_drop_one"].items():
            print(f"  {k:26s} P={m2['precision']:.3f} R={m2['recall']:.3f} F1={m2['f1']:.3f} "
                  f"mcc={m2['mcc']:.3f}")
        for k, m2 in b["k_of_4_at_current_thresholds"].items():
            print(f"  k>={k} signals            P={m2['precision']:.3f} R={m2['recall']:.3f} "
                  f"F1={m2['f1']:.3f} mcc={m2['mcc']:.3f}")
        for k, m2 in b["bootstrap_vs_current"].items():
            print(f"  boot {k:20s} dF1={m2['d_f1_mean']:+.3f} "
                  f"[{m2['d_f1_ci95'][0]:+.3f},{m2['d_f1_ci95'][1]:+.3f}]  "
                  f"dMCC={m2['d_mcc_mean']:+.3f} "
                  f"[{m2['d_mcc_ci95'][0]:+.3f},{m2['d_mcc_ci95'][1]:+.3f}]")
    print("\ncross-validation:", json.dumps(out["cross_validation"], indent=2))


if __name__ == "__main__":
    main()
