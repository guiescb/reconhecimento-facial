"""E01 - Baseline: FaceNet pre-treinado, sem fine-tuning.

Uso:
    python experiments/run_e01.py --config configs/e01_baseline.yaml
    python experiments/run_e01.py --config configs/e01_baseline.yaml --synthetic   # teste rapido
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facerec.config import environment_info, load_config, set_seed  # noqa: E402
from facerec.data import load_lfw, synthetic_embeddings  # noqa: E402
from facerec.embeddings import cached_embeddings, extract_embeddings  # noqa: E402
from facerec.identification import open_set_identification  # noqa: E402
from facerec.metrics import evaluate_at_threshold, make_pairs, pair_scores, roc_eer  # noqa: E402
from facerec.splits import check_no_leakage, split_identities  # noqa: E402


def pairs_on(subset_idx, labels, emb, n_pairs, seed, metric):
    a, b, y = make_pairs(labels[subset_idx], n_pairs, seed)
    return pair_scores(emb, subset_idx[a], subset_idx[b], metric), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--synthetic", action="store_true", help="usa embeddings sinteticos (sem baixar dados)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    exp_id = cfg["experiment_id"] + ("_synthetic" if args.synthetic else "")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.synthetic:
        emb, labels = synthetic_embeddings(seed=seed)
        images = None
    else:
        d, m = cfg["dataset"], cfg["model"]
        images, labels, _names = load_lfw(d["min_faces_per_person"], d["crop"], d["image_size"])
        cache = Path(cfg["cache_dir"]) / f"{cfg['experiment_id']}_{m['pretrained']}_crop{d['crop']}.npz"
        emb = cached_embeddings(cache, lambda: extract_embeddings(
            images, m["batch_size"], m["device"], m["pretrained"]))

    splits = split_identities(labels, cfg["split"]["fractions"], seed)
    check_no_leakage(splits, labels, images)  # levanta erro se houver vazamento
    print("Checagem de leakage: OK (identidades disjuntas entre treino/validacao/teste)")

    results = {
        "experiment": exp_id,
        "hypothesis": " ".join(cfg["hypothesis"].split()),
        "config": cfg,
        "environment": environment_info(),
        "dataset": {
            "n_images": int(len(labels)),
            "n_identities": int(len(np.unique(labels))),
            **{f"{k}_images": int(len(v)) for k, v in splits.items()},
            **{f"{k}_identities": int(len(np.unique(labels[v]))) for k, v in splits.items()},
        },
        "metrics": {},
    }

    cmp_cfg, os_cfg = cfg["comparison"], cfg["open_set"]
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    fig_dist, axes = plt.subplots(1, len(cmp_cfg["metrics"]), figsize=(5 * len(cmp_cfg["metrics"]), 3.5))
    axes = np.atleast_1d(axes)

    for ax_d, metric in zip(axes, cmp_cfg["metrics"]):
        # 1) o threshold e escolhido SO na validacao
        s_val, y_val = pairs_on(splits["val"], labels, emb, cmp_cfg["n_val_pairs"], seed, metric)
        val_sum, _ = roc_eer(s_val, y_val)
        thr = val_sum["eer_threshold"]
        # 2) o teste so e usado para reportar
        s_te, y_te = pairs_on(splits["test"], labels, emb, cmp_cfg["n_test_pairs"], seed + 1, metric)
        te_sum, (fpr, tpr) = roc_eer(s_te, y_te)
        te_thr = evaluate_at_threshold(s_te, y_te, thr)

        test_ids = np.random.default_rng(seed).permutation(np.unique(labels[splits["test"]]))
        n_known = int(round(os_cfg["known_fraction"] * len(test_ids)))
        ident = open_set_identification(emb, labels, test_ids[:n_known], test_ids[n_known:], thr,
                                        os_cfg["n_enroll"], seed, metric)

        results["metrics"][metric] = {
            "validation": val_sum,
            "test_roc": te_sum,
            "test_at_validation_threshold": te_thr,
            "open_set_identification": ident,
        }
        ax_roc.plot(fpr, tpr, label=f"{metric} (AUC={te_sum['auc']:.3f})")
        ax_d.hist(s_te[y_te == 1], bins=40, alpha=0.6, label="mesma pessoa")
        ax_d.hist(s_te[y_te == 0], bins=40, alpha=0.6, label="pessoas diferentes")
        ax_d.axvline(thr, color="k", ls="--", label="threshold (val)")
        ax_d.set_title(f"Similaridade - {metric}")
        ax_d.legend(fontsize=7)

        print(f"\n[{metric}] threshold(val)={thr:.4f} | AUC={te_sum['auc']:.4f} | EER(teste)={te_sum['eer']:.4f}")
        print(f"  teste no threshold: acc={te_thr['accuracy']:.4f} prec={te_thr['precision']:.4f} "
              f"rec={te_thr['recall']:.4f} f1={te_thr['f1']:.4f} FAR={te_thr['far']:.4f} FRR={te_thr['frr']:.4f}")
        print(f"  conjunto aberto: id correta={ident['correct_identification_rate']:.4f} "
              f"desconhecido aceito={ident['false_acceptance_rate_unknown']:.4f}")

    ax_roc.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax_roc.set_xlabel("FAR")
    ax_roc.set_ylabel("1 - FRR")
    ax_roc.set_title("ROC (teste)")
    ax_roc.legend()
    fig_roc.tight_layout()
    fig_roc.savefig(out_dir / f"{exp_id}_roc.png", dpi=130)
    fig_dist.tight_layout()
    fig_dist.savefig(out_dir / f"{exp_id}_distribuicoes.png", dpi=130)

    results["runtime_seconds"] = round(time.time() - t0, 1)
    with open(out_dir / f"{exp_id}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em {out_dir}/ ({results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
