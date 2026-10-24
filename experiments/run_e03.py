"""E03 - Camada de projecao treinada sobre embeddings congelados (Triplet x Cross-Entropy).

Usa os MESMOS embeddings do E01/E02 (le o cache). Treina so com as identidades de TREINO,
escolhe a epoca na VALIDACAO e usa o TESTE apenas para reportar.

Uso:
    python experiments/run_e03.py --config configs/e03_head.yaml
    python experiments/run_e03.py --config configs/e03_head.yaml --synthetic   # teste rapido
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
from facerec.headtrain import embed_numpy, train_head  # noqa: E402
from facerec.metrics import evaluate_at_threshold, make_pairs, pair_scores, roc_eer  # noqa: E402
from facerec.openset import (build_open_set, collect_unknown_scores, evaluate_open_set,  # noqa: E402
                             identity_scores, open_set_curve, threshold_for_far)
from facerec.splits import check_no_leakage, split_identities  # noqa: E402

LOSS_NAMES = {"triplet": "Triplet (batch-hard)", "cross_entropy": "Cross-Entropy (cosseno)"}


def load_labels(cfg):
    """Rotulos do LFW em cache (evita recarregar as imagens a cada experimento)."""
    path = Path(cfg["labels_cache"])
    if path.exists():
        return np.load(path)
    d = cfg["dataset"]
    _, labels, _ = load_lfw(d["min_faces_per_person"], d["crop"], d["image_size"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, labels)
    return labels


def evaluate_embeddings(emb, labels, splits, test_set, val_ids, cfg, seed):
    """Mesma avaliacao para o baseline e para cada camada treinada."""
    pc, oc = cfg["pairs"], cfg["open_set"]
    va, vb, vy = make_pairs(labels[splits["val"]], pc["n_val_pairs"], seed)
    s_val = pair_scores(emb, splits["val"][va], splits["val"][vb], "cosine")
    val_sum, _ = roc_eer(s_val, vy)
    thr_eer = val_sum["eer_threshold"]
    ta, tb, ty = make_pairs(labels[splits["test"]], pc["n_test_pairs"], seed + 1)
    s_te = pair_scores(emb, splits["test"][ta], splits["test"][tb], "cosine")
    te_sum, _ = roc_eer(s_te, ty)
    te_thr = evaluate_at_threshold(s_te, ty, thr_eer)

    scores, ids = identity_scores(emb, test_set, "prototype")
    unknown_val = collect_unknown_scores(emb, labels, val_ids, "prototype", oc["known_fraction"],
                                         oc["n_enroll"], oc["n_calibration_repeats"], seed)
    thresholds = {"pair_eer": thr_eer}
    for t in oc["far_targets"]:
        thresholds[f"far_{t}"] = threshold_for_far(unknown_val, t)   # calibrado so na validacao
    open_set = {n: evaluate_open_set(scores, ids, test_set, thr) for n, thr in thresholds.items()}
    curve = open_set_curve(scores, ids, test_set)
    return {"pairs": {"validation": val_sum, "test_roc": te_sum, "test_at_validation_threshold": te_thr},
            "open_set": open_set}, curve


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
    oc, tr = cfg["open_set"], cfg["training"]
    t0 = time.time()

    if args.synthetic:
        emb, labels = synthetic_embeddings(seed=seed)
    else:
        d, m = cfg["dataset"], cfg["model"]
        emb = cached_embeddings(cfg["embeddings_cache"], lambda: extract_embeddings(
            load_lfw(d["min_faces_per_person"], d["crop"], d["image_size"])[0],
            m["batch_size"], m["device"], m["pretrained"]))
        labels = load_labels(cfg)
    assert len(emb) == len(labels), "cache de embeddings e rotulos com tamanhos diferentes"

    splits = split_identities(labels, cfg["split"]["fractions"], seed)
    check_no_leakage(splits, labels)   # a checagem por hash de imagem ja foi feita no E01/E02
    print("Checagem de leakage: OK (identidades disjuntas entre treino/validacao/teste)")

    val_ids = np.unique(labels[splits["val"]])
    test_ids = np.random.default_rng(seed).permutation(np.unique(labels[splits["test"]]))
    n_known = int(round(oc["known_fraction"] * len(test_ids)))
    test_set = build_open_set(labels, test_ids[:n_known], test_ids[n_known:], oc["n_enroll"], seed)

    results = {
        "experiment": exp_id,
        "hypothesis": " ".join(cfg["hypothesis"].split()),
        "config": cfg,
        "environment": environment_info(),
        "dataset": {f"{k}_images": int(len(v)) for k, v in splits.items()}
        | {f"{k}_identities": int(len(np.unique(labels[v]))) for k, v in splits.items()},
        "models": {},
    }

    # --- Baseline (sem treino): deve reproduzir o prototipo do E02
    base, base_curve = evaluate_embeddings(emb, labels, splits, test_set, val_ids, cfg, seed)
    results["models"]["baseline"] = {"evaluation": base}
    curves = {"baseline (sem treino)": base_curve}
    all_eval = {"baseline (sem treino)": base}

    # --- Treino so com as identidades de TREINO; melhor epoca escolhida na VALIDACAO
    va, vb, vy = make_pairs(labels[splits["val"]], cfg["pairs"]["n_val_pairs"], seed)
    emb_val = emb[splits["val"]]
    histories = {}
    for loss_name in tr["losses"]:
        print(f"\nTreinando camada com {LOSS_NAMES[loss_name]} ...")
        out = train_head(emb[splits["train"]], labels[splits["train"]], emb_val, (va, vb, vy),
                         loss_name, tr, seed)
        emb2 = embed_numpy(out["head"], emb)
        ev, curve = evaluate_embeddings(emb2, labels, splits, test_set, val_ids, cfg, seed)
        results["models"][loss_name] = {"best_epoch": out["best_epoch"], "best_val_eer": out["best_val_eer"],
                                        "history": out["history"], "evaluation": ev}
        histories[loss_name] = out["history"]
        curves[LOSS_NAMES[loss_name]] = curve
        all_eval[LOSS_NAMES[loss_name]] = ev
        if not args.synthetic:
            import torch
            Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
            torch.save(out["head"].state_dict(), Path(cfg["save_dir"]) / f"E03_head_{loss_name}.pt")

    # --- Tabelas
    print("\nPares (teste, threshold de EER escolhido na validacao):")
    print(f"{'modelo':<26}{'AUC':>8}{'EER':>8}{'acc':>8}{'F1':>8}{'FAR':>8}{'FRR':>8}")
    for name, ev in all_eval.items():
        p = ev["pairs"]
        t = p["test_at_validation_threshold"]
        print(f"{name:<26}{p['test_roc']['auc']:>8.4f}{p['test_roc']['eer']:>8.4f}{t['accuracy']:>8.4f}"
              f"{t['f1']:>8.4f}{t['far']:>8.4f}{t['frr']:>8.4f}")
    print("\nConjunto aberto (teste, prototipo):")
    print(f"{'modelo':<26}{'threshold':<12}{'thr':>8}{'id_correta':>12}{'desc_aceito':>13}"
          f"{'FRR_conh':>10}{'erro_pessoa':>13}{'acc_aberto':>12}")
    for name, ev in all_eval.items():
        for tname, r in ev["open_set"].items():
            print(f"{name:<26}{tname:<12}{r['threshold']:>8.4f}{r['correct_identification_rate']:>12.4f}"
                  f"{r['false_acceptance_rate_unknown']:>13.4f}{r['false_rejection_rate_known']:>10.4f}"
                  f"{r['misidentification_rate_known']:>13.4f}{r['open_set_accuracy']:>12.4f}")
    for loss_name in tr["losses"]:
        mdl = results["models"][loss_name]
        print(f"\n{LOSS_NAMES[loss_name]}: melhor epoca (validacao) = {mdl['best_epoch']} "
              f"de {tr['epochs']}, val_EER = {mdl['best_val_eer']:.4f}")

    # --- Graficos
    fig, (ax_l, ax_e) = plt.subplots(1, 2, figsize=(11, 4))
    for loss_name, h in histories.items():
        ax_l.plot([r["epoch"] for r in h[1:]], [r["train_loss"] for r in h[1:]], label=LOSS_NAMES[loss_name])
        ax_e.plot([r["epoch"] for r in h], [r["val_eer"] for r in h], label=LOSS_NAMES[loss_name])
    ax_l.set_xlabel("epoca")
    ax_l.set_ylabel("perda de treino")
    ax_l.set_title("Curva de treino")
    ax_l.legend(fontsize=8)
    ax_e.set_xlabel("epoca (0 = baseline)")
    ax_e.set_ylabel("EER de pares na validacao")
    ax_e.set_title("Validacao (escolhe a melhor epoca)")
    ax_e.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{exp_id}_treino.png", dpi=130)

    fig2, ax = plt.subplots(figsize=(6, 4.5))
    for name, (_, far, cir) in curves.items():
        line, = ax.plot(far, cir, label=name)
        for tname, r in all_eval[name]["open_set"].items():
            ax.plot(r["false_acceptance_rate_unknown"], r["correct_identification_rate"], "o", ms=5,
                    color=line.get_color(), mfc="white" if tname == "pair_eer" else line.get_color())
    ax.set_xlabel("desconhecidos aceitos (FAR)")
    ax.set_ylabel("identificacao correta de conhecidos")
    ax.set_title("Conjunto aberto (teste); circulo vazado = EER por pares")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"{exp_id}_conjunto_aberto.png", dpi=130)

    results["runtime_seconds"] = round(time.time() - t0, 1)
    with open(out_dir / f"{exp_id}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em {out_dir}/ ({results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
