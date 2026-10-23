"""E02 - Galeria (prototipo x vizinho mais proximo) e threshold calibrado no conjunto aberto.

Usa os MESMOS embeddings do E01 (le o cache; nao recalcula o FaceNet).

Uso:
    python experiments/run_e02.py --config configs/e02_galeria_threshold.yaml
    python experiments/run_e02.py --config configs/e02_galeria_threshold.yaml --synthetic   # teste rapido
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
from facerec.metrics import make_pairs, pair_scores, roc_eer  # noqa: E402
from facerec.openset import (build_open_set, collect_unknown_scores, evaluate_open_set,  # noqa: E402
                             identity_scores, open_set_curve, threshold_for_far)
from facerec.splits import check_no_leakage, split_identities  # noqa: E402

METHOD_NAMES = {"prototype": "prototipo (media)", "nearest": "vizinho mais proximo"}


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
    os_cfg = cfg["open_set"]
    t0 = time.time()

    if args.synthetic:
        emb, labels = synthetic_embeddings(seed=seed)
        images = None
    else:
        d, m = cfg["dataset"], cfg["model"]
        images, labels, _names = load_lfw(d["min_faces_per_person"], d["crop"], d["image_size"])
        emb = cached_embeddings(cfg["embeddings_cache"], lambda: extract_embeddings(
            images, m["batch_size"], m["device"], m["pretrained"]))

    splits = split_identities(labels, cfg["split"]["fractions"], seed)
    check_no_leakage(splits, labels, images)
    print("Checagem de leakage: OK (identidades disjuntas entre treino/validacao/teste)")

    # --- Estrategia A: threshold de EER por pares, escolhido na validacao (igual ao E01)
    a, b, y = make_pairs(labels[splits["val"]], cfg["pairs"]["n_val_pairs"], seed)
    s_val = pair_scores(emb, splits["val"][a], splits["val"][b], "cosine")
    thr_eer = roc_eer(s_val, y)[0]["eer_threshold"]

    # --- Conjunto aberto do TESTE: mesmas pessoas conhecidas/desconhecidas e mesmo cadastro do E01
    test_ids = np.random.default_rng(seed).permutation(np.unique(labels[splits["test"]]))
    n_known = int(round(os_cfg["known_fraction"] * len(test_ids)))
    known_ids, unknown_ids = test_ids[:n_known], test_ids[n_known:]
    test_set = build_open_set(labels, known_ids, unknown_ids, os_cfg["n_enroll"], seed)

    # --- Checagem de consistencia com o E01 (prototipo + threshold de EER por pares)
    e01 = open_set_identification(emb, labels, known_ids, unknown_ids, thr_eer,
                                  os_cfg["n_enroll"], seed, "cosine")
    sc_p, ids_p = identity_scores(emb, test_set, "prototype")
    e02 = evaluate_open_set(sc_p, ids_p, test_set, thr_eer)
    same = all(np.isclose(e01[k], e02[k]) for k in
               ("correct_identification_rate", "false_acceptance_rate_unknown", "open_set_accuracy"))
    print(f"Consistencia com o E01 (prototipo + EER por pares): {'OK, mesmos numeros' if same else 'DIFERENTE!'}")

    val_ids = np.unique(labels[splits["val"]])
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
        "consistent_with_e01": bool(same),
        "metrics": {},
    }

    fig, (ax_c, ax_h) = plt.subplots(1, 2, figsize=(11, 4))
    header = f"{'galeria':<22}{'threshold':<16}{'thr':>8}{'id_correta':>12}{'desc_aceito':>13}{'FRR_conh':>10}{'erro_pessoa':>13}{'acc_aberto':>12}"
    print("\n" + header)
    primary_name = f"far_{os_cfg['primary_far_target']}"

    for method in os_cfg["gallery_methods"]:
        scores, ids = identity_scores(emb, test_set, method)
        # 2) Estrategia B: threshold calibrado SO na validacao, para cada alvo de FAR
        unknown_val = collect_unknown_scores(emb, labels, val_ids, method, os_cfg["known_fraction"],
                                             os_cfg["n_enroll"], os_cfg["n_calibration_repeats"], seed)
        thresholds = {"pair_eer": thr_eer}
        for t in os_cfg["far_targets"]:
            thresholds[f"far_{t}"] = threshold_for_far(unknown_val, t)

        results["metrics"][method] = {}
        for name, thr in thresholds.items():
            r = evaluate_open_set(scores, ids, test_set, thr)   # o teste so REPORTA
            results["metrics"][method][name] = r
            print(f"{METHOD_NAMES[method]:<22}{name:<16}{thr:>8.4f}{r['correct_identification_rate']:>12.4f}"
                  f"{r['false_acceptance_rate_unknown']:>13.4f}{r['false_rejection_rate_known']:>10.4f}"
                  f"{r['misidentification_rate_known']:>13.4f}{r['open_set_accuracy']:>12.4f}")

        # grafico: curva no teste (so para mostrar) + pontos de operacao escolhidos na validacao
        _, far, cir = open_set_curve(scores, ids, test_set)
        line, = ax_c.plot(far, cir, label=METHOD_NAMES[method])
        for name, thr in thresholds.items():
            r = results["metrics"][method][name]
            ax_c.plot(r["false_acceptance_rate_unknown"], r["correct_identification_rate"], "o",
                      color=line.get_color(), ms=5, mfc="white" if name == "pair_eer" else line.get_color())
        if method == "prototype":
            top = scores.max(axis=1)
            ax_h.hist(top[test_set["probe_known"]], bins=40, alpha=0.6, label="conhecidos")
            ax_h.hist(top[~test_set["probe_known"]], bins=40, alpha=0.6, label="desconhecidos")
            ax_h.axvline(thr_eer, color="k", ls="--", label="threshold EER por pares (E01)")
            ax_h.axvline(thresholds[primary_name], color="r", ls="--", label=f"threshold FAR {os_cfg['primary_far_target']} (val)")

    ax_c.set_xlabel("desconhecidos aceitos (FAR)")
    ax_c.set_ylabel("identificacao correta de conhecidos")
    ax_c.set_title("Conjunto aberto (teste); circulo vazado = EER por pares")
    ax_c.legend(fontsize=8)
    ax_h.set_xlabel("melhor similaridade (cosseno)")
    ax_h.set_title("Prototipo: conhecidos x desconhecidos (teste)")
    ax_h.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"{exp_id}_conjunto_aberto.png", dpi=130)

    results["runtime_seconds"] = round(time.time() - t0, 1)
    with open(out_dir / f"{exp_id}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em {out_dir}/ ({results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
