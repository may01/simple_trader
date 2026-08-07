"""Mechanical md tables from the results JSON. Written analysis goes on top."""
from . import config

KINDS = ("strict", "plain")


def kind_mi(cell: dict, kind: str, split: str) -> float:
    cols = [f"{kind}_n{n}_{s}" for n in (1, 2) for s in ("long", "short")]
    return sum(cell[split]["labels"][c]["mi"] for c in cols) / len(cols)


def _rank_table(cells, kind: str, top: int = 15) -> str:
    rows = sorted(cells, key=lambda c: (-kind_mi(c, kind, "oos"),
                                        -kind_mi(c, kind, "train")))
    lines = ["| window | tf | technique | classes | oos MI | train MI | mono ρ (oos) |",
             "|---|---|---|---|---|---|---|"]
    for c in rows[:top]:
        rho = c["oos"]["pairs"][f"{kind}_n1"]["monotonicity_rho"]
        rho_str = f"{rho:.2f}" if rho is not None else "n/a"
        lines.append(
            f"| {c['window']} | {c['tf']} | {c['technique']} | {c['n_classes']} "
            f"| {kind_mi(c, kind, 'oos'):.5f} | {kind_mi(c, kind, 'train'):.5f} "
            f"| {rho_str} |")
    return "\n".join(lines)


def _class_count_table(cells, kind: str) -> str:
    lines = ["| classes | mean oos MI | best cell |", "|---|---|---|"]
    for nc in config.CLASS_COUNTS:
        sub = [c for c in cells if c["n_classes"] == nc]
        mean = sum(kind_mi(c, kind, "oos") for c in sub) / len(sub)
        best = max(sub, key=lambda c: kind_mi(c, kind, "oos"))
        lines.append(f"| {nc} | {mean:.5f} | ma{best['window']} tf{best['tf']} "
                     f"{best['technique']} ({kind_mi(best, kind, 'oos'):.5f}) |")
    return "\n".join(lines)


def _technique_table(cells, kind: str) -> str:
    lines = ["| technique | mean oos MI | mean train MI |", "|---|---|---|"]
    for t in config.TECHNIQUES:
        sub = [c for c in cells if c["technique"] == t]
        lines.append(f"| {t} | "
                     f"{sum(kind_mi(c, kind, 'oos') for c in sub) / len(sub):.5f} | "
                     f"{sum(kind_mi(c, kind, 'train') for c in sub) / len(sub):.5f} |")
    return "\n".join(lines)


def _flip_table(cells) -> str:
    lines = ["| window | tf | technique | kind | split | flips (of 4) | extra pop − / + |",
             "|---|---|---|---|---|---|---|"]
    for c in cells:
        if c["n_classes"] != 7:
            continue
        for split in ("train", "oos"):
            pop = c[split]["population"]
            for kind in KINDS:
                flips = 0
                for n in (1, 2):
                    fl = c[split]["pairs"][f"{kind}_n{n}"]["flip"]
                    for side in ("neg", "pos"):
                        flips += bool(fl[side]["sign_flip"])
                lines.append(
                    f"| {c['window']} | {c['tf']} | {c['technique']} | {kind} "
                    f"| {split} | {flips} | {pop['-3']} / {pop['3']} |")
    return "\n".join(lines)


def _counts(meta: dict) -> str:
    lines = ["| split | tf | closed rows | NaN diffs (8/12/24) |", "|---|---|---|---|"]
    for split in ("train", "oos"):
        for tf, n in meta["row_counts"][split].items():
            d = meta["dropped_nan_feature"][split][tf]
            lines.append(f"| {split} | {tf} | {n} | {d['8']}/{d['12']}/{d['24']} |")
    return "\n".join(lines)


def render(result: dict) -> str:
    cells = result["cells"]
    parts = ["# RSI Parameters Selection — Results",
             "",
             f"Source: `rsi_parameters_selection_results.json` "
             f"({result['meta'].get('parts_used', '?')} parts).",
             "", "## Ranking — strict", "", _rank_table(cells, "strict"),
             "", "## Ranking — non-strict", "", _rank_table(cells, "plain"),
             "", "## 5 vs 7 classes", "",
             "strict:", _class_count_table(cells, "strict"), "",
             "non-strict:", _class_count_table(cells, "plain"),
             "", "## Technique comparison", "",
             "strict:", _technique_table(cells, "strict"), "",
             "non-strict:", _technique_table(cells, "plain"),
             "", "## Flip test (7-class)", "", _flip_table(cells),
             "", "## Row counts & drops", "", _counts(result["meta"]), ""]
    return "\n".join(parts)


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "external/docs/superpowers/experiment/results/"
        "rsi_parameters_selection_results.json")
    md = render(json.load(open(path)))
    out = path.replace(".json", ".md")
    with open(out, "w") as fh:
        fh.write(md)
    print(f"wrote {out}")
