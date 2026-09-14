"""Summarize training-asset distribution from existing metadata and references."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT.parent / "newton-cloth-rl/assets/cloth/garmentcodedata_style3d"
REFERENCE = ROOT / "data/benchmarks/large_support_v1_20260908/templates/templates"
SOURCES = {
    "展平": "data/datasets/shirt_flatten_reviewed_good_white/meta/clothdojo.json",
    "折叠": "data/datasets/shirt_fold_lr_allgood_runtime_target_nocs_20260903/meta/clothdojo.json",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(relative):
    path = ROOT / relative
    data = json.loads(path.read_text())
    source = Path(data["source"]["trajectory"])
    ids = sorted({e["source_trajectory"] for e in data["episodes"]
                  if not e["source_trajectory"].endswith("__mirror")})
    counts = collections.Counter()
    for tid in ids:
        manifest = json.loads((source / "trajectories" / tid / "manifest.json").read_text())
        counts[manifest["metadata"]["asset_id"]] += 1
    return counts, sha(path)


def asset_row(asset_id):
    reference = REFERENCE / (asset_id + ".npz")
    with np.load(reference, allow_pickle=False) as data:
        assert str(data["asset_id"]) == asset_id
        xy = np.ptp(data["particle_q"][:, :2].astype(float), axis=0)
        assert np.isclose(np.linalg.norm(xy), float(data["normalization_scale_m"]), atol=1e-6)
    specification = LEGACY / "official_sources" / asset_id / (asset_id + "_specification.json")
    result = {
        "asset_id": asset_id,
        "long_edge_cm": float(max(xy) * 100),
        "short_edge_cm": float(min(xy) * 100),
        "reference": str(reference.relative_to(ROOT)),
        "reference_sha256": sha(reference),
        "specification": str(specification) if specification.exists() else None,
        "has_sleeve": None, "has_hood": None, "has_skirt": None,
    }
    if specification.exists():
        panels = sorted(json.loads(specification.read_text())["pattern"]["panels"])
        result.update(
            specification_sha256=sha(specification),
            panel_names=panels,
            has_sleeve=any("sleeve" in name for name in panels),
            has_hood=any("hood" in name for name in panels),
            has_skirt=any("skirt" in name and not name.startswith("sl_") for name in panels),
        )
    return result


def percent(n, total):
    return f"{n}（{100*n/total:.1f}%）"


def main():
    datasets = {}
    for task, path in SOURCES.items():
        counts, fingerprint = collect(path)
        datasets[task] = {"source": path, "source_sha256": fingerprint, "counts": dict(sorted(counts.items()))}
    assets = {i: asset_row(i) for i in sorted(set().union(*(d["counts"] for d in datasets.values())))}
    assert len(datasets["展平"]["counts"]) == 182
    assert sum(datasets["展平"]["counts"].values()) == 889
    assert len(datasets["折叠"]["counts"]) == 156
    assert sum(datasets["折叠"]["counts"].values()) == 536
    rows = collections.defaultdict(list)
    for task, dataset in datasets.items():
        counts = dataset["counts"]
        values = np.array(list(counts.values()))
        selected = [assets[i] for i in counts]
        known = [a for a in selected if a["specification"]]
        rows["训练服装资产数"].append(str(len(counts)))
        rows["人工示范数（去重来源）"].append(str(values.sum()))
        rows["每件示范数：中位数 [P25, P75]"].append(
            f"{np.median(values):g} [{np.percentile(values,25):g}, {np.percentile(values,75):g}]")
        rows["每件示范数：最少–最多"].append(f"{values.min()}–{values.max()}")
        for lo, hi, label in [(1,2,"1–2 条示范的资产"),(3,5,"3–5 条示范的资产"),
                               (6,10,"6–10 条示范的资产"),(11,100000,"超过 10 条示范的资产")]:
            rows[label].append(percent(int(((values >= lo) & (values <= hi)).sum()),len(counts)))
        rows["示范最多的单件资产占全部示范"].append(f"{100*values.max()/values.sum():.1f}%")
        for key, label in [("long_edge_cm","展平参考长边 / cm：中位数 [最小, 最大]"),
                           ("short_edge_cm","展平参考短边 / cm：中位数 [最小, 最大]")]:
            sizes = [a[key] for a in selected]
            rows[label].append(f"{np.median(sizes):.1f} [{min(sizes):.1f}, {max(sizes):.1f}]")
        rows["可核验裁片结构的资产"].append(f"{len(known)}/{len(counts)}")
        for key, label in [("has_sleeve","有袖裁片"),("has_hood","含帽裁片"),("has_skirt","含裙身裁片（排除裙形袖口）")]:
            rows[label].append(percent(sum(a[key] for a in known),len(known)))
        rows["无袖裁片"].append(percent(sum(not a["has_sleeve"] for a in known),len(known)))
        rows["结构标签暂缺的资产"].append(str(len(counts)-len(known)))
    report = "# 专家示范训练资产分布\n\n"
    report += "统计口径：实际进入专家训练集的原始采集轨迹，按来源 ID 去重，不含镜像或清洗切分副本。不是全部采集尝试或自主 rollout 数据。\n\n"
    report += "| 统计项 | 展平 | 折叠 |\n| --- | ---: | ---: |\n"
    report += "".join(f"| {label} | {v[0]} | {v[1]} |\n" for label,v in rows.items())
    report += "\n尺寸覆盖全部训练资产，来自同一 large_support_free_settle_v1 展平参考。长边/短边是参考网格 XY 包围盒两边的较大/较小值，不是胸围、人体尺码或语义衣长。\n"
    report += "\n结构属性只对找到原始 specification 裁片文件的资产统计，百分比分母为对应任务的可核验资产数。帽、袖、裙身可重叠；不据此推断长袖/短袖、T 恤/卫衣等语义类别。缺失结构标签不计为无袖、无帽或无裙身。无袖指没有 sleeve 命名裁片。裙形袖口 sl_*_cuff_skirt 不计为裙身。\n"
    report += "\n折叠的 156 件资产均在展平的 182 件资产集合内，两任务资产数不可相加作为唯一服装总数。每件示范数量明显不均衡；这里只描述已采集数据，不声称预先按类别均衡采样。\n"
    report += "\n## 来源与复核\n\n"
    for task,d in datasets.items():
        report += f"- {task}：{d['source']}，SHA256 {d['source_sha256']}。\n"
    report += "\n逐资产计数、尺寸、裁片标签及文件指纹见相邻 asset_distribution_20260910.json。执行 python paper/experiments/build_asset_distribution.py 可重新生成 JSON（含本表文本），无需仿真或 GPU。\n"
    print(json.dumps({"datasets": datasets, "assets": assets, "table_rows": rows, "report_markdown": report},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
