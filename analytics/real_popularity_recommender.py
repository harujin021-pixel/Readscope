#!/usr/bin/env python3
"""
真实热度读书推荐器（Real-Popularity Recommender）

把“语义相似”与“真实借阅热度”结合：
- 语义相似：沿用 LsaSearchEngine（TF-IDF + TruncatedSVD），对合并语料检索
- 真实热度：来自 Seattle Public Library 的公开借阅数据（SPL，Public Domain）

合并语料：
- 360 本 Open Library 批量抓取图书（source = catalog）
- SPL 2018–2025 年 Top30 印刷图书（source = spl，带真实 checkouts 与上榜年数）

设计原则：
- 不修改任何原始数据文件；只读 SPL CSV 与目录 CSV
- SPL 条目没有 subject 字段，用 title 的文本参与语义匹配
- 热度只用于排序与展示，不参与语义向量计算，保证相似性客观
- 结果中明确标注来源，避免把 Open Library 目录当成“真实图书馆借阅记录”
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from semantic_search import LsaSearchEngine

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CATALOG_CSV = PROCESSED_DIR / "books_openlibrary_batch.csv"
SPL_TOP_BY_YEAR_CSV = PROCESSED_DIR / "spl_top_titles_by_year.csv"


def _norm_title(title: str) -> str:
    """归一化书名用于去重：取主标题段（冒号/斜杠/括号前），小写、去标点、保留前4词。"""
    t = (title or "").lower()
    t = re.split(r"[/:(\[]", t)[0]
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(t.split()[:4])


# ---------------------------------------------------------------- 数据加载

def load_catalog() -> pd.DataFrame:
    df = pd.read_csv(CATALOG_CSV, dtype=str).fillna("")
    df["source"] = "catalog"
    return df


def load_spl_books() -> pd.DataFrame:
    """从逐年 Top30 聚合：按归一化书名合并所有格式，累加 2018–2025 借阅量。"""
    by_year = pd.read_csv(SPL_TOP_BY_YEAR_CSV, dtype=str).fillna("")
    by_year["checkouts"] = pd.to_numeric(by_year["checkouts"], errors="coerce").fillna(0)
    by_year["year"] = pd.to_numeric(by_year["year"], errors="coerce").fillna(0).astype(int)
    by_year["key"] = by_year["title"].apply(_norm_title)

    records = []
    for key, g in by_year.groupby("key"):
        # 展示用标题：取借阅量最高的一条原始标题
        best = g.sort_values("checkouts", ascending=False).iloc[0]
        records.append({
            "title": best["title"],
            "creator": best["creator"],
            "material_type": "MULTI" if g["material_type"].nunique() > 1 else best["material_type"],
            "checkouts": int(g["checkouts"].sum()),
            "years_on_chart": int(g["year"].nunique()),
        })

    df = pd.DataFrame(records)
    df["source"] = "spl"
    return df


def normalize_for_engine(catalog: pd.DataFrame, spl: pd.DataFrame) -> pd.DataFrame:
    """
    把两个数据集对齐到 LsaSearchEngine 需要的列名。
    catalog: title, author, publication_year, publisher, subject, isbn, ...
    spl:     title, creator, material_type, checkouts, years_on_chart, ...
    """
    catalog = catalog.copy()
    catalog["author"] = catalog.get("author", "")
    catalog["subject"] = catalog.get("subject", "")
    catalog["publication_year"] = catalog.get("publication_year", "")

    spl = spl.copy()
    spl["author"] = spl.get("creator", "")
    spl["subject"] = ""  # SPL 当前切片未含 Subjects，语义匹配主要靠 title/author
    spl["publication_year"] = ""

    cols = ["title", "author", "subject", "publication_year", "source",
            "checkouts", "years_on_chart", "material_type"]
    for c in cols:
        if c not in catalog.columns:
            catalog[c] = ""
        if c not in spl.columns:
            spl[c] = ""
    combined = pd.concat([catalog[cols], spl[cols]], ignore_index=True)
    return combined.fillna("")


# ---------------------------------------------------------------- 推荐引擎

class RealPopularityRecommender:
    """LSA 语义检索 + SPL 真实热度排序。"""

    def __init__(self):
        self.catalog = load_catalog()
        self.spl = load_spl_books()
        self.df = normalize_for_engine(self.catalog, self.spl)
        self.engine = LsaSearchEngine(self.df)

    def search(self, query: str, top_n: int = 10) -> pd.DataFrame:
        """按查询词返回相似图书，附带真实借阅热度（若来源为 SPL）。"""
        res = self.engine.search(query, top_n=top_n)
        if res.empty:
            return res
        return self._enrich(res)

    def similar_books(self, title_keyword: str, top_n: int = 8) -> pd.DataFrame:
        """以某本书为种子，返回语义相似图书，并标注真实热度。"""
        idx = self.df.index[
            self.df["title"].str.contains(title_keyword, case=False, na=False)
        ]
        if len(idx) == 0:
            return pd.DataFrame()
        res = self.engine.similar_books(int(idx[0]), top_n=top_n)
        if res.empty:
            return res
        return self._enrich(res)

    def _enrich(self, res: pd.DataFrame) -> pd.DataFrame:
        """把数值列转为可读类型，并给出热度解释字段。"""
        res = res.copy()
        res["checkouts"] = pd.to_numeric(res["checkouts"], errors="coerce").fillna(0).astype(int)
        res["years_on_chart"] = pd.to_numeric(res["years_on_chart"], errors="coerce").fillna(0).astype(int)
        res["heat_note"] = res.apply(self._heat_note, axis=1)
        return res

    @staticmethod
    def _heat_note(row) -> str:
        if row["source"] == "spl":
            years = f"，连续 {int(row['years_on_chart'])} 年进 SPL Top30" if row["years_on_chart"] > 0 else ""
            return f"SPL 真实借阅 {int(row['checkouts']):,} 次{years}"
        return "馆藏目录（无真实借阅数据）"


# ---------------------------------------------------------------- CLI 测试

def format_demo(results: pd.DataFrame) -> str:
    if results.empty:
        return "（无结果）"
    lines = []
    for _, r in results.iterrows():
        src = "SPL" if r["source"] == "spl" else "目录"
        heat = f" | {r['heat_note']}" if r["source"] == "spl" else ""
        lines.append(
            f"  [{r['score']:.3f}] [{src}] {r['title']} — {r['author']}"
            f" ({r['publication_year'] or '?'}){heat}"
        )
    return "\n".join(lines)


def _row_to_md(r) -> str:
    src = "SPL" if r["source"] == "spl" else "目录"
    heat = f" | {r['heat_note']}" if r["source"] == "spl" else ""
    return f"| {r['score']:.3f} | {src} | {r['title']} | {r['author']} | {r['publication_year'] or '—'}{heat} |"


def save_demo_report(rec: RealPopularityRecommender, path: Path):
    queries = [
        ("artificial intelligence machine learning", "AI / 机器学习"),
        ("social media society", "社交媒体与社会"),
        ("memoir personal growth", "回忆录 / 个人成长"),
        ("history politics power", "历史 / 政治 / 权力"),
    ]
    seeds = ["Educated", "Where the Crawdads Sing", "Becoming"]

    lines = [
        "# 真实热度读书推荐器 — 演示报告",
        "",
        "## 语料",
        "",
        f"- 合并语料：**{len(rec.df)}** 条（Open Library 目录 {len(rec.catalog)} + SPL 真实热门 {len(rec.spl)}）",
        f"- LSA 组件数：{rec.engine.n_components}",
        "",
        "## 检索演示",
        "",
    ]
    for q, label in queries:
        lines.append(f"### {label}（`{q}`）")
        lines.append("")
        lines.append("| 相似度 | 来源 | 书名 | 作者 | 年份 | 热度 |")
        lines.append("|---|---|---|---|---|---|")
        res = rec.search(q, top_n=6)
        if res.empty:
            lines.append("| — | — | 无结果 | — | — | — |")
        else:
            for _, r in res.iterrows():
                lines.append(_row_to_md(r))
        lines.append("")

    lines.append("## 以书找书演示")
    lines.append("")
    for seed in seeds:
        lines.append(f"### 与《{seed}》相似")
        lines.append("")
        lines.append("| 相似度 | 来源 | 书名 | 作者 | 年份 | 热度 |")
        lines.append("|---|---|---|---|---|---|")
        res = rec.similar_books(seed, top_n=6)
        if res.empty:
            lines.append("| — | — | 未找到种子书 | — | — | — |")
        else:
            for _, r in res.iterrows():
                lines.append(_row_to_md(r))
        lines.append("")

    lines.extend([
        "## 已知局限（如实说明）",
        "",
        '1. LSA 基于词袋，短标题可能出现表面模式误配（如 "a memoir / X" 与 "a cookbook / Y" 被拉近）。',
        "2. SPL 数据仅覆盖西雅图公共图书馆 2018–2025 年 Top30 条目，不等于全库热度。",
        "3. Open Library 目录与 SPL 语料主题分布不同，跨源相似度可能偏向某一方。",
        "4. 热度不参与语义向量计算，仅用于标注与排序辅助。",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"演示报告已保存：{path}")


def main():
    rec = RealPopularityRecommender()
    print(f"合并语料：{len(rec.df)} 条（catalog={len(rec.catalog)}, spl={len(rec.spl)}）")
    print(f"LSA 组件数：{rec.engine.n_components}\n")

    queries = [
        "artificial intelligence machine learning",
        "social media society",
        "memoir personal growth",
    ]
    for q in queries:
        print(f"检索: {q!r}")
        print(format_demo(rec.search(q, top_n=6)))
        print()

    seed = "Educated"
    print(f"以 {seed!r} 为种子的相似推荐：")
    print(format_demo(rec.similar_books(seed, top_n=6)))

    report_path = BASE_DIR / "analytics" / "output" / "real_popularity_recommender_demo.md"
    save_demo_report(rec, report_path)


if __name__ == "__main__":
    main()
