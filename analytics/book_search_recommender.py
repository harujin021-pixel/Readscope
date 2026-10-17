#!/usr/bin/env python3
"""
图书智能检索与相似推荐模块（ReadScope AI 层）

方法（如实描述，不夸大）：
- 文本向量化：TF-IDF（sklearn TfidfVectorizer，英文停用词，1-2 元词组）
- 相似度：余弦相似度（cosine similarity）
- 检索：用户查询词与图书文本（书名/作者/主题加权拼接）计算相似度排序
- 推荐：以某本书为查询，返回内容最相似的其他图书（content-based，非协同过滤）

边界说明：
- 这是基于内容的关键词语义匹配，不是大语言模型生成，也不是协同过滤推荐
- 只使用图书元数据（书名/作者/主题），不涉及任何借阅行为数据
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

BASE_DIR = Path(__file__).parent.parent
BATCH_CSV = BASE_DIR / "data" / "processed" / "books_openlibrary_batch.csv"

# 字段权重：书名和主题重复拼接以加权，作者单份
FIELD_WEIGHTS = {"title": 3, "subject": 2, "author": 1}


def load_catalog(csv_path: Path = BATCH_CSV) -> pd.DataFrame:
    """加载目录 CSV，全字段按字符串读取，空值填充为空串。"""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    return df


def build_text(df: pd.DataFrame) -> list:
    """按字段权重拼接每本书的检索文本。"""
    texts = []
    for _, r in df.iterrows():
        parts = []
        for field, weight in FIELD_WEIGHTS.items():
            val = str(r.get(field, "")).strip()
            if val:
                parts.extend([val] * weight)
        texts.append(" ".join(parts))
    return texts


class BookSearchRecommender:
    """TF-IDF 检索 + 余弦相似度内容推荐。"""

    name = "TF-IDF (lexical)"

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(build_text(self.df))

    def scores(self, query: str):
        """返回查询词对全部图书的相似度分数数组（评测用，统一排序口径）。"""
        if not (query or "").strip():
            return np.zeros(self.matrix.shape[0])
        q_vec = self.vectorizer.transform([query.strip()])
        return cosine_similarity(q_vec, self.matrix).flatten()

    def search(self, query: str, top_n: int = 10) -> pd.DataFrame:
        """按查询词检索图书，返回带相似度分数的 DataFrame。"""
        query = (query or "").strip()
        if not query:
            return pd.DataFrame(columns=["score", *self.df.columns])
        return self._rank(self.scores(query), top_n)

    def similar_books(self, row_index: int, top_n: int = 8) -> pd.DataFrame:
        """返回与指定图书内容最相似的其他图书（不含自身）。"""
        if row_index < 0 or row_index >= len(self.df):
            return pd.DataFrame(columns=["score", *self.df.columns])
        scores = linear_kernel(self.matrix[row_index], self.matrix).flatten()
        scores[row_index] = -1.0  # 排除自身
        return self._rank(scores, top_n)

    def _rank(self, scores, top_n: int) -> pd.DataFrame:
        """按分数降序取前 N，过滤掉分数 <= 0 的结果。"""
        order = scores.argsort()[::-1]
        rows = []
        for idx in order[:top_n]:
            if scores[idx] <= 0:
                break
            row = self.df.iloc[idx].to_dict()
            row["score"] = round(float(scores[idx]), 4)
            rows.append(row)
        out = pd.DataFrame(rows)
        cols = ["score"] + [c for c in self.df.columns if c != "score"]
        return out[cols] if not out.empty else pd.DataFrame(columns=cols)


def format_results(results: pd.DataFrame, columns=None) -> pd.DataFrame:
    """格式化展示列：中文列名 + 分数。"""
    if results.empty:
        return results
    col_map = {
        "title": "书名",
        "author": "作者",
        "publication_year": "出版年份",
        "publisher": "出版社",
        "subject": "主题",
        "isbn": "ISBN",
        "score": "相似度",
    }
    cols = columns or [c for c in ["title", "author", "publication_year", "subject", "score"] if c in results.columns]
    out = results[cols].rename(columns=col_map)
    return out


if __name__ == "__main__":
    # 命令行自测
    df = load_catalog()
    rec = BookSearchRecommender(df)
    print(f"目录规模: {len(df)} 本，TF-IDF 矩阵: {rec.matrix.shape}")

    for q in ["deep learning neural networks", "social media and society", "python programming"]:
        print(f"\n检索: {q!r}")
        res = rec.search(q, top_n=5)
        for _, r in res.iterrows():
            print(f"  [{r['score']:.3f}] {r['title']} — {r['author']} ({r['publication_year']})")

    print("\n推荐: 与 'The Book of Why' 相似的书")
    idx = df.index[df["title"].str.contains("Book of Why", case=False)][0]
    sim = rec.similar_books(int(idx), top_n=5)
    for _, r in sim.iterrows():
        print(f"  [{r['score']:.3f}] {r['title']} — {r['author']} ({r['subject']})")
