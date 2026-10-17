#!/usr/bin/env python3
"""
LSA（潜在语义分析 / Latent Semantic Indexing）图书检索引擎

方法（如实描述，不夸大）：
- 先用 TF-IDF 构造词-文档矩阵（与 book_search_recommender 同一套参数）
- 再用 TruncatedSVD 做潜在语义降维（把高维词空间压到低维"概念"空间，
  使字面不同但语义相近的词得以接近，例如 "car" 与 "automobile"）
- 检索时把查询投影到同一低维空间，用余弦相似度排序

所需依赖：仅 scikit-learn（无 torch / 无外部模型下载）

边界说明：
- LSA 是经典的信息检索方法（Deerwester 等, 1990），不是深度神经嵌入
- 只使用图书元数据（书名/作者/主题），不涉及任何借阅行为数据
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from book_search_recommender import build_text

DEFAULT_COMPONENTS = 100


class LsaSearchEngine:
    """TF-IDF + TruncatedSVD 潜在语义检索 + 内容推荐。"""

    name = "LSA (TF-IDF + TruncatedSVD)"

    def __init__(self, df: pd.DataFrame, n_components: int = DEFAULT_COMPONENTS):
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf = self.vectorizer.fit_transform(build_text(self.df))

        # 组件数不能超过矩阵维度
        max_comp = min(n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        self.n_components = max(int(max_comp), 2)

        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.normalizer = Normalizer(copy=False)
        # 归一化后，余弦相似度等价于点积
        self.doc_vectors = self.normalizer.fit_transform(self.svd.fit_transform(tfidf))

    def _project(self, text: str):
        """把文本投影到 LSA 潜在语义空间（归一化后单位向量）。"""
        vec = self.vectorizer.transform([text])
        vec = self.svd.transform(vec)
        return self.normalizer.transform(vec)

    def scores(self, query: str):
        """返回查询对全部图书的相似度分数数组（评测用，统一排序口径）。"""
        if not (query or "").strip():
            return np.zeros(len(self.df))
        q = self._project(query.strip())
        return cosine_similarity(q, self.doc_vectors).flatten()

    def search(self, query: str, top_n: int = 10) -> pd.DataFrame:
        """按查询词检索图书，返回带相似度分数的 DataFrame。"""
        query = (query or "").strip()
        if not query:
            return pd.DataFrame(columns=["score", *self.df.columns])
        return self._rank(self.scores(query), top_n)

    def similar_books(self, row_index: int, top_n: int = 8) -> pd.DataFrame:
        """返回与指定图书语义最相似的其他图书（不含自身）。"""
        if row_index < 0 or row_index >= len(self.df):
            return pd.DataFrame(columns=["score", *self.df.columns])
        scores = cosine_similarity(
            self.doc_vectors[row_index:row_index + 1], self.doc_vectors
        ).flatten()
        scores[row_index] = -1.0
        return self._rank(scores, top_n)

    def _rank(self, scores, top_n: int) -> pd.DataFrame:
        order = np.argsort(scores)[::-1]
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


if __name__ == "__main__":
    from book_search_recommender import load_catalog

    df = load_catalog()
    engine = LsaSearchEngine(df)
    print(f"目录规模: {len(df)} 本 | LSA 组件数: {engine.n_components}")
    print(f"文档语义向量矩阵: {engine.doc_vectors.shape}")

    for q in ["deep learning neural networks", "how machines learn from examples"]:
        print(f"\n检索: {q!r}")
        for _, r in engine.search(q, top_n=5).iterrows():
            print(f"  [{r['score']:.3f}] {r['title']} — {r['subject']}")
