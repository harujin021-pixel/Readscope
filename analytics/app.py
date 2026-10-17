#!/usr/bin/env python3
"""
ReadScope — Streamlit dashboard for book catalog analytics.

Data rules:
1. TEST_SAMPLE (manually constructed) and OPEN_LIBRARY_API (public API data) are shown
   separately by default, and are never merged or presented as a real library collection.
2. No circulation volume, popular-loan lists, reader preferences, or overdue rates are
   generated or inferred from book metadata.
3. Missing fields are kept empty and counted explicitly; charts are hidden with an
   explanation when a field has no valid data.
4. Read-only display; no data files are modified.

Statistics: reuses the analysis functions in analytics/book_catalog_analyzer.py so the
numbers stay consistent with the Markdown analysis report and catalog_statistics.csv.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chinese font config: matplotlib's default DejaVu Sans has no CJK glyphs, which garbles
# titles/axis labels. Try Linux (Noto, deployed via packages.txt) then Windows Chinese fonts
# in order, and let the Chinese font render minus signs (otherwise minus signs show as boxes).
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import pandas as pd
import streamlit as st

# Reuse the statistics logic from the existing analysis module (same directory)
# to avoid divergent re-implementations.
from book_catalog_analyzer import (
    analyze_authors,
    analyze_publication_years,
    analyze_publishers,
    analyze_subjects,
    generate_quality_overview,
)
from book_search_recommender import BookSearchRecommender, format_results, load_catalog
from semantic_search import LsaSearchEngine
from real_popularity_recommender import RealPopularityRecommender

# Chinese descriptions of quality flags are defined centrally in the import module; reuse them here.
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "import"))
from import_book_metadata import QUALITY_FLAGS  # noqa: E402

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# English descriptions for the quality flags (Chinese originals live in import_book_metadata.py)
QUALITY_FLAGS_EN = {
    "MISSING_TITLE": "Missing title",
    "MISSING_AUTHOR": "Missing author",
    "MISSING_YEAR": "Missing publication year",
    "INVALID_YEAR": "Invalid year format",
    "MISSING_SUBJECT": "Missing subject",
    "MISSING_SOURCE": "Missing source",
    "MISSING_SOURCE_ID": "Missing source record ID",
    "DUPLICATE_ISBN": "Duplicate ISBN",
}

# Preset datasets: keys double as radio option identifiers and must stay unchanged.
# Language-dependent label/badge/nature live in I18N under "ds_<key>".
# Dict order sets the radio order; the 360-book real catalog is first so reviewers
# land on it by default instead of the manual test sample.
DATASETS = {
    "OPEN_LIBRARY_BATCH — 批量API目录（360本）": {
        "path": PROCESSED_DIR / "books_openlibrary_batch.csv",
    },
    "OPEN_LIBRARY_API — 公开API数据": {
        "path": PROCESSED_DIR / "books_from_api.csv",
    },
    "TEST_SAMPLE — 人工测试样本": {
        "path": PROCESSED_DIR / "books_cleaned.csv",
    },
}

# ---------------------------------------------------------------- i18n
I18N = {
    "zh": {
        "app_caption": "图书目录分析仪表板 · Book Catalog Analytics (demo)",
        "select_dataset": "选择数据集",
        "usage_expander": "数据使用声明",
        "usage_rules": (
            "- TEST_SAMPLE 与 OPEN_LIBRARY_API 数据分开展示，不合并宣称代表真实图书馆藏书\n"
            "- 图书元数据**不能**推断借阅量、热门借阅图书、读者偏好或逾期率\n"
            "- 缺失字段保留空值，不自行编造\n"
            "- 本页面只读展示，不修改任何数据文件"
        ),
        "main_title": "图书目录分析仪表板 · Book Catalog Analytics",
        "load_error": "数据加载失败：",
        "run_import_hint": "请先运行 data/import/import_book_metadata.py 生成清洗后的数据文件。",
        "current_dataset": "当前数据集：",
        # 数据概览
        "total_records": "总记录数",
        "valid_records": "有效记录（PASS）",
        "invalid_records": "问题记录",
        "valid_rate": "有效率",
        "source_dist": "数据来源分布（source 字段）",
        "source_missing_caption": "source 字段不存在或全部缺失，无法展示来源分布。",
        "dup_isbn_header": "重复 ISBN 记录",
        "dup_isbn_metric": "标记为 DUPLICATE_ISBN 的记录数",
        "missing_header": "关键字段缺失概况",
        "col_field": "字段",
        "col_missing": "缺失数量",
        "flags_header": "质量标记分布",
        "col_flag": "质量标记",
        "col_desc": "说明",
        "col_count": "数量",
        "no_flags": "当前数据集没有质量标记，全部记录为 PASS。",
        # 目录分析
        "catalog_caption": "统计口径与 analytics/book_catalog_analyzer.py 完全一致；图表仅基于下方筛选后的记录。",
        "filter_source": "按数据来源筛选",
        "source_missing_filter": "source 字段不存在或全部缺失，无来源筛选。",
        "filter_year": "按出版年份筛选（缺失/非法年份记录 {n} 条，筛选时不计入）",
        "no_year_filter": "没有有效出版年份数据，无年份筛选。",
        "filter_author": "按作者筛选（最多可选 20 位）",
        "no_author_filter": "没有作者数据，无作者筛选。",
        "filter_subject": "按分类筛选",
        "no_subject_filter": "没有分类数据，无分类筛选。",
        "no_filtered_records": "当前筛选条件下没有记录，请调整筛选器。",
        "filtered_count": "筛选后记录数：{a} / {b}",
        "year_dist_header": "出版年份分布（{n} 条有效年份记录）",
        "chart_year_title": "出版年份分布",
        "chart_year_xlabel": "出版年份",
        "chart_ylabel": "图书数量",
        "year_range_caption": "年份范围 {min_y}–{max_y}，平均 {mean:.1f}；缺失 {miss} 条，非法 {inv} 条（不参与绘图）。",
        "no_year_data": "筛选后没有有效年份数据，无法绘制年份分布图。",
        "top_authors_header": "作者作品数量 Top10（唯一作者 {n} 位）",
        "top_authors_title": "作者作品数量 Top10",
        "multi_author_note": "多作者图书按分号拆分后分别计数，因此计数总和可能大于记录数。",
        "no_author_data": "筛选后没有作者数据。",
        "top_publishers_header": "出版社分布 Top10（唯一出版社 {n} 家）",
        "top_publishers_title": "出版社分布 Top10",
        "no_publisher_data": "筛选后没有出版社数据。",
        "subjects_header": "分类分布（唯一分类 {n} 种）",
        "top_subjects_title": "分类分布 Top10",
        "one_subject_only": "筛选后只有 1 种分类「{name}」（{cnt} 条），变化不足，不适合绘制分布图。",
        "no_subject_data": "筛选后没有分类数据。",
        # 数据质量
        "quality_caption": "仅展示带质量标记的问题记录（data_quality_flags != PASS）。",
        "no_flagged": "当前数据集没有问题记录，全部为 PASS。",
        "filter_flags": "按问题类型筛选（不选 = 显示全部问题记录）",
        "flag_option": "{flag}（{desc}）",
        "flagged_count": "问题记录数",
        "download_csv": "下载当前筛选结果 CSV",
        # 智能检索与推荐
        "ai_method_caption": (
            "方法：TF-IDF 文本向量 + 余弦相似度（基于书名/作者/主题的内容匹配）。"
            "可切换 LSA（潜在语义）——在 TF-IDF 基础上做 SVD 降维，使字面不同但语义相近的词相互接近。"
            "**不是**大语言模型生成，也不使用任何借阅行为数据。"
        ),
        "method_label": "检索方法",
        "opt_tfidf": "TF-IDF（词汇匹配，基线）",
        "opt_lsa": "LSA（潜在语义，TruncatedSVD）",
        "note_lsa": "LSA：词-文档 TF-IDF 矩阵经 SVD 降至低维“概念空间”后计算余弦相似度。",
        "note_tfidf": "TF-IDF：字面词的加权重合度 + 余弦相似度（词汇匹配基线）。",
        "batch_missing_error": "批量目录文件缺失，无法使用智能检索：",
        "run_fetch_hint": "请先运行 data/import/fetch_openlibrary_batch.py 生成 books_openlibrary_batch.csv。",
        "engine_info": "当前引擎：{engine} ｜ 检索语料：OPEN_LIBRARY_BATCH 批量目录 {n} 本真实图书（独立于上方所选数据集）",
        "eval_expander": "TF-IDF vs LSA 评测结果（14 条查询，元数据代理相关性）",
        "eval_design": (
            "评测设计：7 个主题各配 1 条 literal（字面）与 1 条 paraphrase（语义改写）查询，"
            "相关性以图书 `subject` 精确匹配定义（**元数据代理相关性，非人工判断**）。"
        ),
        "by_type_caption": "按查询类型拆分：",
        "eval_conclusion": (
            "结论：LSA 在字面查询上优于 TF-IDF，但 paraphrase 查询两种方法都表现差——"
            "语料小、字段短，潜在语义训不动。完整报告见 analytics/output/semantic_search_eval_report.md"
        ),
        "eval_missing": "尚未生成评测结果，请运行 `python analytics/semantic_search_eval.py`。",
        "query_label": "输入感兴趣的领域或关键词，例如：deep learning / social media / climate psychology",
        "query_placeholder": "例如：deep learning neural networks",
        "search_btn": "检索",
        "empty_query": "请先输入检索词。",
        "no_results": "没有匹配的图书，试试更通用的英文关键词。",
        "results_header": "检索结果 Top{n}",
        "similarity_note": "相似度 = 查询词与图书文本的余弦相似度，范围 0-1，越高越相关。",
        "similar_header": "相似图书推荐",
        "similar_caption": "先按书名关键词找到一本书，系统再用**当前引擎（{engine}）**基于内容相似度推荐最接近的其他图书。",
        "pick_label": "按书名关键词找书",
        "pick_placeholder": "例如：Book of Why",
        "no_title_match": "没有找到匹配的书名，试试更短的关键词。",
        "pick_radio": "找到 {n} 本匹配图书，选择一本：",
        "current_book": "**当前图书**：{title} — {author} （{year}，{subject}）",
        "no_similar": "这本书在当前语料中没有找到相似度 > 0 的其他图书。",
        "rec_basis": "推荐依据：内容相似度，与借阅行为无关。",
        # 真实热度推荐
        "heat_caption": (
            "把 **LSA 语义相似** 与 **SPL 真实借阅热度** 结合："
            "先找到语义相近的书，再告诉你这些书在西雅图公共图书馆被真实借阅了多少次、连续几年上榜。"
            "数据来源：Seattle Public Library（Public Domain），2018–2025 年 Top30 条目。"
        ),
        "heat_data_missing": "缺少必要数据文件：",
        "heat_files_hint": "请确保 data/processed/ 下存在 books_openlibrary_batch.csv 与 spl_top_titles_by_year.csv。",
        "heat_corpus": "合并语料：{n} 条 （Open Library 目录 {a} + SPL 真实热门 {b}）",
        "mode_label": "推荐方式",
        "mode_search": "按关键词检索",
        "mode_find": "以书找书",
        "heat_query_label": "输入主题或关键词",
        "heat_query_placeholder": "例如：memoir personal growth / social media society",
        "heat_no_results": "没有匹配结果，试试更通用的英文关键词。",
        "top1_prefix": "**Top1 推荐**：",
        "similarity_prefix": "相似度",
        "top1_catalog_note": "来自 Open Library 目录，无真实借阅数据",
        "col_similarity": "相似度",
        "col_source": "来源",
        "col_title": "书名",
        "col_author": "作者",
        "col_year": "年份",
        "col_heat": "热度说明",
        "source_legend": (
            "来源 = SPL 表示该书来自西雅图公共图书馆真实借阅排行；"
            "来源 = 目录 表示该书来自 Open Library 批量抓取目录，无真实借阅数据。"
        ),
        "seed_label": "输入书名关键词作为种子",
        "seed_placeholder": "例如：Educated / Becoming / Crawdads",
        "no_seed_match": "没有找到匹配的种子书，试试更短的关键词。",
        "similar_seed_header": "与《{seed}》语义相似的书 Top{n}",
        "methods_expander": "方法与局限",
        "methods_content": (
            "- **语义相似**：TF-IDF + TruncatedSVD（LSA），只使用书名/作者/主题文本，与热度无关\n"
            "- **真实热度**：SPL 公开借阅数据，已按归一化书名合并不同格式，统计 2018–2025 年\n"
            "- **局限**：LSA 在短标题上可能出现表面模式误配；SPL 数据只覆盖西雅图，不代表全球；"
            "跨源主题分布不同可能导致结果偏向某一方"
        ),
        "footer": (
            "ReadScope 演示原型 | 数据来源：人工测试样本 / Open Library 公开 API / Seattle Public Library | "
            "智能检索为内容匹配；“真实热度推荐”页使用 SPL 公开借阅数据（2018–2025 Top30）。"
        ),
        # 数据集（label 用作单选按钮显示文本；badge/nature 用于侧边栏说明）
        "ds_TEST_SAMPLE — 人工测试样本": {
            "label": "TEST_SAMPLE — 人工测试样本",
            "badge": "人工测试样本（非真实馆藏）",
            "nature": "人工构造的测试样本，仅用于验证数据导入与质量检查流程，不代表真实图书馆藏书",
        },
        "ds_OPEN_LIBRARY_API — 公开API数据": {
            "label": "OPEN_LIBRARY_API — 公开API数据",
            "badge": "公开 API 数据（Open Library）",
            "nature": "从 Open Library API 实际获取的公开图书元数据，样本量小，仅覆盖部分英文图书",
        },
        "ds_OPEN_LIBRARY_BATCH — 批量API目录（360本）": {
            "label": "OPEN_LIBRARY_BATCH — 批量API目录（360本）",
            "badge": "公开 API 数据（Open Library 批量）",
            "nature": "按 8 个学科批量抓取的 360 本真实英文图书（已过滤虚构类），是统计分析与智能检索的主数据集",
        },
    },
    "en": {
        "app_caption": "Book Catalog Analytics Dashboard (demo)",
        "select_dataset": "Select dataset",
        "usage_expander": "Data usage statement",
        "usage_rules": (
            "- TEST_SAMPLE and OPEN_LIBRARY_API data are shown separately and are never merged "
            "or presented as a real library collection\n"
            "- Book metadata **cannot** infer circulation counts, popular-loan lists, reader "
            "preferences, or overdue rates\n"
            "- Missing fields are kept as empty values and are never fabricated\n"
            "- This dashboard is read-only; no data files are modified"
        ),
        "main_title": "Book Catalog Analytics Dashboard",
        "load_error": "Failed to load data: ",
        "run_import_hint": "Run data/import/import_book_metadata.py first to generate the cleaned data files.",
        "current_dataset": "Current dataset: ",
        # Overview
        "total_records": "Total records",
        "valid_records": "Valid records (PASS)",
        "invalid_records": "Flagged records",
        "valid_rate": "Valid rate",
        "source_dist": "Data source distribution (source field)",
        "source_missing_caption": "The source field does not exist or is entirely missing, so the source distribution cannot be shown.",
        "dup_isbn_header": "Duplicate ISBN records",
        "dup_isbn_metric": "Records flagged DUPLICATE_ISBN",
        "missing_header": "Missing key fields",
        "col_field": "Field",
        "col_missing": "Missing count",
        "flags_header": "Quality flag distribution",
        "col_flag": "Quality flag",
        "col_desc": "Description",
        "col_count": "Count",
        "no_flags": "No quality flags in the current dataset — all records are PASS.",
        # Catalog analysis
        "catalog_caption": "Statistics follow exactly the same rules as analytics/book_catalog_analyzer.py; charts are based only on the filtered records below.",
        "filter_source": "Filter by data source",
        "source_missing_filter": "The source field does not exist or is entirely missing — no source filter.",
        "filter_year": "Filter by publication year ({n} records with missing/invalid years are excluded)",
        "no_year_filter": "No valid publication years — no year filter.",
        "filter_author": "Filter by author (up to 20 selectable)",
        "no_author_filter": "No author data — no author filter.",
        "filter_subject": "Filter by subject",
        "no_subject_filter": "No subject data — no subject filter.",
        "no_filtered_records": "No records match the current filters. Try adjusting them.",
        "filtered_count": "Filtered records: {a} / {b}",
        "year_dist_header": "Publication year distribution ({n} records with valid years)",
        "chart_year_title": "Publication year distribution",
        "chart_year_xlabel": "Publication year",
        "chart_ylabel": "Number of books",
        "year_range_caption": "Year range {min_y}–{max_y}, mean {mean:.1f}; {miss} missing, {inv} invalid (excluded from the chart).",
        "no_year_data": "No valid year data after filtering — the year chart cannot be drawn.",
        "top_authors_header": "Top 10 authors by work count ({n} unique authors)",
        "top_authors_title": "Top 10 authors by work count",
        "multi_author_note": "Multi-author books are split on semicolons and counted per author, so the counts may exceed the number of records.",
        "no_author_data": "No author data after filtering.",
        "top_publishers_header": "Top 10 publishers ({n} unique publishers)",
        "top_publishers_title": "Top 10 publishers",
        "no_publisher_data": "No publisher data after filtering.",
        "subjects_header": "Subject distribution ({n} unique subjects)",
        "top_subjects_title": "Top 10 subjects",
        "one_subject_only": "Only one subject remains after filtering: \"{name}\" ({cnt} records) — not enough variation for a distribution chart.",
        "no_subject_data": "No subject data after filtering.",
        # Data quality
        "quality_caption": "Only flagged problem records are shown (data_quality_flags != PASS).",
        "no_flagged": "No problem records in the current dataset — all records are PASS.",
        "filter_flags": "Filter by issue type (leave empty to show all flagged records)",
        "flag_option": "{flag} ({desc})",
        "flagged_count": "Flagged records",
        "download_csv": "Download filtered results as CSV",
        # Intelligent search & recommendation
        "ai_method_caption": (
            "Method: TF-IDF text vectors + cosine similarity (content matching on title/author/subject). "
            "You can switch to LSA (latent semantic analysis), which adds SVD dimensionality reduction on top "
            "of TF-IDF so that lexically different but semantically related terms move closer together. "
            "This is **not** an LLM and uses no circulation-behavior data."
        ),
        "method_label": "Search method",
        "opt_tfidf": "TF-IDF (lexical matching, baseline)",
        "opt_lsa": "LSA (latent semantic, TruncatedSVD)",
        "note_lsa": "LSA: the term–document TF-IDF matrix is reduced with SVD into a low-dimensional \"concept space\", then cosine similarity is computed there.",
        "note_tfidf": "TF-IDF: weighted overlap of literal terms + cosine similarity (lexical baseline).",
        "batch_missing_error": "Batch catalog file missing — intelligent search is unavailable: ",
        "run_fetch_hint": "Run data/import/fetch_openlibrary_batch.py first to generate books_openlibrary_batch.csv.",
        "engine_info": "Engine: {engine} ｜ Corpus: OPEN_LIBRARY_BATCH catalog, {n} real books (independent of the dataset selected above)",
        "eval_expander": "TF-IDF vs LSA evaluation (14 queries, metadata-proxy relevance)",
        "eval_design": (
            "Evaluation design: 7 subjects, each with one literal query and one paraphrase query. "
            "Relevance is defined as an exact match on the book's `subject` field "
            "(**metadata-proxy relevance, not human judgment**)."
        ),
        "by_type_caption": "Broken down by query type:",
        "eval_conclusion": (
            "Conclusion: LSA beats TF-IDF on literal queries, but both methods perform poorly on paraphrase "
            "queries — the corpus is small and the text fields are short, leaving latent semantics little to "
            "learn from. Full report: analytics/output/semantic_search_eval_report.md"
        ),
        "eval_missing": "Evaluation results have not been generated yet. Run `python analytics/semantic_search_eval.py`.",
        "query_label": "Enter a topic or keywords of interest, e.g. deep learning / social media / climate psychology",
        "query_placeholder": "e.g. deep learning neural networks",
        "search_btn": "Search",
        "empty_query": "Enter a search query first.",
        "no_results": "No matching books. Try more general English keywords.",
        "results_header": "Search results Top{n}",
        "similarity_note": "Similarity = cosine similarity between the query and the book's text, ranging 0–1; higher means more relevant.",
        "similar_header": "Similar-book recommendations",
        "similar_caption": "Find one book by a title keyword; the system then uses the **current engine ({engine})** to recommend the most content-similar books.",
        "pick_label": "Find a book by title keyword",
        "pick_placeholder": "e.g. Book of Why",
        "no_title_match": "No matching title found. Try a shorter keyword.",
        "pick_radio": "Found {n} matching books — pick one:",
        "current_book": "**Selected book**: {title} — {author} ({year}, {subject})",
        "no_similar": "No other books with similarity > 0 were found for this book in the current corpus.",
        "rec_basis": "Basis: content similarity only — unrelated to circulation behavior.",
        # Real-heat recommendations
        "heat_caption": (
            "Combines **LSA semantic similarity** with **real checkout popularity from Seattle Public Library (SPL)**: "
            "first find semantically similar books, then see how many times they were actually borrowed at SPL "
            "and for how many consecutive years they made the Top 30. "
            "Data source: Seattle Public Library (Public Domain), 2018–2025 Top 30 entries."
        ),
        "heat_data_missing": "Required data file missing: ",
        "heat_files_hint": "Make sure books_openlibrary_batch.csv and spl_top_titles_by_year.csv exist under data/processed/.",
        "heat_corpus": "Merged corpus: {n} records (Open Library catalog {a} + real SPL popular titles {b})",
        "mode_label": "Recommendation mode",
        "mode_search": "Search by keyword",
        "mode_find": "Start from a book",
        "heat_query_label": "Enter a topic or keywords",
        "heat_query_placeholder": "e.g. memoir personal growth / social media society",
        "heat_no_results": "No matching results. Try more general English keywords.",
        "top1_prefix": "**Top pick**: ",
        "similarity_prefix": "Similarity",
        "top1_catalog_note": "From the Open Library catalog — no real circulation data",
        "col_similarity": "Similarity",
        "col_source": "Source",
        "col_title": "Title",
        "col_author": "Author",
        "col_year": "Year",
        "col_heat": "Popularity note",
        "source_legend": (
            "Source = SPL means the book comes from the Seattle Public Library's real circulation rankings; "
            "Source = Catalog means it comes from the batch-scraped Open Library catalog and has no real circulation data."
        ),
        "seed_label": "Enter a title keyword as the seed",
        "seed_placeholder": "e.g. Educated / Becoming / Crawdads",
        "no_seed_match": "No matching seed book found. Try a shorter keyword.",
        "similar_seed_header": "Top {n} books semantically similar to \"{seed}\"",
        "methods_expander": "Method & limitations",
        "methods_content": (
            "- **Semantic similarity**: TF-IDF + TruncatedSVD (LSA) over title/author/subject text only — "
            "popularity plays no role in the vectors\n"
            "- **Real popularity**: SPL public checkout data, aggregated by normalized title across formats, "
            "covering 2018–2025; no reader identifiers are involved\n"
            "- **Limitations**: LSA can mismatch on surface patterns in short titles; SPL only covers Seattle "
            "and is not representative globally; differing subject distributions across the two sources can "
            "bias results toward one side"
        ),
        "footer": (
            "ReadScope demo prototype | Data sources: manual test sample / Open Library public API / "
            "Seattle Public Library | Intelligent search is content-based matching; the Real-Heat tab uses "
            "SPL public checkout data (2018–2025 Top 30)."
        ),
        # Datasets (label feeds the radio display; badge/nature feed the sidebar note)
        "ds_TEST_SAMPLE — 人工测试样本": {
            "label": "TEST_SAMPLE — Manual test sample",
            "badge": "Manual test sample (not a real library collection)",
            "nature": "Synthetically constructed test sample, used only to validate the data import and "
                      "quality-check pipeline. It does not represent a real library collection.",
        },
        "ds_OPEN_LIBRARY_API — 公开API数据": {
            "label": "OPEN_LIBRARY_API — Public API data",
            "badge": "Public API data (Open Library)",
            "nature": "Public book metadata retrieved live from the Open Library API. The sample is small "
                      "and covers only a subset of English-language books.",
        },
        "ds_OPEN_LIBRARY_BATCH — 批量API目录（360本）": {
            "label": "OPEN_LIBRARY_BATCH — Batch API catalog (360 books)",
            "badge": "Public API data (Open Library, batch)",
            "nature": "360 real English-language books fetched in batches across 8 subjects (fiction filtered "
                      "out). This is the primary dataset for statistical analysis and intelligent search.",
        },
    },
}

st.set_page_config(page_title="ReadScope Book Catalog Analytics", page_icon=":books:", layout="wide")

# Permanent bilingual toggle so users can always find it; everything else switches via T.
lang = st.sidebar.radio("Language / 语言", ["English", "中文"], index=0, horizontal=True)
_lang = "en" if lang == "English" else "zh"
T = I18N[_lang]


def flag_desc(flag: str) -> str:
    """Quality-flag description: zh reuses QUALITY_FLAGS from the import module; en uses the local mapping."""
    if _lang == "zh":
        return QUALITY_FLAGS.get(flag, flag)
    return QUALITY_FLAGS_EN.get(flag, flag)


def heat_note_display(row) -> str:
    """Localized popularity note: zh keeps the module-generated heat_note; en rephrases it from raw fields."""
    if _lang == "zh":
        return row["heat_note"]
    if row["source"] == "spl":
        note = f"SPL checkouts: {int(row.get('checkouts', 0)):,}"
        years = int(row.get("years_on_chart", 0))
        if years > 0:
            note += f" · {years} consecutive year(s) in SPL Top 30"
        return note
    return "Catalog record (no real circulation data)"


def display_results(df: pd.DataFrame) -> pd.DataFrame:
    """format_results() emits Chinese column names; restore English column names for display in en mode."""
    if _lang == "zh" or df.empty:
        return df
    col_en = {"书名": "Title", "作者": "Author", "出版年份": "Year", "出版社": "Publisher",
              "主题": "Subject", "相似度": "Score"}
    return df.rename(columns={k: v for k, v in col_en.items() if k in df.columns})


# ---------------------------------------------------------------- Data loading
@st.cache_data(show_spinner=False)
def load_records(path_str: str):
    """Read the cleaned CSV, return (DataFrame, error message). All fields are treated as strings."""
    path = Path(path_str)
    if not path.exists():
        return None, f"Input file does not exist: {path}"
    if path.stat().st_size == 0:
        return None, f"Input file is empty: {path}"
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        return None, f"Failed to read: {e}"
    if df.empty:
        return None, f"File has no data rows: {path}"
    return df, None


# ---------------------------------------------------------------- Chart helpers
COLORS = {"year": "#4C78A8", "author": "#9467BD", "publisher": "#E45756", "subject": "#54A24B"}


def _label_bars_h(ax, bars, pad=0.05):
    for bar in bars:
        w = bar.get_width()
        ax.text(w + pad, bar.get_y() + bar.get_height() / 2, f"{int(w)}",
                va="center", ha="left", fontsize=9)


def _label_bars_v(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{int(h)}",
                ha="center", va="bottom", fontsize=9)


def fig_year(counts: dict, t: dict):
    keys = sorted(counts)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    bars = ax.bar([str(k) for k in keys], [counts[k] for k in keys],
                  color=COLORS["year"], edgecolor="black", alpha=0.75)
    _label_bars_v(ax, bars)
    # 跨度大时（如 1907–2025 共 90+ 年）逐年在横轴上会全部重叠：
    # 只每隔 step 年显示一个刻度，并斜 45° 排布。
    step = max(1, len(keys) // 15)
    tick_pos = range(0, len(keys), step)
    ax.set_xticks(list(tick_pos))
    ax.set_xticklabels([str(keys[i]) for i in tick_pos], rotation=45, ha="right", fontsize=8)
    ax.set_title(t["chart_year_title"], fontsize=13)
    ax.set_xlabel(t["chart_year_xlabel"])
    ax.set_ylabel(t["chart_ylabel"])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_h(counts_top: list, title: str, xlabel: str, key: str):
    """counts_top: [(name, count)]; horizontal bar chart"""
    names = [n for n, _ in counts_top][::-1]
    vals = [c for _, c in counts_top][::-1]
    fig, ax = plt.subplots(figsize=(8, max(2.2, 0.5 * len(names) + 1)))
    bars = ax.barh(names, vals, color=COLORS[key], edgecolor="black", alpha=0.75)
    _label_bars_h(ax, bars)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def records_to_df(records: list) -> pd.DataFrame:
    return pd.DataFrame(records)


# ---------------------------------------------------------------- Sidebar
st.sidebar.title("ReadScope")
st.sidebar.caption(T["app_caption"])

dataset_name = st.sidebar.radio(
    T["select_dataset"],
    options=list(DATASETS.keys()),
    format_func=lambda k: T[f"ds_{k}"]["label"],
)
ds = DATASETS[dataset_name]
ds_i18n = T[f"ds_{dataset_name}"]

df, err = load_records(str(ds["path"]))
st.sidebar.info(f"**{ds_i18n['badge']}**\n\n{ds_i18n['nature']}")

with st.sidebar.expander(T["usage_expander"]):
    st.markdown(T["usage_rules"])

st.title(T["main_title"])
if err:
    st.error(f"{T['load_error']}{err}")
    st.info(T["run_import_hint"])
    st.stop()

st.info(f"{T['current_dataset']}**{dataset_name}** — {ds_i18n['badge']}", icon=":material/info:")

records = df.to_dict("records")

tab_overview, tab_catalog, tab_quality, tab_ai, tab_heat = st.tabs(
    ["数据概览 Overview", "目录分析 Catalog", "数据质量 Data Quality", "检索与推荐 Search", "真实热度推荐 Real-Heat"]
)

# ================================================================ A. Overview
with tab_overview:
    quality = generate_quality_overview(records)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(T["total_records"], quality["total"])
    c2.metric(T["valid_records"], quality["valid"])
    c3.metric(T["invalid_records"], quality["invalid"])
    c4.metric(T["valid_rate"], f"{quality['valid_rate']:.1%}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader(T["source_dist"])
        if "source" in df.columns and df["source"].str.strip().ne("").any():
            src_counts = df["source"].str.strip().value_counts()
            st.bar_chart(src_counts)
        else:
            st.caption(T["source_missing_caption"])

        st.subheader(T["dup_isbn_header"])
        dup_count = quality["flag_counts"].get("DUPLICATE_ISBN", 0)
        st.metric(T["dup_isbn_metric"], dup_count)

    with right:
        st.subheader(T["missing_header"])
        year_stats = analyze_publication_years(records)
        author_stats = analyze_authors(records)
        publisher_stats = analyze_publishers(records)
        subject_stats = analyze_subjects(records)
        missing_tbl = pd.DataFrame(
            {
                T["col_field"]: ["title", "author", "publication_year", "publisher", "subject"],
                T["col_missing"]: [
                    sum(1 for r in records if not r.get("title", "").strip()),
                    author_stats["missing"],
                    year_stats["missing"],
                    publisher_stats["missing"],
                    subject_stats["missing"],
                ],
            }
        )
        st.table(missing_tbl)

        st.subheader(T["flags_header"])
        if quality["flag_counts"]:
            flags_df = pd.DataFrame(
                [
                    {T["col_flag"]: f, T["col_desc"]: flag_desc(f), T["col_count"]: n}
                    for f, n in quality["flag_counts"].most_common()
                ]
            )
            st.table(flags_df)
        else:
            st.caption(T["no_flags"])

# ================================================================ B. Catalog analysis
with tab_catalog:
    st.caption(T["catalog_caption"])

    # ---- Filters (only shown when the field exists and has data)
    f1, f2 = st.columns(2)

    with f1:
        # Source filter
        if "source" in df.columns and df["source"].str.strip().ne("").any():
            src_options = sorted(df["source"].str.strip().unique())
            sel_sources = st.multiselect(T["filter_source"], src_options, default=src_options)
        else:
            sel_sources = None
            st.caption(T["source_missing_filter"])

        # Year filter
        valid_years = year_stats["years"]
        if valid_years:
            y_min, y_max = min(valid_years), max(valid_years)
            y_range = st.slider(
                T["filter_year"].format(n=year_stats["missing"] + year_stats["invalid"]),
                y_min, y_max, (y_min, y_max),
            )
        else:
            y_range = None
            st.caption(T["no_year_filter"])

    with f2:
        # Author filter
        author_options = [a for a, _ in author_stats["author_counts"].most_common(20)]
        if author_options:
            sel_authors = st.multiselect(T["filter_author"], author_options)
        else:
            sel_authors = None
            st.caption(T["no_author_filter"])

        # Subject filter
        subject_options = sorted(subject_stats["subject_counts"].keys())
        if subject_options:
            sel_subjects = st.multiselect(T["filter_subject"], subject_options)
        else:
            sel_subjects = None
            st.caption(T["no_subject_filter"])

    # ---- Apply filters
    def _match(r) -> bool:
        if sel_sources is not None and r.get("source", "").strip() not in sel_sources:
            return False
        if y_range is not None:
            y = r.get("publication_year", "").strip()
            if not (y.isdigit() and len(y) == 4 and y_range[0] <= int(y) <= y_range[1]):
                return False
        if sel_authors:
            authors = [a.strip() for a in r.get("author", "").split(";") if a.strip()]
            if not any(a in sel_authors for a in authors):
                return False
        if sel_subjects and r.get("subject", "").strip() not in sel_subjects:
            return False
        return True

    filtered = [r for r in records if _match(r)]

    if not filtered:
        st.warning(T["no_filtered_records"])
    else:
        st.caption(T["filtered_count"].format(a=len(filtered), b=len(records)))

        fy = analyze_publication_years(filtered)
        fa = analyze_authors(filtered)
        fp = analyze_publishers(filtered)
        fs = analyze_subjects(filtered)

        st.subheader(T["year_dist_header"].format(n=len(fy["years"])))
        if fy["year_counts"]:
            st.pyplot(fig_year(fy["year_counts"], T))
            st.caption(
                T["year_range_caption"].format(
                    min_y=fy["min_year"], max_y=fy["max_year"], mean=fy["mean_year"],
                    miss=fy["missing"], inv=fy["invalid"],
                )
            )
        else:
            st.info(T["no_year_data"])

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader(T["top_authors_header"].format(n=fa["unique_authors"]))
            if fa["top_authors"]:
                st.pyplot(fig_h(fa["top_authors"][:10], T["top_authors_title"], T["chart_ylabel"], "author"))
                st.caption(T["multi_author_note"])
            else:
                st.info(T["no_author_data"])

        with col_b:
            st.subheader(T["top_publishers_header"].format(n=fp["unique_publishers"]))
            if fp["top_publishers"]:
                st.pyplot(fig_h(fp["top_publishers"][:10], T["top_publishers_title"], T["chart_ylabel"], "publisher"))
            else:
                st.info(T["no_publisher_data"])

        st.subheader(T["subjects_header"].format(n=fs["unique_subjects"]))
        if len(fs["subject_counts"]) >= 2:
            st.pyplot(fig_h(fs["top_subjects"][:10], T["top_subjects_title"], T["chart_ylabel"], "subject"))
        elif len(fs["subject_counts"]) == 1:
            only, cnt = fs["top_subjects"][0]
            st.info(T["one_subject_only"].format(name=only, cnt=cnt))
        else:
            st.info(T["no_subject_data"])

# ================================================================ C. Data quality check
with tab_quality:
    st.caption(T["quality_caption"])

    flagged = [r for r in records if r.get("data_quality_flags", "").strip() not in ("", "PASS")]

    if not flagged:
        st.success(T["no_flagged"])
    else:
        all_flags = sorted({f for r in flagged for f in r["data_quality_flags"].split(";") if f})

        sel_flags = st.multiselect(
            T["filter_flags"],
            all_flags,
            format_func=lambda f: T["flag_option"].format(flag=f, desc=flag_desc(f)),
        )

        shown = [
            r for r in flagged
            if not sel_flags or any(f in sel_flags for f in r["data_quality_flags"].split(";"))
        ]

        st.metric(T["flagged_count"], f"{len(shown)} / {len(records)}")

        cols = ["isbn", "title", "author", "publication_year", "publisher",
                "subject", "source", "source_record_id", "data_quality_flags"]
        cols = [c for c in cols if c in df.columns]
        out_df = pd.DataFrame(shown)[cols] if shown else pd.DataFrame(columns=cols)
        st.dataframe(out_df, use_container_width=True, hide_index=True)

        st.download_button(
            T["download_csv"],
            data=out_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="quality_flagged_records.csv",
            mime="text/csv",
            disabled=out_df.empty,
        )

# ================================================================ D. Intelligent search & recommendation
@st.cache_resource(show_spinner=False)
def get_tfidf_engine():
    """TF-IDF lexical matching engine; corpus is the 360-book batch catalog."""
    return BookSearchRecommender(load_catalog())


@st.cache_resource(show_spinner=False)
def get_lsa_engine():
    """LSA latent-semantic engine (TF-IDF + TruncatedSVD); corpus is the 360-book batch catalog."""
    return LsaSearchEngine(load_catalog())


with tab_ai:
    st.caption(T["ai_method_caption"])

    try:
        engine_choice = st.radio(
            T["method_label"],
            [T["opt_tfidf"], T["opt_lsa"]],
            horizontal=True,
        )
        if engine_choice.startswith("LSA"):
            engine = get_lsa_engine()
            method_note = T["note_lsa"]
        else:
            engine = get_tfidf_engine()
            method_note = T["note_tfidf"]
    except FileNotFoundError as e:
        st.error(f"{T['batch_missing_error']}{e}")
        st.info(T["run_fetch_hint"])
        st.stop()

    st.success(T["engine_info"].format(engine=engine.name, n=len(engine.df)))
    st.caption(method_note)

    with st.expander(T["eval_expander"]):
        eval_csv = Path(__file__).parent / "output" / "semantic_search_eval_results.csv"
        st.markdown(T["eval_design"])
        if eval_csv.exists():
            ev = pd.read_csv(eval_csv)
            agg = (
                ev.groupby("method")[["precision@5", "recall@10", "mrr", "ndcg@10"]]
                .mean()
                .round(4)
                .reset_index()
            )
            st.dataframe(agg, use_container_width=True, hide_index=True)
            by_type = (
                ev.groupby(["method", "type"])[["precision@5", "recall@10", "mrr", "ndcg@10"]]
                .mean()
                .round(4)
                .reset_index()
            )
            st.caption(T["by_type_caption"])
            st.dataframe(by_type, use_container_width=True, hide_index=True)
            st.caption(T["eval_conclusion"])
        else:
            st.caption(T["eval_missing"])

    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input(
            T["query_label"],
            placeholder=T["query_placeholder"],
        )
    with c2:
        st.write("")
        st.write("")
        do_search = st.button(T["search_btn"], type="primary", use_container_width=True)

    if do_search and not query.strip():
        st.warning(T["empty_query"])
    elif query.strip():
        # Triggered by pressing Enter in the input or clicking the search button
        results = engine.search(query, top_n=10)
        if results.empty:
            st.warning(T["no_results"])
        else:
            st.subheader(T["results_header"].format(n=len(results)))
            st.dataframe(display_results(format_results(results)), use_container_width=True, hide_index=True)
            st.caption(T["similarity_note"])

    st.divider()

    st.subheader(T["similar_header"])
    st.caption(T["similar_caption"].format(engine=engine.name))

    pick_query = st.text_input(
        T["pick_label"],
        placeholder=T["pick_placeholder"],
    )

    if pick_query.strip():
        matches = engine.df[
            engine.df["title"].str.contains(pick_query.strip(), case=False, regex=False)
        ]
        if matches.empty:
            st.warning(T["no_title_match"])
        else:
            matches = matches.head(8).reset_index()
            options = {
                f"{r['title']} — {r['author']} ({r['publication_year']})": int(r["index"])
                for _, r in matches.iterrows()
            }
            if len(matches) > 1:
                chosen = st.radio(
                    T["pick_radio"].format(n=len(matches)),
                    options=list(options.keys()),
                )
                idx = options[chosen]
            else:
                idx = next(iter(options.values()))

            picked = engine.df.iloc[idx]
            st.markdown(
                T["current_book"].format(
                    title=picked["title"], author=picked["author"],
                    year=picked["publication_year"], subject=picked["subject"],
                )
            )
            sim = engine.similar_books(idx, top_n=8)
            if sim.empty:
                st.info(T["no_similar"])
            else:
                col_t, col_c = st.columns([3, 2])
                with col_t:
                    st.dataframe(display_results(format_results(sim)), use_container_width=True, hide_index=True)
                with col_c:
                    chart_df = sim[["title", "score"]].set_index("title")
                    st.bar_chart(chart_df, height=320)
                    st.caption(T["rec_basis"])

# ================================================================ E. Real-heat recommendations
@st.cache_resource(show_spinner=False)
def get_heat_recommender():
    return RealPopularityRecommender()


with tab_heat:
    st.caption(T["heat_caption"])

    try:
        heat_rec = get_heat_recommender()
    except FileNotFoundError as e:
        st.error(f"{T['heat_data_missing']}{e}")
        st.info(T["heat_files_hint"])
        st.stop()

    st.success(T["heat_corpus"].format(
        n=len(heat_rec.df), a=len(heat_rec.catalog), b=len(heat_rec.spl),
    ))

    mode = st.radio(T["mode_label"], [T["mode_search"], T["mode_find"]], horizontal=True)

    if mode == T["mode_search"]:
        q = st.text_input(
            T["heat_query_label"],
            placeholder=T["heat_query_placeholder"],
        )
        if q.strip():
            results = heat_rec.search(q.strip(), top_n=8)
            if results.empty:
                st.warning(T["heat_no_results"])
            else:
                top = results.iloc[0]
                if top["source"] == "spl":
                    st.success(
                        f"{T['top1_prefix']}{top['title']} — {top['author']}  \n"
                        f"{T['similarity_prefix']} {top['score']:.3f} ｜ {heat_note_display(top)}",
                        icon=":material/star:",
                    )
                else:
                    st.info(
                        f"{T['top1_prefix']}{top['title']} — {top['author']}  \n"
                        f"{T['similarity_prefix']} {top['score']:.3f} ｜ {T['top1_catalog_note']}",
                        icon=":material/info:",
                    )
                st.subheader(T["results_header"].format(n=len(results)))
                show = results[["score", "source", "title", "author", "publication_year", "heat_note"]].copy()
                show["heat_note"] = show.apply(heat_note_display, axis=1)
                show.columns = [T["col_similarity"], T["col_source"], T["col_title"],
                                T["col_author"], T["col_year"], T["col_heat"]]
                st.dataframe(show, use_container_width=True, hide_index=True)
                st.caption(T["source_legend"])
    else:
        seed = st.text_input(
            T["seed_label"],
            placeholder=T["seed_placeholder"],
        )
        if seed.strip():
            results = heat_rec.similar_books(seed.strip(), top_n=8)
            if results.empty:
                st.warning(T["no_seed_match"])
            else:
                top = results.iloc[0]
                if top["source"] == "spl":
                    st.success(
                        f"{T['top1_prefix']}{top['title']} — {top['author']}  \n"
                        f"{T['similarity_prefix']} {top['score']:.3f} ｜ {heat_note_display(top)}",
                        icon=":material/star:",
                    )
                else:
                    st.info(
                        f"{T['top1_prefix']}{top['title']} — {top['author']}  \n"
                        f"{T['similarity_prefix']} {top['score']:.3f} ｜ {T['top1_catalog_note']}",
                        icon=":material/info:",
                    )
                st.subheader(T["similar_seed_header"].format(seed=seed.strip(), n=len(results)))
                show = results[["score", "source", "title", "author", "publication_year", "heat_note"]].copy()
                show["heat_note"] = show.apply(heat_note_display, axis=1)
                show.columns = [T["col_similarity"], T["col_source"], T["col_title"],
                                T["col_author"], T["col_year"], T["col_heat"]]
                st.dataframe(show, use_container_width=True, hide_index=True)

    with st.expander(T["methods_expander"]):
        st.markdown(T["methods_content"])

st.divider()
st.caption(T["footer"])
