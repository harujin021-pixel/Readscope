#!/usr/bin/env python3
"""
图书目录分析模块
功能：
1. 读取清洗后的图书元数据
2. 统计出版年份分布、作者作品数量、出版社分布、分类分布
3. 生成数据质量概览
4. 输出图表和分析报告

约束：
- 明确区分 TEST_SAMPLE 与 OPEN_LIBRARY 数据
- 不合并数据宣称代表真实图书馆藏书
- 不从元数据推断借阅行为
- 缺失字段明确处理
"""

import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt

# 中文字体配置：matplotlib 默认字体 DejaVu Sans 不含中文字形，会导致标题/坐标轴乱码。
# 依次尝试 Windows 常见中文字体，并把负号交给中文字体渲染（否则负号会显示为方块）。
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 项目根目录
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYTICS_DIR = BASE_DIR / "analytics"
OUTPUT_DIR = ANALYTICS_DIR / "output"
FIGURES_DIR = ANALYTICS_DIR / "figures"

# 输入文件
INPUT_FILES = {
    "test_sample": PROCESSED_DIR / "books_cleaned.csv",
    "api_data": PROCESSED_DIR / "books_from_api.csv",
    "openlibrary_batch": PROCESSED_DIR / "books_openlibrary_batch.csv",
}

# 输出文件
REPORT_PATH = OUTPUT_DIR / "catalog_analysis_report.md"
STATS_PATH = OUTPUT_DIR / "catalog_statistics.csv"


def ensure_dirs():
    """确保输出目录存在"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(filepath):
    """读取 CSV 文件，返回记录列表和错误信息"""
    records = []
    errors = []

    if not filepath.exists():
        errors.append(f"文件不存在: {filepath}")
        return records, errors

    if filepath.stat().st_size == 0:
        errors.append(f"文件为空: {filepath}")
        return records, errors

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                errors.append(f"无法解析 CSV 表头: {filepath}")
                return records, errors

            for row in reader:
                records.append(row)

    except Exception as e:
        errors.append(f"读取失败: {e}")

    return records, errors


def filter_valid_records(records):
    """过滤有效记录（PASS）"""
    return [r for r in records if r.get("data_quality_flags", "").strip() == "PASS"]


def filter_by_source(records, source):
    """按来源过滤"""
    return [r for r in records if r.get("source", "").strip().upper() == source.upper()]


def analyze_publication_years(records):
    """分析出版年份分布"""
    years = []
    missing = 0
    invalid = 0

    for r in records:
        year = r.get("publication_year", "").strip()
        if not year:
            missing += 1
        elif year.isdigit() and len(year) == 4:
            years.append(int(year))
        else:
            invalid += 1

    return {
        "years": years,
        "year_counts": Counter(years),
        "missing": missing,
        "invalid": invalid,
        "min_year": min(years) if years else None,
        "max_year": max(years) if years else None,
        "mean_year": sum(years) / len(years) if years else None,
    }


def analyze_authors(records):
    """分析作者作品数量"""
    author_counts = Counter()
    missing = 0

    for r in records:
        author = r.get("author", "").strip()
        if not author:
            missing += 1
        else:
            # 处理多作者（分号分隔）
            for a in author.split(";"):
                a = a.strip()
                if a:
                    author_counts[a] += 1

    return {
        "author_counts": author_counts,
        "unique_authors": len(author_counts),
        "missing": missing,
        "top_authors": author_counts.most_common(10),
    }


def analyze_publishers(records):
    """分析出版社分布"""
    publisher_counts = Counter()
    missing = 0

    for r in records:
        publisher = r.get("publisher", "").strip()
        if not publisher:
            missing += 1
        else:
            publisher_counts[publisher] += 1

    return {
        "publisher_counts": publisher_counts,
        "unique_publishers": len(publisher_counts),
        "missing": missing,
        "top_publishers": publisher_counts.most_common(10),
    }


def analyze_subjects(records):
    """分析分类分布"""
    subject_counts = Counter()
    missing = 0

    for r in records:
        subject = r.get("subject", "").strip()
        if not subject:
            missing += 1
        else:
            subject_counts[subject] += 1

    return {
        "subject_counts": subject_counts,
        "unique_subjects": len(subject_counts),
        "missing": missing,
        "top_subjects": subject_counts.most_common(10),
    }


def generate_quality_overview(records):
    """生成数据质量概览"""
    total = len(records)
    valid = len(filter_valid_records(records))
    invalid = total - valid

    flag_counts = Counter()
    for r in records:
        flags = r.get("data_quality_flags", "").strip()
        if flags and flags != "PASS":
            for f in flags.split(";"):
                flag_counts[f.strip()] += 1

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "valid_rate": valid / total if total > 0 else 0,
        "flag_counts": flag_counts,
    }


def plot_year_distribution(year_stats, source_label, output_path):
    """绘制出版年份分布图"""
    if not year_stats["years"]:
        return False, "无有效年份数据，跳过绘图"

    fig, ax = plt.subplots(figsize=(10, 6))
    years = sorted(year_stats["year_counts"].keys())
    counts = [year_stats["year_counts"][y] for y in years]

    ax.bar(years, counts, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Publication Year", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Publication Year Distribution - {source_label}", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # 添加数值标签
    for i, (y, c) in enumerate(zip(years, counts)):
        ax.text(y, c + 0.1, str(c), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return True, f"已保存: {output_path}"


def plot_publisher_distribution(publisher_stats, source_label, output_path):
    """绘制出版社分布图"""
    if not publisher_stats["publisher_counts"]:
        return False, "无出版社数据，跳过绘图"

    # 取前 10 个
    top = publisher_stats["publisher_counts"].most_common(10)
    if len(top) < 2:
        return False, "出版社种类不足（<2），不适合绘图"

    fig, ax = plt.subplots(figsize=(10, 6))
    publishers = [p[0] for p in top]
    counts = [p[1] for p in top]

    bars = ax.barh(publishers, counts, color="coral", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel("Publisher", fontsize=12)
    ax.set_title(f"Publisher Distribution - {source_label}", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), ha="left", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return True, f"已保存: {output_path}"


def plot_subject_distribution(subject_stats, source_label, output_path):
    """绘制分类分布图"""
    if not subject_stats["subject_counts"]:
        return False, "无分类数据，跳过绘图"

    # 取前 10 个
    top = subject_stats["subject_counts"].most_common(10)
    if len(top) < 2:
        return False, f"分类种类不足（{len(top)} 种），不适合绘图"

    fig, ax = plt.subplots(figsize=(10, 6))
    subjects = [s[0][:30] for s in top]  # 截断长标签
    counts = [s[1] for s in top]

    bars = ax.barh(subjects, counts, color="seagreen", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel("Subject", fontsize=12)
    ax.set_title(f"Subject Distribution - {source_label}", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), ha="left", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return True, f"已保存: {output_path}"


def plot_author_distribution(author_stats, source_label, output_path):
    """绘制作者作品数量分布图"""
    if not author_stats["author_counts"]:
        return False, "无作者数据，跳过绘图"

    # 取前 10 个
    top = author_stats["author_counts"].most_common(10)
    if len(top) < 2:
        return False, "作者种类不足（<2），不适合绘图"

    fig, ax = plt.subplots(figsize=(10, 6))
    authors = [a[0][:25] for a in top]  # 截断长名字
    counts = [a[1] for a in top]

    bars = ax.barh(authors, counts, color="mediumpurple", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Book Count", fontsize=12)
    ax.set_ylabel("Author", fontsize=12)
    ax.set_title(f"Top Authors by Book Count - {source_label}", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), ha="left", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return True, f"已保存: {output_path}"


def write_statistics_csv(all_stats, output_path):
    """写入统计结果 CSV"""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "metric", "value", "source"])

        for source_label, stats in all_stats.items():
            q = stats["quality"]
            writer.writerow(["quality", "total_records", q["total"], source_label])
            writer.writerow(["quality", "valid_records", q["valid"], source_label])
            writer.writerow(["quality", "invalid_records", q["invalid"], source_label])
            writer.writerow(["quality", "valid_rate", f"{q['valid_rate']:.2%}", source_label])

            y = stats["years"]
            if y["years"]:
                writer.writerow(["years", "min_year", y["min_year"], source_label])
                writer.writerow(["years", "max_year", y["max_year"], source_label])
                writer.writerow(["years", "mean_year", f"{y['mean_year']:.1f}", source_label])
            writer.writerow(["years", "missing_count", y["missing"], source_label])
            writer.writerow(["years", "invalid_count", y["invalid"], source_label])

            a = stats["authors"]
            writer.writerow(["authors", "unique_authors", a["unique_authors"], source_label])
            writer.writerow(["authors", "missing_count", a["missing"], source_label])

            p = stats["publishers"]
            writer.writerow(["publishers", "unique_publishers", p["unique_publishers"], source_label])
            writer.writerow(["publishers", "missing_count", p["missing"], source_label])

            s = stats["subjects"]
            writer.writerow(["subjects", "unique_subjects", s["unique_subjects"], source_label])
            writer.writerow(["subjects", "missing_count", s["missing"], source_label])


def write_analysis_report(all_stats, plot_results, output_path):
    """写入 Markdown 分析报告"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 图书目录分析报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 数据来源说明\n\n")
        f.write("本报告基于以下两个独立数据源：\n\n")
        f.write("1. **TEST_SAMPLE**：人工构造的测试样本，用于验证数据导入流程。**不代表真实图书馆藏书**。\n")
        f.write("2. **OPEN_LIBRARY_API**：从 Open Library API 实际获取的公开图书元数据。\n\n")
        f.write("**重要声明**：\n")
        f.write("- 本报告不将两个数据源合并后宣称代表真实图书馆藏书。\n")
        f.write("- 图书元数据**不能**用于推断借阅次数、热门图书、读者偏好或逾期率。\n")
        f.write("- 样本量较小，统计结论仅供参考，不具备广泛代表性。\n\n")

        for source_label, stats in all_stats.items():
            f.write(f"## {source_label} 数据集分析\n\n")

            q = stats["quality"]
            f.write(f"### 数据质量概览\n\n")
            f.write(f"- 总记录数: {q['total']}\n")
            f.write(f"- 有效记录: {q['valid']} ({q['valid_rate']:.1%})\n")
            f.write(f"- 有问题记录: {q['invalid']}\n\n")

            if q["flag_counts"]:
                f.write("质量问题分布:\n")
                for flag, count in q["flag_counts"].most_common():
                    f.write(f"- {flag}: {count}\n")
                f.write("\n")

            y = stats["years"]
            f.write(f"### 出版年份分析\n\n")
            if y["years"]:
                f.write(f"- 有效年份记录: {len(y['years'])}\n")
                f.write(f"- 缺失年份: {y['missing']}\n")
                f.write(f"- 非法年份: {y['invalid']}\n")
                f.write(f"- 年份范围: {y['min_year']} - {y['max_year']}\n")
                f.write(f"- 平均年份: {y['mean_year']:.1f}\n\n")
            else:
                f.write(f"- 无有效年份数据（缺失: {y['missing']}, 非法: {y['invalid']}）\n\n")

            a = stats["authors"]
            f.write(f"### 作者分析\n\n")
            f.write(f"- 唯一作者数: {a['unique_authors']}\n")
            f.write(f"- 缺失作者: {a['missing']}\n\n")
            if a["top_authors"]:
                f.write("作品数量前 5 的作者:\n")
                for author, count in a["top_authors"][:5]:
                    f.write(f"- {author}: {count} 本\n")
                f.write("\n")

            p = stats["publishers"]
            f.write(f"### 出版社分析\n\n")
            f.write(f"- 唯一出版社数: {p['unique_publishers']}\n")
            f.write(f"- 缺失出版社: {p['missing']}\n\n")
            if p["top_publishers"]:
                f.write("出版数量前 5 的出版社:\n")
                for pub, count in p["top_publishers"][:5]:
                    f.write(f"- {pub}: {count} 本\n")
                f.write("\n")

            s = stats["subjects"]
            f.write(f"### 分类分析\n\n")
            f.write(f"- 唯一分类数: {s['unique_subjects']}\n")
            f.write(f"- 缺失分类: {s['missing']}\n\n")
            if s["top_subjects"]:
                f.write("主要分类:\n")
                for subj, count in s["top_subjects"][:5]:
                    f.write(f"- {subj}: {count} 本\n")
                f.write("\n")

        f.write("## 图表输出\n\n")
        for source_label, results in plot_results.items():
            f.write(f"### {source_label}\n\n")
            for plot_type, (success, message) in results.items():
                status = "✅" if success else "⚠️"
                f.write(f"- {status} {plot_type}: {message}\n")
            f.write("\n")

        f.write("## 样本限制与不可推断结论\n\n")
        f.write("### 样本限制\n\n")
        f.write("1. **样本量小**：当前数据集仅包含少量图书，统计结果不具备广泛代表性。\n")
        f.write("2. **来源单一**：TEST_SAMPLE 为人工构造，OPEN_LIBRARY_API 仅覆盖英文图书。\n")
        f.write("3. **字段缺失**：部分记录缺失作者、年份等关键字段。\n")
        f.write("4. **分类单一**：TEST_SAMPLE 中所有图书分类均为 Computer Science，无多样性。\n\n")

        f.write("### 不可推断的结论\n\n")
        f.write("基于当前数据，**不能**得出以下结论：\n\n")
        f.write("- ❌ 图书馆的借阅趋势或热门图书\n")
        f.write("- ❌ 读者的阅读偏好或行为模式\n")
        f.write("- ❌ 图书的逾期率或流通效率\n")
        f.write("- ❌ 任何关于真实图书馆藏书的统计推断\n\n")

        f.write("### 后续改进方向\n\n")
        f.write("1. 扩大样本量，覆盖更多学科分类\n")
        f.write("2. 引入真实的借阅记录数据（需合法获取）\n")
        f.write("3. 增加图书分类字段到原系统\n")
        f.write("4. 建立数据更新机制，保持目录时效性\n")


def analyze_dataset(filepath, source_label):
    """分析单个数据集"""
    print(f"\n分析数据集: {source_label}")
    print(f"  输入文件: {filepath}")

    records, errors = read_csv(filepath)

    if errors:
        for err in errors:
            print(f"  错误: {err}")
        return None, []

    print(f"  读取记录: {len(records)} 条")

    # 质量概览
    quality = generate_quality_overview(records)
    print(f"  有效记录: {quality['valid']} ({quality['valid_rate']:.1%})")

    # 各项分析
    year_stats = analyze_publication_years(records)
    author_stats = analyze_authors(records)
    publisher_stats = analyze_publishers(records)
    subject_stats = analyze_subjects(records)

    stats = {
        "quality": quality,
        "years": year_stats,
        "authors": author_stats,
        "publishers": publisher_stats,
        "subjects": subject_stats,
    }

    # 生成图表
    plot_results = {}

    plot_results["year_distribution"] = plot_year_distribution(
        year_stats, source_label, FIGURES_DIR / f"{source_label}_year_distribution.png"
    )

    plot_results["publisher_distribution"] = plot_publisher_distribution(
        publisher_stats, source_label, FIGURES_DIR / f"{source_label}_publisher_distribution.png"
    )

    plot_results["subject_distribution"] = plot_subject_distribution(
        subject_stats, source_label, FIGURES_DIR / f"{source_label}_subject_distribution.png"
    )

    plot_results["author_distribution"] = plot_author_distribution(
        author_stats, source_label, FIGURES_DIR / f"{source_label}_author_distribution.png"
    )

    for plot_type, (success, msg) in plot_results.items():
        status = "✅" if success else "⚠️"
        print(f"  {status} {plot_type}: {msg}")

    return stats, plot_results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="图书目录分析模块")
    parser.add_argument("--input", "-i", type=str, help="输入 CSV 文件路径")
    parser.add_argument("--source", "-s", type=str, default="CUSTOM", help="数据来源标签")
    parser.add_argument("--all", action="store_true", help="分析所有预设数据集")

    args = parser.parse_args()

    ensure_dirs()

    all_stats = {}
    all_plot_results = {}

    if args.all:
        # 分析所有预设数据集
        for source_label, filepath in INPUT_FILES.items():
            stats, plot_results = analyze_dataset(filepath, source_label)
            if stats:
                all_stats[source_label] = stats
                all_plot_results[source_label] = plot_results
    elif args.input:
        # 分析指定文件
        filepath = Path(args.input)
        stats, plot_results = analyze_dataset(filepath, args.source)
        if stats:
            all_stats[args.source] = stats
            all_plot_results[args.source] = plot_results
    else:
        print("错误: 请指定 --input 或 --all")
        sys.exit(1)

    if not all_stats:
        print("错误: 没有成功分析任何数据集")
        sys.exit(1)

    # 写入统计 CSV
    print(f"\n写入统计结果: {STATS_PATH}")
    write_statistics_csv(all_stats, STATS_PATH)

    # 写入分析报告
    print(f"写入分析报告: {REPORT_PATH}")
    write_analysis_report(all_stats, all_plot_results, REPORT_PATH)

    print("\n分析完成!")
    print(f"  统计结果: {STATS_PATH}")
    print(f"  分析报告: {REPORT_PATH}")
    print(f"  图表目录: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
