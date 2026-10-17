#!/usr/bin/env python3
"""
图书元数据导入与质量检查模块
功能：
1. 读取 data/external_raw/ 下的原始 CSV
2. 规范化字段并输出到 data/processed/
3. 执行数据质量检查并生成报告
4. 可选：从 Open Library API 获取数据（需网络）

约束：
- 不修改原有 book.csv / borrower.csv / record.csv
- 不生成借阅记录
- 缺失字段保留空值，不自行编造
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录（import_book_metadata.py 位于 data/import/ 下，向上两级到项目根目录）
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "external_raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
IMPORT_DIR = BASE_DIR / "data" / "import"

# 原始数据字段（输入）
RAW_FIELDS = ["isbn", "title", "author", "publisher", "publication_year", "subject", "source", "source_record_id"]

# 清洗后字段（输出）
CLEAN_FIELDS = ["isbn", "title", "author", "publisher", "publication_year", "subject", "source", "source_record_id", "import_date", "data_quality_flags"]

# 质量检查标记
QUALITY_FLAGS = {
    "MISSING_TITLE": "缺失书名",
    "MISSING_AUTHOR": "缺失作者",
    "MISSING_YEAR": "缺失出版年份",
    "INVALID_YEAR": "年份格式非法",
    "MISSING_SUBJECT": "缺失分类",
    "MISSING_SOURCE": "缺失来源",
    "MISSING_SOURCE_ID": "缺失来源记录ID",
    "DUPLICATE_ISBN": "重复ISBN",
}


def ensure_dirs():
    """确保目录存在"""
    for d in [RAW_DIR, PROCESSED_DIR, IMPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def read_raw_csv(filepath):
    """读取原始 CSV 文件"""
    records = []
    errors = []

    if not filepath.exists():
        errors.append(f"输入文件不存在: {filepath}")
        return records, errors

    if filepath.stat().st_size == 0:
        errors.append(f"输入文件为空: {filepath}")
        return records, errors

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # 跳过注释行
            lines = [line for line in f if not line.strip().startswith("#")]

        if not lines:
            errors.append(f"输入文件无有效数据行: {filepath}")
            return records, errors

        reader = csv.DictReader(lines)
        # 验证表头
        if reader.fieldnames is None:
            errors.append(f"无法解析 CSV 表头: {filepath}")
            return records, errors

        missing_fields = [f for f in RAW_FIELDS if f not in reader.fieldnames]
        if missing_fields:
            errors.append(f"缺少必需字段: {missing_fields}")

        for i, row in enumerate(reader, start=1):
            row["_line_number"] = i
            records.append(row)

    except Exception as e:
        errors.append(f"读取文件失败: {e}")

    return records, errors


def normalize_record(raw):
    """规范化单条记录"""
    clean = {}

    # 直接映射字段
    for field in RAW_FIELDS:
        value = raw.get(field, "").strip()
        clean[field] = value if value else ""

    # 年份规范化
    year = clean.get("publication_year", "")
    if year:
        # 尝试提取4位年份
        import re
        match = re.search(r"\b(\d{4})\b", year)
        if match:
            clean["publication_year"] = match.group(1)
        # 如果无法提取，保留原值，质量检查会标记

    # 分类字段：如果是多个分类，取第一个
    subject = clean.get("subject", "")
    if ";" in subject:
        clean["subject"] = subject.split(";")[0].strip()

    # 添加导入日期
    clean["import_date"] = datetime.now().strftime("%Y-%m-%d")

    return clean


def check_quality(records):
    """执行数据质量检查"""
    report = {
        "total_records": len(records),
        "issues": [],
        "issue_counts": {k: 0 for k in QUALITY_FLAGS.keys()},
        "valid_records": 0,
        "invalid_records": 0,
    }

    seen_isbns = {}

    for i, record in enumerate(records):
        flags = []

        # 检查缺失书名
        if not record.get("title"):
            flags.append("MISSING_TITLE")

        # 检查缺失作者
        if not record.get("author"):
            flags.append("MISSING_AUTHOR")

        # 检查出版年份
        year = record.get("publication_year", "")
        if not year:
            flags.append("MISSING_YEAR")
        else:
            # 检查年份格式（应该是4位数字）
            import re
            if not re.match(r"^\d{4}$", year):
                flags.append("INVALID_YEAR")

        # 检查分类
        if not record.get("subject"):
            flags.append("MISSING_SUBJECT")

        # 检查来源
        if not record.get("source"):
            flags.append("MISSING_SOURCE")

        # 检查来源记录ID
        if not record.get("source_record_id"):
            flags.append("MISSING_SOURCE_ID")

        # 检查重复 ISBN
        isbn = record.get("isbn", "")
        if isbn:
            if isbn in seen_isbns:
                flags.append("DUPLICATE_ISBN")
            else:
                seen_isbns[isbn] = i

        # 记录问题
        for flag in flags:
            report["issue_counts"][flag] += 1
            report["issues"].append({
                "line": record.get("_line_number", i + 1),
                "isbn": isbn,
                "title": record.get("title", "N/A"),
                "flag": flag,
                "description": QUALITY_FLAGS[flag],
            })

        # 标记到记录上
        record["data_quality_flags"] = ";".join(flags) if flags else "PASS"

        if flags:
            report["invalid_records"] += 1
        else:
            report["valid_records"] += 1

    return report


def write_processed_csv(records, output_path):
    """写入清洗后的 CSV"""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CLEAN_FIELDS)
        writer.writeheader()
        for record in records:
            # 移除内部字段
            row = {k: v for k, v in record.items() if not k.startswith("_")}
            writer.writerow(row)


def write_report(report, output_path):
    """写入质量检查报告"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 数据导入质量检查报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 总体统计\n\n")
        f.write(f"- 总记录数: {report['total_records']}\n")
        f.write(f"- 有效记录: {report['valid_records']}\n")
        f.write(f"- 有问题记录: {report['invalid_records']}\n\n")

        f.write(f"## 问题分类统计\n\n")
        for flag, count in report["issue_counts"].items():
            if count > 0:
                f.write(f"- {QUALITY_FLAGS[flag]}: {count}\n")

        f.write(f"\n## 详细问题列表\n\n")
        if report["issues"]:
            f.write("| 行号 | ISBN | 书名 | 问题标记 | 问题描述 |\n")
            f.write("|------|------|------|---------|---------|\n")
            for issue in report["issues"]:
                f.write(f"| {issue['line']} | {issue['isbn']} | {issue['title']} | {issue['flag']} | {issue['description']} |\n")
        else:
            f.write("无问题记录\n")


def import_from_openlibrary(isbn_list):
    """
    从 Open Library API 获取图书元数据
    注意：此功能需要网络连接，返回原始 API 数据保存到 external_raw/
    """
    try:
        import urllib.request
    except ImportError:
        return {"error": "urllib 不可用，跳过 API 导入"}

    results = []
    errors = []

    for isbn in isbn_list:
        url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                key = f"ISBN:{isbn}"
                if key in data:
                    book = data[key]
                    record = {
                        "isbn": isbn,
                        "title": book.get("title", ""),
                        "author": "; ".join([a.get("name", "") for a in book.get("authors", [])]),
                        "publisher": "; ".join([p.get("name", "") for p in book.get("publishers", [])]),
                        "publication_year": book.get("publish_date", ""),
                        "subject": "; ".join([s.get("name", "") for s in book.get("subjects", [])[:3]]),
                        "source": "OPEN_LIBRARY",
                        "source_record_id": book.get("key", "").replace("/books/", ""),
                    }
                    results.append(record)
                else:
                    errors.append(f"ISBN {isbn} 未在 Open Library 找到")
        except Exception as e:
            errors.append(f"获取 ISBN {isbn} 失败: {e}")

    return {"records": results, "errors": errors}


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="图书元数据导入与质量检查")
    parser.add_argument("--input", "-i", type=str, help="输入 CSV 文件路径（默认: data/external_raw/test_sample_openlibrary.csv）")
    parser.add_argument("--output", "-o", type=str, help="输出 CSV 文件路径（默认: data/processed/books_cleaned.csv）")
    parser.add_argument("--report", "-r", type=str, help="质量报告路径（默认: data/processed/import_report.md）")
    parser.add_argument("--api", action="store_true", help="从 Open Library API 获取数据（需提供 --isbn-list）")
    parser.add_argument("--isbn-list", type=str, help="ISBN 列表文件路径（每行一个 ISBN）")
    parser.add_argument("--no-check", action="store_true", help="跳过质量检查")

    args = parser.parse_args()

    ensure_dirs()

    # 确定输入文件
    if args.api and args.isbn_list:
        # 从 API 获取
        isbn_file = Path(args.isbn_list)
        if not isbn_file.exists():
            print(f"错误: ISBN 列表文件不存在: {isbn_file}")
            sys.exit(1)

        with open(isbn_file, "r") as f:
            isbn_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        print(f"从 Open Library API 获取 {len(isbn_list)} 本图书...")
        api_result = import_from_openlibrary(isbn_list)

        if api_result.get("error"):
            print(f"API 错误: {api_result['error']}")
            sys.exit(1)

        if api_result["errors"]:
            print("API 获取警告:")
            for err in api_result["errors"]:
                print(f"  - {err}")

        records = api_result["records"]

        # 保存原始 API 结果
        api_output = RAW_DIR / f"openlibrary_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(api_output, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"原始 API 数据已保存: {api_output}")

    else:
        # 从本地 CSV 读取
        input_path = Path(args.input) if args.input else RAW_DIR / "test_sample_openlibrary.csv"
        print(f"读取输入文件: {input_path}")

        records, errors = read_raw_csv(input_path)
        if errors:
            print("读取错误:")
            for err in errors:
                print(f"  - {err}")
            if not records:
                sys.exit(1)

    print(f"读取到 {len(records)} 条记录")

    # 规范化
    print("规范化数据...")
    normalized = [normalize_record(r) for r in records]

    # 质量检查
    if not args.no_check:
        print("执行数据质量检查...")
        report = check_quality(normalized)

        print(f"\n质量检查结果:")
        print(f"  有效记录: {report['valid_records']}")
        print(f"  有问题记录: {report['invalid_records']}")

        for flag, count in report["issue_counts"].items():
            if count > 0:
                print(f"  {QUALITY_FLAGS[flag]}: {count}")
    else:
        report = {"total_records": len(normalized), "valid_records": len(normalized), "invalid_records": 0, "issues": [], "issue_counts": {}}

    # 输出路径
    output_path = Path(args.output) if args.output else PROCESSED_DIR / "books_cleaned.csv"
    report_path = Path(args.report) if args.report else PROCESSED_DIR / "import_report.md"

    # 写入结果
    print(f"\n写入清洗后数据: {output_path}")
    write_processed_csv(normalized, output_path)

    print(f"写入质量报告: {report_path}")
    write_report(report, report_path)

    print("\n导入完成!")
    return report


if __name__ == "__main__":
    main()
