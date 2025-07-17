"""
分类增强脚本
根据taxonomy代码为provider数据添加classification和specialization信息
"""

import pandas as pd
import os
import sys

# 添加utils模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils import (
    read_csv_full, log_progress, load_json, save_json_streaming,
    print_sample_record, ProgressTracker
)


def enrich_taxonomy():
    """
    使用taxonomy数据增强provider信息
    """

    # 文件路径配置
    taxonomy_file = r"D:\EMRTS\PROVIDER_LOOKUP\data\taxonomy\taxonomy.csv"
    input_file = r"D:\EMRTS\PROVIDER_LOOKUP\output\provider_with_endpoints.json"
    output_file = r"D:\EMRTS\PROVIDER_LOOKUP\output\provider_with_taxonomy.json"

    log_progress("开始taxonomy分类增强")

    # 检查输入文件
    if not os.path.exists(input_file):
        log_progress(f"错误：输入数据文件不存在: {input_file}")
        log_progress("请先运行 3_merge_endpoints.py")
        return False

    if not os.path.exists(taxonomy_file):
        log_progress(f"错误：taxonomy文件不存在: {taxonomy_file}")
        return False

    try:
        # 1. 读取taxonomy数据
        log_progress("正在读取taxonomy数据")
        taxonomy_df = read_csv_full(taxonomy_file, dtype={'Code': str})
        log_progress(f"Taxonomy数据加载完成，共 {len(taxonomy_df):,} 条记录")

        # 显示taxonomy文件的列名
        log_progress(f"Taxonomy文件列名: {list(taxonomy_df.columns)}")

        # 检查关键列是否存在
        if 'Code' not in taxonomy_df.columns:
            log_progress("错误：taxonomy文件缺少'Code'列")
            return False

        # 智能检测classification和specialization列
        classification_col = None
        specialization_col = None

        for col in taxonomy_df.columns:
            col_lower = col.lower()
            if 'classification' in col_lower and not classification_col:
                classification_col = col
            elif ('specialization' in col_lower or 'specialty' in col_lower) and not specialization_col:
                specialization_col = col

        # 如果没有找到，使用默认假设
        if not classification_col and len(taxonomy_df.columns) >= 2:
            classification_col = taxonomy_df.columns[1]
            log_progress(f"使用 '{classification_col}' 作为classification列")

        if not specialization_col and len(taxonomy_df.columns) >= 3:
            specialization_col = taxonomy_df.columns[2]
            log_progress(f"使用 '{specialization_col}' 作为specialization列")

        # 创建taxonomy字典用于快速查找
        log_progress("创建taxonomy查找字典")
        taxonomy_dict = {}

        for _, row in taxonomy_df.iterrows():
            code = str(row['Code']).strip() if pd.notna(row['Code']) else ''
            if code:
                taxonomy_dict[code] = {
                    'classification': str(row[classification_col]).strip() if classification_col and pd.notna(
                        row[classification_col]) else '',
                    'specialization': str(row[specialization_col]).strip() if specialization_col and pd.notna(
                        row[specialization_col]) else ''
                }

        log_progress(f"Taxonomy字典创建完成，共 {len(taxonomy_dict):,} 个有效代码")

        # 2. 读取provider数据
        provider_data = load_json(input_file)

        # 3. 合并taxonomy信息
        log_progress("正在合并taxonomy信息")

        tracker = ProgressTracker(len(provider_data), 100000, "分类增强")

        matched_count = 0
        unmatched_count = 0

        for record in provider_data:
            # 获取primary_taxonomy_code，安全处理各种数据类型
            taxonomy_code_raw = record.get('primary_taxonomy_code')

            if taxonomy_code_raw is None or taxonomy_code_raw == '' or (
                    isinstance(taxonomy_code_raw, float) and pd.isna(taxonomy_code_raw)):
                taxonomy_code = ''
            else:
                taxonomy_code = str(taxonomy_code_raw).strip()

            if taxonomy_code and taxonomy_code in taxonomy_dict:
                # 找到匹配的taxonomy信息
                taxonomy_info = taxonomy_dict[taxonomy_code]
                record['classification'] = taxonomy_info['classification']
                record['specialization'] = taxonomy_info['specialization']
                matched_count += 1
            else:
                # 没有找到匹配的taxonomy信息
                record['classification'] = ''
                record['specialization'] = ''
                unmatched_count += 1

            tracker.update()

        tracker.finish()

        log_progress(f"分类增强完成:")
        log_progress(f"  匹配成功: {matched_count:,} 条记录")
        log_progress(f"  未匹配: {unmatched_count:,} 条记录")
        log_progress(f"  匹配率: {(matched_count / len(provider_data) * 100):.2f}%")

        # 4. 保存最终数据
        save_json_streaming(provider_data, output_file)

        # 显示示例记录
        sample_keys = ['npi', 'provider_name', 'primary_taxonomy_code', 'classification', 'specialization']
        print_sample_record(provider_data, sample_keys)

        log_progress("Taxonomy分类增强完成")
        return True

    except Exception as e:
        log_progress(f"处理过程中出错: {e}")
        return False


if __name__ == "__main__":
    success = enrich_taxonomy()
    if success:
        print("\n✅ Taxonomy分类增强成功完成！")
        print("输出文件: D:\\EMRTS\\PROVIDER_LOOKUP\\output\\provider_with_taxonomy.json")
        print("🎉 所有数据处理步骤已完成！")
    else:
        print("\n❌ Taxonomy分类增强失败")
        sys.exit(1)