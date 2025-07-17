"""
合并端点脚本
将endpoint_pfile.csv中的端点信息合并到provider数据中
"""

import pandas as pd
import os
import sys

# 添加utils模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils import (
    read_csv_chunked, read_csv_full, check_file_size, log_progress,
    load_json, save_json_streaming, print_sample_record,
    track_chunk_processing, ProgressTracker
)


def process_endpoints_chunk(chunk_df, endpoints_data):
    """处理endpoint数据块"""
    for npi, group in chunk_df.groupby('NPI'):
        if npi not in endpoints_data:
            endpoints_data[npi] = []

        for _, row in group.iterrows():
            endpoint = {}
            # 提取endpoint信息
            endpoint_fields = {
                'Endpoint': 'endpoint_url',
                'Endpoint Type': 'endpoint_type',
                'Endpoint Type Description': 'endpoint_type_description',
                'Endpoint Description': 'endpoint_description',
                'Affiliation': 'affiliation'
            }

            for old_field, new_field in endpoint_fields.items():
                if old_field in row and pd.notna(row[old_field]):
                    endpoint[new_field] = str(row[old_field]).strip()

            if endpoint:  # 只添加非空的endpoint
                endpoints_data[npi].append(endpoint)


def merge_endpoints():
    """
    合并端点数据到provider数据中
    """

    # 文件路径配置
    data_dir = r"D:\EMRTS\PROVIDER_LOOKUP\data\nppes\NPPES_Data_Dissemination_June_2025_V2"
    output_dir = r"D:\EMRTS\PROVIDER_LOOKUP\output"

    input_file = os.path.join(output_dir, "provider_with_locations.json")
    endpoint_file = os.path.join(data_dir, "endpoint_pfile_20050523-20250608.csv")
    output_file = os.path.join(output_dir, "provider_with_endpoints.json")

    log_progress("开始合并端点数据")

    # 检查输入文件
    if not os.path.exists(input_file):
        log_progress(f"错误：输入数据文件不存在: {input_file}")
        log_progress("请先运行 2_merge_locations.py")
        return False

    if not os.path.exists(endpoint_file):
        log_progress(f"警告：端点文件不存在: {endpoint_file}")
        log_progress("将跳过端点合并，直接复制现有数据")

        # 直接复制现有数据
        try:
            provider_data = load_json(input_file)
            # 为每条记录添加空的endpoints字段
            for record in provider_data:
                record['endpoints'] = []

            save_json_streaming(provider_data, output_file)
            log_progress("现有数据复制完成（无端点）")
            return True
        except Exception as e:
            log_progress(f"复制现有数据时出错: {e}")
            return False

    try:
        # 1. 读取现有provider数据
        provider_data = load_json(input_file)

        # 2. 处理端点文件
        log_progress("正在处理端点文件")

        # 检查文件大小决定处理方式
        file_size_mb = check_file_size(endpoint_file)
        log_progress(f"端点文件大小: {file_size_mb:.2f} MB")

        endpoints_data = {}
        chunk_size = 10000

        if file_size_mb > 500:  # 大文件使用分块处理
            log_progress("使用分块处理端点文件")
            chunks = read_csv_chunked(
                endpoint_file,
                chunk_size=chunk_size,
                dtype={'NPI': str}
            )

            for chunk_num, chunk_df in track_chunk_processing(chunks, chunk_size, "处理端点"):
                process_endpoints_chunk(chunk_df, endpoints_data)
        else:
            log_progress("直接读取端点文件")
            endpoint_df = read_csv_full(endpoint_file, dtype={'NPI': str})
            log_progress(f"端点文件记录数: {len(endpoint_df):,}")
            process_endpoints_chunk(endpoint_df, endpoints_data)

        log_progress(f"端点处理完成，涉及 {len(endpoints_data):,} 个NPI")

        # 3. 合并数据
        log_progress("正在合并端点到provider数据")

        tracker = ProgressTracker(len(provider_data), 100000, "合并端点")

        merged_count = 0
        for record in provider_data:
            npi = record.get('npi')
            if npi and npi in endpoints_data:
                record['endpoints'] = endpoints_data[npi]
                merged_count += 1
            else:
                record['endpoints'] = []

            tracker.update()

        tracker.finish()

        log_progress(f"合并完成:")
        log_progress(f"  包含端点的记录: {merged_count:,}")
        log_progress(f"  总记录数: {len(provider_data):,}")
        log_progress(f"  端点覆盖率: {(merged_count / len(provider_data) * 100):.2f}%")

        # 4. 保存合并后的数据
        save_json_streaming(provider_data, output_file)

        # 显示示例记录
        sample_keys = ['npi', 'provider_name', 'primary_taxonomy_code', 'additional_locations', 'endpoints']
        print_sample_record(provider_data, sample_keys)

        log_progress("端点合并完成")
        return True

    except Exception as e:
        log_progress(f"处理过程中出错: {e}")
        return False


if __name__ == "__main__":
    success = merge_endpoints()
    if success:
        print("\n✅ 端点合并成功完成！")
        print("输出文件: D:\\EMRTS\\PROVIDER_LOOKUP\\output\\provider_with_endpoints.json")
        print("下一步: 运行 4_enrich_taxonomy.py")
    else:
        print("\n❌ 端点合并失败")
        sys.exit(1)