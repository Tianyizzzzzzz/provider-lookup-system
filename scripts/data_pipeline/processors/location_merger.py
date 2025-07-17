"""
合并附加地址脚本
将pl_pfile.csv中的附加地址信息合并到基础数据中
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


def process_locations_chunk(chunk_df, additional_locations):
    """处理附加地址数据块"""
    for npi, group in chunk_df.groupby('NPI'):
        if npi not in additional_locations:
            additional_locations[npi] = []

        for _, row in group.iterrows():
            location = {}
            # 提取地址信息
            address_fields = {
                'Provider Secondary Practice Location Address- Address Line 1': 'address_line1',
                'Provider Secondary Practice Location Address- Address Line 2': 'address_line2',
                'Provider Secondary Practice Location Address - City Name': 'city',
                'Provider Secondary Practice Location Address - State Name': 'state',
                'Provider Secondary Practice Location Address - Postal Code': 'postal_code',
                'Provider Secondary Practice Location Address - Country Code (If outside US)': 'country',
                'Provider Secondary Practice Location Address - Telephone Number': 'phone'
            }

            for old_field, new_field in address_fields.items():
                if old_field in row and pd.notna(row[old_field]):
                    location[new_field] = str(row[old_field]).strip()

            if location:  # 只添加非空的地址
                additional_locations[npi].append(location)


def merge_locations():
    """
    合并附加地址数据到基础provider数据中
    """

    # 文件路径配置
    data_dir = r"D:\EMRTS\PROVIDER_LOOKUP\data\nppes\NPPES_Data_Dissemination_June_2025_V2"
    output_dir = r"D:\EMRTS\PROVIDER_LOOKUP\output"

    base_file = os.path.join(output_dir, "base_provider_data.json")
    pl_file = os.path.join(data_dir, "pl_pfile_20050523-20250608.csv")
    output_file = os.path.join(output_dir, "provider_with_locations.json")

    log_progress("开始合并附加地址数据")

    # 检查输入文件
    if not os.path.exists(base_file):
        log_progress(f"错误：基础数据文件不存在: {base_file}")
        log_progress("请先运行 1_parse_nppes.py")
        return False

    if not os.path.exists(pl_file):
        log_progress(f"警告：附加地址文件不存在: {pl_file}")
        log_progress("将跳过附加地址合并，直接复制基础数据")

        # 直接复制基础数据
        try:
            provider_data = load_json(base_file)
            # 为每条记录添加空的additional_locations字段
            for record in provider_data:
                record['additional_locations'] = []

            save_json_streaming(provider_data, output_file)
            log_progress("基础数据复制完成（无附加地址）")
            return True
        except Exception as e:
            log_progress(f"复制基础数据时出错: {e}")
            return False

    try:
        # 1. 读取基础provider数据
        provider_data = load_json(base_file)

        # 2. 处理附加地址文件
        log_progress("正在处理附加地址文件")

        # 检查文件大小决定处理方式
        file_size_mb = check_file_size(pl_file)
        log_progress(f"附加地址文件大小: {file_size_mb:.2f} MB")

        additional_locations = {}
        chunk_size = 10000

        if file_size_mb > 500:  # 大文件使用分块处理
            log_progress("使用分块处理附加地址文件")
            chunks = read_csv_chunked(
                pl_file,
                chunk_size=chunk_size,
                dtype={'NPI': str}
            )

            for chunk_num, chunk_df in track_chunk_processing(chunks, chunk_size, "处理附加地址"):
                process_locations_chunk(chunk_df, additional_locations)
        else:
            log_progress("直接读取附加地址文件")
            pl_df = read_csv_full(pl_file, dtype={'NPI': str})
            log_progress(f"附加地址文件记录数: {len(pl_df):,}")
            process_locations_chunk(pl_df, additional_locations)

        log_progress(f"附加地址处理完成，涉及 {len(additional_locations):,} 个NPI")

        # 3. 合并数据
        log_progress("正在合并附加地址到provider数据")

        tracker = ProgressTracker(len(provider_data), 100000, "合并附加地址")

        merged_count = 0
        for record in provider_data:
            npi = record.get('npi')
            if npi and npi in additional_locations:
                record['additional_locations'] = additional_locations[npi]
                merged_count += 1
            else:
                record['additional_locations'] = []

            tracker.update()

        tracker.finish()

        log_progress(f"合并完成:")
        log_progress(f"  包含附加地址的记录: {merged_count:,}")
        log_progress(f"  总记录数: {len(provider_data):,}")
        log_progress(f"  附加地址覆盖率: {(merged_count / len(provider_data) * 100):.2f}%")

        # 4. 保存合并后的数据
        save_json_streaming(provider_data, output_file)

        # 显示示例记录
        sample_keys = ['npi', 'provider_name', 'primary_taxonomy_code', 'additional_locations']
        print_sample_record(provider_data, sample_keys)

        log_progress("附加地址合并完成")
        return True

    except Exception as e:
        log_progress(f"处理过程中出错: {e}")
        return False


if __name__ == "__main__":
    success = merge_locations()
    if success:
        print("\n✅ 附加地址合并成功完成！")
        print("输出文件: D:\\EMRTS\\PROVIDER_LOOKUP\\output\\provider_with_locations.json")
        print("下一步: 运行 3_merge_endpoints.py")
    else:
        print("\n❌ 附加地址合并失败")
        sys.exit(1)