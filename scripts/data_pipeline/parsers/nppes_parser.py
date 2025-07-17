"""
NPPES主数据解析脚本
从npidata_pfile.csv中提取基础provider信息
"""

import pandas as pd
import os
import sys

# 添加utils模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils import (
    read_csv_chunked, get_available_columns, log_progress,
    save_json_streaming, print_sample_record, track_chunk_processing
)


def parse_nppes_main():
    """
    解析NPPES主数据文件
    """

    # 文件路径配置
    data_dir = r"D:\EMRTS\PROVIDER_LOOKUP\data\nppes\NPPES_Data_Dissemination_June_2025_V2"
    output_dir = r"D:\EMRTS\PROVIDER_LOOKUP\output"

    main_file = os.path.join(data_dir, "npidata_pfile_20050523-20250608.csv")
    output_file = os.path.join(output_dir, "base_provider_data.json")

    log_progress("开始解析NPPES主数据")

    # 定义需要提取的列
    main_columns = {
        'NPI': 'npi',
        'Entity Type Code': 'entity_type_code',
        'Provider Last Name (Legal Name)': 'provider_last_name',
        'Provider First Name': 'provider_first_name',
        'Provider Middle Name': 'provider_middle_name',
        'Provider Organization Name (Legal Business Name)': 'organization_name',
        'Provider First Line Business Mailing Address': 'business_address_line1',
        'Provider Second Line Business Mailing Address': 'business_address_line2',
        'Provider Business Mailing Address City Name': 'business_city',
        'Provider Business Mailing Address State Name': 'business_state',
        'Provider Business Mailing Address Postal Code': 'business_postal_code',
        'Provider Business Mailing Address Country Code (If outside U.S.)': 'business_country',
        'Healthcare Provider Taxonomy Code_1': 'primary_taxonomy_code'
    }

    try:
        # 检查可用列
        available_columns = get_available_columns(main_file, main_columns)

        if not available_columns:
            log_progress("错误：没有找到任何需要的列")
            return False

        # 分块读取主文件
        chunk_size = 10000
        main_data_list = []

        log_progress("开始分块读取主文件")
        chunks = read_csv_chunked(
            main_file,
            chunk_size=chunk_size,
            dtype={'NPI': str},
            usecols=list(available_columns.keys())
        )

        for chunk_num, chunk_df in track_chunk_processing(chunks, chunk_size, "处理主数据"):
            # 重命名列
            chunk_df = chunk_df.rename(columns=available_columns)

            # 组合provider名字
            if 'provider_first_name' in chunk_df.columns and 'provider_last_name' in chunk_df.columns:
                chunk_df['provider_name'] = chunk_df.apply(
                    lambda row: ' '.join(filter(None, [
                        str(row.get('provider_first_name', '')).strip() if pd.notna(
                            row.get('provider_first_name')) else '',
                        str(row.get('provider_middle_name', '')).strip() if pd.notna(
                            row.get('provider_middle_name')) else '',
                        str(row.get('provider_last_name', '')).strip() if pd.notna(
                            row.get('provider_last_name')) else ''
                    ])), axis=1
                )

            # 使用organization_name作为备选名字
            if 'organization_name' in chunk_df.columns:
                chunk_df['provider_name'] = chunk_df.apply(
                    lambda row: row.get('provider_name', '') if row.get('provider_name', '').strip()
                    else str(row.get('organization_name', '')).strip() if pd.notna(
                        row.get('organization_name')) else '',
                    axis=1
                )

            main_data_list.append(chunk_df)

        # 合并所有块
        log_progress("合并所有数据块")
        main_data = pd.concat(main_data_list, ignore_index=True)
        log_progress(f"主数据处理完成，总记录数: {len(main_data):,}")

        # 转换为JSON格式
        log_progress("转换为JSON格式")
        result_data = []

        for _, row in main_data.iterrows():
            record = {
                'npi': row['npi'],
                'entity_type_code': row.get('entity_type_code'),
                'provider_name': row.get('provider_name', ''),
                'primary_address': {
                    'line1': row.get('business_address_line1'),
                    'line2': row.get('business_address_line2'),
                    'city': row.get('business_city'),
                    'state': row.get('business_state'),
                    'postal_code': row.get('business_postal_code'),
                    'country': row.get('business_country')
                },
                'primary_taxonomy_code': row.get('primary_taxonomy_code')
            }

            # 清理空值
            record = {k: v for k, v in record.items() if v is not None and v != ''}
            if 'primary_address' in record:
                record['primary_address'] = {k: v for k, v in record['primary_address'].items()
                                             if v is not None and v != ''}

            result_data.append(record)

        # 保存结果
        save_json_streaming(result_data, output_file)

        # 显示示例记录
        sample_keys = ['npi', 'entity_type_code', 'provider_name', 'primary_taxonomy_code']
        print_sample_record(result_data, sample_keys)

        log_progress("主数据解析完成")
        return True

    except Exception as e:
        log_progress(f"处理过程中出错: {e}")
        return False


if __name__ == "__main__":
    success = parse_nppes_main()
    if success:
        print("\n✅ 主数据解析成功完成！")
        print("输出文件: D:\\EMRTS\\PROVIDER_LOOKUP\\output\\base_provider_data.json")
        print("下一步: 运行 2_merge_locations.py")
    else:
        print("\n❌ 主数据解析失败")
        sys.exit(1)