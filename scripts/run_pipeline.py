"""
NPPES数据处理管道控制脚本
按顺序执行所有数据处理步骤
"""

import os
import sys
import subprocess
from datetime import datetime


def run_script(script_name, script_description):
    """
    运行指定的脚本

    Args:
        script_name: 脚本文件名
        script_description: 脚本描述

    Returns:
        bool: 是否成功
    """
    print(f"\n{'=' * 60}")
    print(f"步骤: {script_description}")
    print(f"脚本: {script_name}")
    print(f"开始时间: {datetime.now()}")
    print(f"{'=' * 60}")

    script_path = os.path.join(os.path.dirname(__file__), script_name)

    if not os.path.exists(script_path):
        print(f"❌ 错误：脚本文件不存在: {script_path}")
        return False

    try:
        # 运行脚本
        result = subprocess.run([sys.executable, script_path],
                                capture_output=True,
                                text=True,
                                encoding='utf-8')

        # 打印输出
        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("错误信息:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✅ {script_description} 完成成功")
            return True
        else:
            print(f"❌ {script_description} 执行失败 (返回码: {result.returncode})")
            return False

    except Exception as e:
        print(f"❌ 运行脚本时出错: {e}")
        return False


def check_prerequisites():
    """检查必要的文件和目录"""
    print("检查运行环境...")

    # 检查数据目录
    data_dir = r"D:\EMRTS\PROVIDER_LOOKUP\data\nppes\NPPES_Data_Dissemination_June_2025_V2"
    if not os.path.exists(data_dir):
        print(f"❌ 错误：NPPES数据目录不存在: {data_dir}")
        return False

    # 检查主要数据文件
    required_files = [
        "npidata_pfile_20050523-20250608.csv"
    ]

    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        print(f"❌ 错误：缺少必要的数据文件: {missing_files}")
        return False

    # 检查可选文件
    optional_files = [
        "pl_pfile_20050523-20250608.csv",
        "endpoint_pfile_20050523-20250608.csv"
    ]

    for file in optional_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"✅ 找到可选文件: {file}")
        else:
            print(f"⚠️  可选文件不存在: {file} (将跳过相关处理)")

    # 检查taxonomy文件
    taxonomy_file = r"D:\EMRTS\PROVIDER_LOOKUP\data\taxonomy\taxonomy.csv"
    if os.path.exists(taxonomy_file):
        print(f"✅ 找到taxonomy文件: taxonomy.csv")
    else:
        print(f"⚠️  Taxonomy文件不存在: {taxonomy_file} (将跳过分类增强)")

    print("✅ 环境检查完成")
    return True


def run_pipeline(start_from=1):
    """
    运行完整的数据处理管道

    Args:
        start_from: 从第几步开始运行 (1-4)
    """

    # 定义处理步骤
    steps = [
        ("1_parse_nppes.py", "解析NPPES主数据"),
        ("2_merge_locations.py", "合并附加地址"),
        ("3_merge_endpoints.py", "合并端点信息"),
        ("4_enrich_taxonomy.py", "Taxonomy分类增强")
    ]

    print("🚀 NPPES数据处理管道启动")
    print(f"总步骤数: {len(steps)}")
    print(f"开始时间: {datetime.now()}")

    if start_from > 1:
        print(f"⏭️  从第 {start_from} 步开始")

    # 检查环境
    if not check_prerequisites():
        print("\n❌ 环境检查失败，无法继续")
        return False

    pipeline_start_time = datetime.now()

    # 执行步骤
    for i, (script_name, description) in enumerate(steps[start_from - 1:], start_from):
        success = run_script(script_name, f"步骤 {i}: {description}")

        if not success:
            print(f"\n❌ 管道在步骤 {i} 失败，停止执行")
            print("您可以修复问题后使用以下命令从失败的步骤重新开始:")
            print(f"python run_pipeline.py --start-from {i}")
            return False

    # 管道完成
    total_time = datetime.now() - pipeline_start_time

    print(f"\n🎉 NPPES数据处理管道成功完成！")
    print(f"总用时: {total_time}")
    print(f"完成时间: {datetime.now()}")
    print("\n📋 处理结果:")

    # 显示输出文件信息
    output_files = [
        ("base_provider_data.json", "基础provider数据"),
        ("provider_with_locations.json", "包含附加地址的数据"),
        ("provider_with_endpoints.json", "包含端点的数据"),
        ("provider_with_taxonomy.json", "最终完整数据")
    ]

    output_dir = r"D:\EMRTS\PROVIDER_LOOKUP\output"

    for filename, description in output_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / 1024 / 1024
            print(f"  ✅ {description}: {filename} ({file_size:.2f} MB)")
        else:
            print(f"  ❌ {description}: {filename} (未生成)")

    return True


def main():
    """主函数"""

    # 解析命令行参数
    start_from = 1
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("NPPES数据处理管道")
            print("用法:")
            print("  python run_pipeline.py              # 运行完整管道")
            print("  python run_pipeline.py --start-from N  # 从第N步开始")
            print("  python run_pipeline.py --help          # 显示帮助")
            print("\n步骤说明:")
            print("  1. 解析NPPES主数据")
            print("  2. 合并附加地址")
            print("  3. 合并端点信息")
            print("  4. Taxonomy分类增强")
            return

        elif sys.argv[1] == "--start-from" and len(sys.argv) > 2:
            try:
                start_from = int(sys.argv[2])
                if start_from < 1 or start_from > 4:
                    print("❌ 错误：start-from 参数必须在 1-4 之间")
                    return
            except ValueError:
                print("❌ 错误：start-from 参数必须是数字")
                return

    # 运行管道
    success = run_pipeline(start_from)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()