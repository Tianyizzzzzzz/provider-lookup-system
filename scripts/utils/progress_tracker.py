"""
进度跟踪工具模块
提供数据处理进度跟踪功能
"""

import time
from datetime import datetime, timedelta


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, total_items, report_interval=100000, task_name="处理"):
        """
        初始化进度跟踪器

        Args:
            total_items: 总项目数
            report_interval: 报告间隔
            task_name: 任务名称
        """
        self.total_items = total_items
        self.report_interval = report_interval
        self.task_name = task_name
        self.processed_items = 0
        self.start_time = time.time()
        self.last_report_time = self.start_time

        print(f"开始{self.task_name}，总计 {self.total_items:,} 项")

    def update(self, increment=1):
        """
        更新进度

        Args:
            increment: 增量
        """
        self.processed_items += increment

        # 检查是否需要报告进度
        if (self.processed_items % self.report_interval == 0 or
                self.processed_items == self.total_items):
            self._report_progress()

    def _report_progress(self):
        """报告当前进度"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # 计算进度百分比
        progress_percent = (self.processed_items / self.total_items) * 100

        # 计算处理速度
        items_per_second = self.processed_items / elapsed_time if elapsed_time > 0 else 0

        # 估算剩余时间
        if items_per_second > 0:
            remaining_items = self.total_items - self.processed_items
            estimated_remaining_seconds = remaining_items / items_per_second
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
            eta_str = f", 预计完成: {estimated_completion.strftime('%H:%M:%S')}"
        else:
            eta_str = ""

        print(f"{self.task_name}进度: {self.processed_items:,}/{self.total_items:,} "
              f"({progress_percent:.1f}%) - "
              f"速度: {items_per_second:,.0f} 项/秒{eta_str}")

    def finish(self):
        """完成进度跟踪"""
        total_time = time.time() - self.start_time
        average_speed = self.processed_items / total_time if total_time > 0 else 0

        print(f"{self.task_name}完成！")
        print(f"总用时: {total_time:.1f} 秒")
        print(f"平均速度: {average_speed:,.0f} 项/秒")
        print(f"总处理项目: {self.processed_items:,}")


def track_chunk_processing(chunks, chunk_size, task_name="处理数据块"):
    """
    跟踪分块处理进度

    Args:
        chunks: 数据块迭代器
        chunk_size: 每块大小
        task_name: 任务名称

    Yields:
        tuple: (chunk_number, chunk_data)
    """
    chunk_num = 0
    total_processed = 0
    start_time = time.time()

    for chunk_df in chunks:
        chunk_num += 1
        chunk_size_actual = len(chunk_df)
        total_processed += chunk_size_actual

        # 每10个块或每100万行报告一次进度
        if chunk_num % 10 == 0 or total_processed % 1000000 == 0:
            elapsed_time = time.time() - start_time
            speed = total_processed / elapsed_time if elapsed_time > 0 else 0

            print(f"{task_name} - 第 {chunk_num} 块，行数: {chunk_size_actual:,} | "
                  f"累计处理: {total_processed:,} 行 | "
                  f"速度: {speed:,.0f} 行/秒")

        yield chunk_num, chunk_df

    # 最终报告
    total_time = time.time() - start_time
    average_speed = total_processed / total_time if total_time > 0 else 0

    print(f"{task_name}完成！")
    print(f"总块数: {chunk_num:,}")
    print(f"总行数: {total_processed:,}")
    print(f"总用时: {total_time:.1f} 秒")
    print(f"平均速度: {average_speed:,.0f} 行/秒")


class BatchProcessor:
    """批处理器"""

    def __init__(self, batch_size=50000, task_name="批处理"):
        """
        初始化批处理器

        Args:
            batch_size: 批大小
            task_name: 任务名称
        """
        self.batch_size = batch_size
        self.task_name = task_name
        self.batch_count = 0
        self.total_processed = 0
        self.start_time = time.time()

    def process_batches(self, data, total_count=None):
        """
        分批处理数据

        Args:
            data: 数据列表
            total_count: 总数量（用于进度计算）

        Yields:
            tuple: (batch_number, batch_data, start_index, end_index)
        """
        total_items = total_count or len(data)

        for i in range(0, len(data), self.batch_size):
            end_idx = min(i + self.batch_size, len(data))
            batch = data[i:end_idx]

            self.batch_count += 1
            batch_size_actual = len(batch)
            self.total_processed += batch_size_actual

            # 进度报告
            progress_percent = (self.total_processed / total_items) * 100
            print(f"{self.task_name} - 批次 {self.batch_count} "
                  f"({i:,} - {end_idx:,}) | "
                  f"进度: {progress_percent:.1f}%")

            yield self.batch_count, batch, i, end_idx

    def finish(self):
        """完成批处理"""
        total_time = time.time() - self.start_time
        average_speed = self.total_processed / total_time if total_time > 0 else 0

        print(f"{self.task_name}完成！")
        print(f"总批次: {self.batch_count}")
        print(f"总处理项目: {self.total_processed:,}")
        print(f"总用时: {total_time:.1f} 秒")
        print(f"平均速度: {average_speed:,.0f} 项/秒")