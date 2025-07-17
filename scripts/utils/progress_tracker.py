"""
Progress tracking utility module
Provides data processing progress tracking functionality
"""

import time
from datetime import datetime, timedelta


class ProgressTracker:
    """Progress tracker"""

    def __init__(self, total_items, report_interval=100000, task_name="Processing"):
        """
        Initialize progress tracker

        Args:
            total_items: Total number of items
            report_interval: Reporting interval
            task_name: Task name
        """
        self.total_items = total_items
        self.report_interval = report_interval
        self.task_name = task_name
        self.processed_items = 0
        self.start_time = time.time()
        self.last_report_time = self.start_time

        print(f"Starting {self.task_name}, total {self.total_items:,} items")

    def update(self, increment=1):
        """
        Update progress

        Args:
            increment: Increment amount
        """
        self.processed_items += increment

        # Check if progress report is needed
        if (self.processed_items % self.report_interval == 0 or
                self.processed_items == self.total_items):
            self._report_progress()

    def _report_progress(self):
        """Report current progress"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Calculate progress percentage
        progress_percent = (self.processed_items / self.total_items) * 100

        # Calculate processing speed
        items_per_second = self.processed_items / elapsed_time if elapsed_time > 0 else 0

        # Estimate remaining time
        if items_per_second > 0:
            remaining_items = self.total_items - self.processed_items
            estimated_remaining_seconds = remaining_items / items_per_second
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
            eta_str = f", ETA: {estimated_completion.strftime('%H:%M:%S')}"
        else:
            eta_str = ""

        print(f"{self.task_name} progress: {self.processed_items:,}/{self.total_items:,} "
              f"({progress_percent:.1f}%) - "
              f"Speed: {items_per_second:,.0f} items/sec{eta_str}")

    def finish(self):
        """Finish progress tracking"""
        total_time = time.time() - self.start_time
        average_speed = self.processed_items / total_time if total_time > 0 else 0

        print(f"{self.task_name} completed!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average speed: {average_speed:,.0f} items/sec")
        print(f"Total processed items: {self.processed_items:,}")


def track_chunk_processing(chunks, chunk_size, task_name="Processing data chunks"):
    """
    Track chunk processing progress

    Args:
        chunks: Data chunk iterator
        chunk_size: Size of each chunk
        task_name: Task name

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

        # Report progress every 10 chunks or every 1 million rows
        if chunk_num % 10 == 0 or total_processed % 1000000 == 0:
            elapsed_time = time.time() - start_time
            speed = total_processed / elapsed_time if elapsed_time > 0 else 0

            print(f"{task_name} - Chunk {chunk_num}, rows: {chunk_size_actual:,} | "
                  f"Total processed: {total_processed:,} rows | "
                  f"Speed: {speed:,.0f} rows/sec")

        yield chunk_num, chunk_df

    # Final report
    total_time = time.time() - start_time
    average_speed = total_processed / total_time if total_time > 0 else 0

    print(f"{task_name} completed!")
    print(f"Total chunks: {chunk_num:,}")
    print(f"Total rows: {total_processed:,}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average speed: {average_speed:,.0f} rows/sec")


class BatchProcessor:
    """Batch processor"""

    def __init__(self, batch_size=50000, task_name="Batch processing"):
        """
        Initialize batch processor

        Args:
            batch_size: Batch size
            task_name: Task name
        """
        self.batch_size = batch_size
        self.task_name = task_name
        self.batch_count = 0
        self.total_processed = 0
        self.start_time = time.time()

    def process_batches(self, data, total_count=None):
        """
        Process data in batches

        Args:
            data: Data list
            total_count: Total count (for progress calculation)

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

            # Progress report
            progress_percent = (self.total_processed / total_items) * 100
            print(f"{self.task_name} - Batch {self.batch_count} "
                  f"({i:,} - {end_idx:,}) | "
                  f"Progress: {progress_percent:.1f}%")

            yield self.batch_count, batch, i, end_idx

    def finish(self):
        """Finish batch processing"""
        total_time = time.time() - self.start_time
        average_speed = self.total_processed / total_time if total_time > 0 else 0

        print(f"{self.task_name} completed!")
        print(f"Total batches: {self.batch_count}")
        print(f"Total processed items: {self.total_processed:,}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average speed: {average_speed:,.0f} items/sec")