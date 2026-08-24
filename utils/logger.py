"""训练日志：TensorBoard + CSV。

- TensorBoard: scalar 与直方图
- CSV: 关键指标追加写，便于后期汇总成实验表格
"""

from __future__ import annotations

import csv
import os

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str, csv_name: str = "metrics.csv") -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.tb = SummaryWriter(log_dir=log_dir)
        self.csv_path = os.path.join(log_dir, csv_name)
        self._csv_fields: list[str] | None = None
        self._csv_file = None
        self._csv_writer = None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.tb.add_scalar(tag, value, step)

    def log_metrics(self, metrics: dict, step: int, prefix: str = "") -> None:
        for key, value in metrics.items():
            self.tb.add_scalar(f"{prefix}/{key}" if prefix else key, value, step)

    def write_csv_row(self, row: dict) -> None:
        """把一行指标写入 CSV（首行自动写表头）。"""
        fields = list(row.keys())
        if self._csv_fields is None:
            self._csv_fields = fields
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fields)
            self._csv_writer.writeheader()
        else:
            for f in fields:
                if f not in self._csv_fields:
                    self._csv_fields.append(f)
                    self._csv_writer.fieldnames = self._csv_fields
                    self._csv_writer.writeheader()
            for f in self._csv_fields:
                row.setdefault(f, "")
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        self.tb.close()
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
