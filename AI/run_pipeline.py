#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script thực thi toàn bộ luồng tiền xử lý dữ liệu cho AI/RecSys (RetailRocket).

Cách sử dụng:
    # 1. Chạy mặc định toàn bộ quy trình:
    python run_pipeline.py

    # 2. Lọc K-Core để giảm bớt độ thưa và tăng tốc độ train:
    python run_pipeline.py --min-user-interactions 3 --min-item-interactions 2

    # 3. Bỏ qua bước trích xuất item_properties nếu chỉ cần ma trận Collaborative Filtering:
    python run_pipeline.py --skip-features
"""

import argparse
import sys
from pathlib import Path

# Đảm bảo mã hóa UTF-8 cho terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Đảm bảo import được src
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.pipeline import run_data_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data Preprocessing Pipeline for RetailRocket RecSys",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Tỷ lệ phân chia tập Train theo mốc thời gian (0.8 = 80% train, 20% test)",
    )
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=1,
        help="Số lượng tương tác tối thiểu của mỗi người dùng (K-core filter cho User)",
    )
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=1,
        help="Số lượng tương tác tối thiểu của mỗi sản phẩm (K-core filter cho Item)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Bỏ qua việc trích xuất item_properties và category_tree",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_data_pipeline(
        raw_dir=ROOT_DIR / "data" / "raw",
        processed_dir=ROOT_DIR / "data" / "processed",
        split_ratio=args.split_ratio,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        build_features=not args.skip_features,
        verbose=True,
    )


if __name__ == "__main__":
    main()
