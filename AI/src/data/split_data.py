"""Module phân chia dữ liệu theo thời gian (Temporal Train/Test Split).

Tại sao phải chia theo thời gian trong Recommender Systems (AI/RecSys)?
- Nếu chia ngẫu nhiên (random k-fold split), mô hình sẽ học tương tác trong tương lai để dự đoán quá khứ
  -> Gây Data Leakage nghiêm trọng và làm sai lệch chỉ số đánh giá thực tế.
- Chia theo mốc thời gian (Temporal Split) phản ánh đúng kịch bản triển khai AI:
  Model được train trên dữ liệu quá khứ và dự đoán hành vi trong tương lai.
"""

from pathlib import Path
import sys
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_events.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def split_events(input_file=None, output_dir=None, split_ratio=0.8, verbose=True):
    in_path = Path(input_file) if input_file else DEFAULT_INPUT_FILE
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    if not in_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {in_path}")

    if verbose:
        print("\n" + "=" * 60)
        print("BƯỚC 2: PHÂN CHIA DỮ LIỆU THEO THỜI GIAN (TEMPORAL SPLIT)")
        print("=" * 60)
        print(f"Tỷ lệ Train: {split_ratio * 100:.1f}%, Test: {(1 - split_ratio) * 100:.1f}%")

    df = pd.read_csv(
        in_path,
        usecols=["timestamp", "visitorid", "event", "itemid", "event_time"],
    )

    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)

    # Tính mốc thời gian phân chia (Cutoff)
    cutoff_time = df["event_time"].quantile(split_ratio)
    cutoff_timestamp = int(df.loc[df["event_time"] < cutoff_time, "timestamp"].max())

    train_df = df[df["event_time"] < cutoff_time].copy()
    test_df = df[df["event_time"] >= cutoff_time].copy()

    train_file = out_dir / "train_events.csv"
    test_file = out_dir / "test_events.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    if verbose:
        print(f"Mốc thời gian cắt (Cutoff): {cutoff_time} (timestamp: {cutoff_timestamp})")
        print(f"Train events: {len(train_df):,} ({len(train_df)/len(df):.2%}) | Từ {train_df['event_time'].min()} đến {train_df['event_time'].max()}")
        print(f"Test events:  {len(test_df):,} ({len(test_df)/len(df):.2%}) | Từ {test_df['event_time'].min()} đến {test_df['event_time'].max()}")
        print(f"Train - Users: {train_df['visitorid'].nunique():,}, Items: {train_df['itemid'].nunique():,}")
        print(f"Test  - Users: {test_df['visitorid'].nunique():,}, Items: {test_df['itemid'].nunique():,}")
        print(f"Đã lưu: {train_file}")
        print(f"Đã lưu: {test_file}")

    return train_df, test_df, cutoff_timestamp


if __name__ == "__main__":
    split_events()
