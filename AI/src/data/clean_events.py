"""Module làm sạch dữ liệu sự kiện (Events Ingestion & Cleaning).

Chức năng:
- Đọc dữ liệu thô events.csv từ data/raw/.
- Loại bỏ dòng trùng lặp hoàn toàn và dòng thiếu trường định danh cốt lõi.
- Chuẩn hóa kiểu dữ liệu ID (visitorid, itemid, timestamp là số nguyên dương).
- Lọc các hành vi tương tác hợp lệ: view, addtocart, transaction.
- Chuyển đổi timestamp Unix ms sang datetime UTC.
- Sắp xếp dữ liệu theo thứ tự thời gian tăng dần.
- Xuất dữ liệu sạch ra data/processed/cleaned_events.csv.
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
DEFAULT_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "events.csv"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_events.csv"


def clean_events(raw_file=None, output_file=None, verbose=True):
    raw_path = Path(raw_file) if raw_file else DEFAULT_RAW_FILE
    out_path = Path(output_file) if output_file else DEFAULT_OUTPUT_FILE

    if not raw_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc: {raw_path}")

    if verbose:
        print("\n" + "=" * 60)
        print("BƯỚC 1: LÀM SẠCH DỮ LIỆU SỰ KIỆN (CLEAN EVENTS)")
        print("=" * 60)
        print(f"Đang đọc: {raw_path}...")

    df = pd.read_csv(raw_path)
    initial_rows = len(df)

    if verbose:
        print(f"Tổng số dòng ban đầu: {initial_rows:,}")

    # 1. Kiểm tra các cột bắt buộc
    required_cols = ["timestamp", "visitorid", "event", "itemid"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing_cols}")

    # 2. Loại bỏ trùng lặp hoàn toàn
    duplicated = df.duplicated().sum()
    if duplicated > 0:
        df = df.drop_duplicates()
        if verbose:
            print(f"Đã loại bỏ {duplicated:,} dòng trùng lặp hoàn toàn.")

    # 3. Loại bỏ dòng thiếu ID cốt lõi
    df = df.dropna(subset=required_cols)

    # 4. Ép kiểu số nguyên và lọc ID hợp lệ
    for col in ["timestamp", "visitorid", "itemid"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "visitorid", "itemid"])
    df = df[
        (df["timestamp"] >= 0)
        & (df["visitorid"] >= 0)
        & (df["itemid"] >= 0)
    ].copy()

    df["timestamp"] = df["timestamp"].astype("int64")
    df["visitorid"] = df["visitorid"].astype("int64")
    df["itemid"] = df["itemid"].astype("int64")

    # 5. Chuẩn hóa tên loại sự kiện (lowercase, strip whitespace)
    df["event"] = df["event"].astype(str).str.strip().str.lower()
    valid_events = ["view", "addtocart", "transaction"]
    df = df[df["event"].isin(valid_events)].copy()

    # 6. Chuyển Unix ms sang datetime UTC
    df["event_time"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["event_time"])

    # 7. Sắp xếp theo dòng thời gian
    df = df.sort_values(["timestamp", "visitorid", "itemid"]).reset_index(drop=True)

    # 8. Lưu kết quả
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    if verbose:
        print(f"Số dòng sau làm sạch: {len(df):,} (Đã loại: {initial_rows - len(df):,})")
        print("Phân bố loại sự kiện:")
        print(df["event"].value_counts().to_string())
        print(f"Khoảng thời gian: {df['event_time'].min()} đến {df['event_time'].max()}")
        print(f"Đã lưu kết quả tại: {out_path}")

    return df


if __name__ == "__main__":
    clean_events()
