import pandas as pd
from pathlib import Path

# Xác định đường dẫn dự án
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_events.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not INPUT_FILE.exists():
        print(f"Chưa có file {INPUT_FILE}, vui lòng chạy pipeline làm sạch trước.")
        return

    # Đọc dữ liệu đã làm sạch
    df = pd.read_csv(INPUT_FILE)

    print("===== 1. TỔNG QUAN =====")
    print(f"Số dòng: {len(df):,}")
    print(f"Số người dùng: {df['visitorid'].nunique():,}")
    print(f"Số sản phẩm: {df['itemid'].nunique():,}")

    # Phân bố sự kiện
    print("\n===== 2. PHÂN BỐ SỰ KIỆN =====")
    event_summary = (
        df["event"]
        .value_counts()
        .rename_axis("event")
        .reset_index(name="count")
    )
    event_summary["percentage"] = (
        event_summary["count"] / len(df) * 100
    ).round(2)

    print(event_summary.to_string(index=False))
    event_summary.to_csv(
        OUTPUT_DIR / "event_summary.csv", index=False
    )

    # Hoạt động người dùng
    print("\n===== 3. HOẠT ĐỘNG NGƯỜI DÙNG =====")
    user_activity = df.groupby("visitorid").size()

    print(user_activity.describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]
    ))

    single_event_users = int((user_activity == 1).sum())
    print("Người dùng chỉ có 1 sự kiện:", single_event_users)
    print(
        "Tỷ lệ người dùng chỉ có 1 sự kiện:",
        round((user_activity == 1).mean() * 100, 2), "%"
    )

    # Hoạt động sản phẩm
    print("\n===== 4. HOẠT ĐỘNG SẢN PHẨM =====")
    item_activity = df.groupby("itemid").size()

    print(item_activity.describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]
    ))

    # Top 10 sản phẩm được tương tác nhiều nhất
    print("\n===== 5. TOP 10 SẢN PHẨM =====")
    top_items = (
        df.groupby("itemid")
        .agg(
            total_events=("event", "size"),
            unique_visitors=("visitorid", "nunique")
        )
        .sort_values("total_events", ascending=False)
        .head(10)
        .reset_index()
    )

    print(top_items.to_string(index=False))
    top_items.to_csv(
        OUTPUT_DIR / "top_10_items.csv", index=False
    )

    # Hoạt động theo ngày và loại sự kiện
    print("\n===== 6. HOẠT ĐỘNG THEO NGÀY =====")
    df["event_time"] = pd.to_datetime(
        df["event_time"], utc=True, errors="coerce"
    )

    daily_events = (
        df.dropna(subset=["event_time"])
        .assign(date=lambda x: x["event_time"].dt.date)
        .groupby(["date", "event"])
        .size()
        .unstack(fill_value=0)
    )

    print(daily_events.head(10))
    daily_events.to_csv(OUTPUT_DIR / "daily_events.csv")

    print("\n===== HOÀN TẤT =====")
    print("Kết quả được lưu tại:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

