import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EDA_DIR = PROJECT_ROOT / "results" / "eda"

EVENTS_FILE = PROCESSED_DIR / "cleaned_events.csv"
ITEM_IDS_FILE = EDA_DIR / "item_property_item_ids.csv"
CATEGORY_FILE = RAW_DIR / "category_tree.csv"


def main():
    if not EVENTS_FILE.exists() or not ITEM_IDS_FILE.exists():
        print("Cần có cleaned_events.csv và item_property_item_ids.csv trước khi kiểm tra.")
        return

    print("Đang đọc danh sách sản phẩm...")

    event_items = set(
        pd.read_csv(
            EVENTS_FILE,
            usecols=["itemid"]
        )["itemid"].dropna().unique()
    )

    property_items = set(
        pd.read_csv(
            ITEM_IDS_FILE,
            usecols=["itemid"]
        )["itemid"].dropna().unique()
    )

    print("\n===== 1. ĐỘ BAO PHỦ SẢN PHẨM =====")
    print("Số sản phẩm trong events:", len(event_items))
    print("Số sản phẩm trong item_properties:", len(property_items))

    common_items = event_items & property_items
    event_only = event_items - property_items
    property_only = property_items - event_items

    print("Sản phẩm xuất hiện trong cả hai:", len(common_items))
    print("Sản phẩm chỉ có trong events:", len(event_only))
    print("Sản phẩm chỉ có trong item_properties:", len(property_only))

    if event_items:
        print(
            "Tỷ lệ sản phẩm trong events có thuộc tính: "
            f"{len(common_items) / len(event_items):.2%}"
        )

    print("\n===== 2. ĐỘ BAO PHỦ SỰ KIỆN =====")
    event_item_series = pd.read_csv(
        EVENTS_FILE,
        usecols=["itemid"]
    )["itemid"]

    total_events = len(event_item_series)
    matched_events = event_item_series.isin(property_items).sum()

    print("Tổng số sự kiện:", total_events)
    print("Sự kiện có sản phẩm xuất hiện trong item_properties:", int(matched_events))

    if total_events:
        print(
            "Tỷ lệ sự kiện được bao phủ: "
            f"{matched_events / total_events:.2%}"
        )

    print("\n===== 3. ĐỐI CHIẾU CATEGORYID =====")
    category_values = set()

    for file_name in [
        "item_properties_part1.csv",
        "item_properties_part2.csv",
    ]:
        file_path = RAW_DIR / file_name
        if not file_path.exists():
            continue

        for chunk in pd.read_csv(
            file_path,
            usecols=["property", "value"],
            dtype={"property": "string", "value": "string"},
            chunksize=500_000,
        ):
            rows = chunk.loc[
                chunk["property"] == "categoryid", "value"
            ].dropna()
            category_values.update(rows.unique())

    category_tree = pd.read_csv(CATEGORY_FILE)
    tree_ids = set(category_tree["categoryid"].dropna().astype("int64"))

    valid_values = set()
    invalid_values = set()

    for value in category_values:
        try:
            number = float(value)
            if number.is_integer():
                category_id = int(number)
                if category_id in tree_ids:
                    valid_values.add(value)
                else:
                    invalid_values.add(value)
            else:
                invalid_values.add(value)
        except (ValueError, TypeError):
            invalid_values.add(value)

    print("Số giá trị categoryid khác nhau trong properties:", len(category_values))
    print("Số giá trị khớp category_tree:", len(valid_values))
    print("Số giá trị chưa khớp category_tree:", len(invalid_values))
    print("Một số giá trị chưa khớp:", sorted(invalid_values)[:20])

    print("\nĐã hoàn thành kiểm tra độ bao phủ dữ liệu.")


if __name__ == "__main__":
    main()

