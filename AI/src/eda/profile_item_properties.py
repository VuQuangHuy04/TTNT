import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eda"

FILES = [
    RAW_DIR / "item_properties_part1.csv",
    RAW_DIR / "item_properties_part2.csv",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    all_items = set()
    all_properties = set()
    property_counts = {}
    category_values = set()
    available_values = {}

    print("Bắt đầu phân tích item_properties...")

    for file_path in FILES:
        if not file_path.exists():
            print(f"Không tìm thấy {file_path}")
            continue

        print(f"\nĐang đọc: {file_path.name}")

        for chunk in pd.read_csv(
            file_path,
            dtype={
                "timestamp": "int64",
                "itemid": "int64",
                "property": "string",
                "value": "string",
            },
            chunksize=500_000,
        ):
            total_rows += len(chunk)

            all_items.update(chunk["itemid"].unique())
            all_properties.update(chunk["property"].dropna().unique())

            counts = chunk["property"].value_counts()
            for prop, count in counts.items():
                property_counts[prop] = (
                    property_counts.get(prop, 0) + int(count)
                )

            category_rows = chunk[chunk["property"] == "categoryid"]
            category_values.update(category_rows["value"].dropna().unique())

            available_rows = chunk[chunk["property"] == "available"]
            if not available_rows.empty:
                counts_available = available_rows["value"].value_counts()
                for value, count in counts_available.items():
                    available_values[value] = (
                        available_values.get(value, 0) + int(count)
                    )

    print("\n===== TỔNG QUAN =====")
    print("Tổng số dòng:", total_rows)
    print("Số sản phẩm duy nhất:", len(all_items))
    print("Số loại property:", len(all_properties))
    print("Số giá trị categoryid khác nhau:", len(category_values))

    print("\n===== 20 PROPERTY PHỔ BIẾN NHẤT =====")
    property_series = pd.Series(property_counts).sort_values(ascending=False)
    print(property_series.head(20))

    print("\n===== GIÁ TRỊ PROPERTY AVAILABLE =====")
    print(pd.Series(available_values).sort_values(ascending=False))

    property_series.rename_axis("property").reset_index(
        name="count"
    ).to_csv(
        OUTPUT_DIR / "item_property_profile.csv",
        index=False,
    )

    pd.DataFrame({
        "itemid": sorted(all_items)
    }).to_csv(
        OUTPUT_DIR / "item_property_item_ids.csv",
        index=False,
    )

    print("\nĐã lưu:")
    print(OUTPUT_DIR / "item_property_profile.csv")
    print(OUTPUT_DIR / "item_property_item_ids.csv")


if __name__ == "__main__":
    main()

