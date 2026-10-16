import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eda"
INPUT_FILE = RAW_DIR / "category_tree.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"KHÔNG TÌM THẤY FILE: {INPUT_FILE}")
        return

    print("Đang đọc category_tree.csv...")
    df = pd.read_csv(INPUT_FILE)

    print("\n===== 1. TỔNG QUAN =====")
    print("Kích thước:", df.shape)
    print("Các cột:", df.columns.tolist())
    print("\n5 dòng đầu:")
    print(df.head())

    print("\n===== 2. KIỂM TRA DỮ LIỆU THIẾU =====")
    print(df.isnull().sum())

    required_columns = {"categoryid", "parentid"}
    if not required_columns.issubset(df.columns):
        print("Không tìm thấy đủ hai cột categoryid và parentid.")
        return

    print("\n===== 3. KIỂM TRA ID DANH MỤC =====")
    print("Số categoryid duy nhất:", df["categoryid"].nunique())
    print("Số categoryid bị trùng:", df["categoryid"].duplicated().sum())

    print("\n===== 4. KIỂM TRA QUAN HỆ DANH MỤC CHA =====")
    category_ids = set(df["categoryid"].dropna())
    parent_ids = df["parentid"].dropna()
    parent_ids = parent_ids[~parent_ids.isin([-1, 0])]

    missing_parents = sorted(set(parent_ids) - category_ids)
    print("Số danh mục có parentid không tìm thấy:", len(missing_parents))
    print("Một số parentid không tìm thấy:", missing_parents[:20])

    self_parent = df[df["categoryid"] == df["parentid"]]
    print("Số danh mục tự làm cha của chính nó:", len(self_parent))

    root_count = df["parentid"].isin([-1, 0]).sum()
    print("Số dòng có parentid bằng -1 hoặc 0:", root_count)

    print("\n===== 5. LƯU BÁO CÁO =====")
    report_file = OUTPUT_DIR / "category_tree_check_report.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("CATEGORY TREE CHECK REPORT\n")
        f.write(f"Kích thước: {df.shape}\n")
        f.write(f"Các cột: {df.columns.tolist()}\n")
        f.write(f"\nDữ liệu thiếu:\n{df.isnull().sum()}\n")
        f.write(f"\nSố categoryid duy nhất: {df['categoryid'].nunique()}\n")
        f.write(f"Số categoryid bị trùng: {df['categoryid'].duplicated().sum()}\n")
        f.write(f"Số parentid không tìm thấy: {len(missing_parents)}\n")
        f.write(f"Số danh mục tự làm cha: {len(self_parent)}\n")

    if missing_parents:
        pd.DataFrame({"missing_parentid": missing_parents}).to_csv(
            OUTPUT_DIR / "missing_parent_categories.csv", index=False
        )

    print("Đã lưu báo cáo:", report_file)
    print("\nHoàn thành kiểm tra category_tree.csv.")


if __name__ == "__main__":
    main()

