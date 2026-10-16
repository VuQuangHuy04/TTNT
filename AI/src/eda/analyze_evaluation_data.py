import sys
from pathlib import Path
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    train_file = DATA_DIR / "train_events.csv"
    test_file = DATA_DIR / "test_events.csv"

    if not train_file.exists() or not test_file.exists():
        print("Cần có train_events.csv và test_events.csv để phân tích.")
        return

    train = pd.read_csv(train_file, usecols=["visitorid", "itemid", "event"])
    test = pd.read_csv(test_file, usecols=["visitorid", "itemid", "event"])

    # Chỉ xét người dùng đã có lịch sử trong Train (Warm users)
    train_users = set(train["visitorid"].unique())
    warm_test = test[test["visitorid"].isin(train_users)].copy()

    print("===== 1. TEST CỦA NGƯỜI DÙNG ĐÃ CÓ LỊCH SỬ (WARM USERS) =====")
    print("Số sự kiện:", len(warm_test))
    print("Số người dùng:", warm_test["visitorid"].nunique())

    print("\n===== 2. PHÂN BỐ SỰ KIỆN TRONG NHÓM NÀY =====")
    print(warm_test["event"].value_counts().to_string())

    print("\n===== 3. SỐ CẶP NGƯỜI DÙNG - SẢN PHẨM =====")
    all_pairs = warm_test[["visitorid", "itemid"]].drop_duplicates()
    print("Mọi loại sự kiện:", len(all_pairs))

    purchase = warm_test[warm_test["event"] == "transaction"]
    purchase_pairs = purchase[["visitorid", "itemid"]].drop_duplicates()

    print("Cặp có giao dịch mua:", len(purchase_pairs))
    print("Người dùng có mua trong Test:", purchase["visitorid"].nunique())
    print("Sản phẩm được mua trong Test:", purchase["itemid"].nunique())

    print("\n===== 4. SẢN PHẨM ĐÃ CÓ TRONG TRAIN =====")
    train_items = set(train["itemid"].unique())
    known_purchase_pairs = purchase_pairs[purchase_pairs["itemid"].isin(train_items)]

    print("Cặp mua hàng:", len(purchase_pairs))
    print("Cặp mua hàng có sản phẩm trong Train:", len(known_purchase_pairs))
    print(
        "Người dùng có mua sản phẩm đã xuất hiện trong Train:",
        known_purchase_pairs["visitorid"].nunique(),
    )


if __name__ == "__main__":
    main()

