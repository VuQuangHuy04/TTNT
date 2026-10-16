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

    train = pd.read_csv(train_file, usecols=["visitorid", "event", "itemid"])
    test = pd.read_csv(test_file, usecols=["visitorid", "event", "itemid"])

    train_users = set(train["visitorid"].unique())
    test_users = set(test["visitorid"].unique())

    train_items = set(train["itemid"].unique())
    test_items = set(test["itemid"].unique())

    known_test_users = test_users & train_users
    new_test_users = test_users - train_users

    known_test_items = test_items & train_items
    new_test_items = test_items - train_items

    print("===== 1. PHÂN BỐ SỰ KIỆN =====")
    for name, df in [("TRAIN", train), ("TEST", test)]:
        print(f"\n{name}:")
        counts = df["event"].value_counts()
        for event in ["view", "addtocart", "transaction"]:
            count = int(counts.get(event, 0))
            print(f"{event}: {count:,} ({count / len(df):.2%})")

    print("\n===== 2. NGƯỜI DÙNG TRONG TEST =====")
    print("Tổng người dùng:", len(test_users))
    print("Đã xuất hiện trong Train (Warm):", len(known_test_users))
    print("Chưa xuất hiện trong Train (Cold):", len(new_test_users))
    print(f"Tỷ lệ người dùng mới (Cold): {len(new_test_users) / len(test_users):.2%}")

    print("\n===== 3. SẢN PHẨM TRONG TEST =====")
    print("Tổng sản phẩm:", len(test_items))
    print("Đã xuất hiện trong Train (Known):", len(known_test_items))
    print("Chưa xuất hiện trong Train (Unseen):", len(new_test_items))
    print(f"Tỷ lệ sản phẩm mới: {len(new_test_items) / len(test_items):.2%}")

    print("\n===== 4. SỰ KIỆN TEST CỦA NGƯỜI DÙNG ĐÃ CÓ TRONG TRAIN =====")
    test_known_user = test[test["visitorid"].isin(train_users)]
    print("Số sự kiện của Warm users:", len(test_known_user))
    print("Số người dùng Warm:", test_known_user["visitorid"].nunique())
    print("\n===== HOÀN TẤT =====")


if __name__ == "__main__":
    main()

