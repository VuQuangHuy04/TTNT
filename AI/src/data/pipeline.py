"""Pipeline tiền xử lý dữ liệu hoàn chỉnh (End-to-End Data Pipeline).

Điều phối toàn bộ quy trình từ dữ liệu thô (Raw Data) đến dữ liệu chuẩn cho AI/ML:
1. Ingestion & Clean Events: làm sạch sự kiện người dùng.
2. Temporal Train/Test Split: phân chia theo mốc thời gian chống rò rỉ dữ liệu.
3. User-Item Interaction Matrix: tính điểm implicit feedback, lọc k-core, mã hóa ID liên tục,
   xuất CSR Matrix (scipy.sparse) và lưu id_mappings.pkl.
4. Test Ground Truth: chuẩn bị dữ liệu đánh giá cho Warm Users.
5. Item Feature Engineering: trích xuất metadata sản phẩm và ma trận đặc trưng từ cây danh mục.
"""

from pathlib import Path
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .clean_events import clean_events
from .split_data import split_events
from .build_interactions import build_train_interactions, build_test_ground_truth
from .build_item_features import build_item_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def run_data_pipeline(
    raw_dir=None,
    processed_dir=None,
    split_ratio=0.8,
    min_user_interactions=1,
    min_item_interactions=1,
    build_features=True,
    verbose=True,
):
    r_dir = Path(raw_dir) if raw_dir else RAW_DIR
    p_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    p_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if verbose:
        print("\n" + "#" * 70)
        print("KHỞI CHẠY PIPELINE TIỀN XỬ LÝ DỮ LIỆU BẬC CAO CHO HỆ THỐNG GỢI Ý (AI/RECSYS)")
        print("#" * 70)

    # BƯỚC 1: Làm sạch dữ liệu sự kiện
    cleaned_file = p_dir / "cleaned_events.csv"
    clean_events(
        raw_file=r_dir / "events.csv",
        output_file=cleaned_file,
        verbose=verbose,
    )

    # BƯỚC 2: Phân chia Train/Test theo thời gian
    _, _, cutoff_timestamp = split_events(
        input_file=cleaned_file,
        output_dir=p_dir,
        split_ratio=split_ratio,
        verbose=verbose,
    )

    # BƯỚC 3: Xây dựng ma trận tương tác và mã hóa ID
    interactions, mapping_payload, user_item_matrix = build_train_interactions(
        train_file=p_dir / "train_events.csv",
        output_dir=p_dir,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        verbose=verbose,
    )

    # BƯỚC 4: Xây dựng Ground Truth tập Test
    ground_truth = build_test_ground_truth(
        test_file=p_dir / "test_events.csv",
        output_dir=p_dir,
        mapping_payload=mapping_payload,
        verbose=verbose,
    )

    # BƯỚC 5: Trích xuất thuộc tính sản phẩm và cây danh mục
    if build_features:
        build_item_features(
            raw_dir=r_dir,
            output_dir=p_dir,
            cutoff_timestamp=cutoff_timestamp,
            item2idx=mapping_payload["item2idx"],
            verbose=verbose,
        )

    elapsed = time.time() - start_time
    if verbose:
        print("\n" + "=" * 70)
        print(f"PIPELINE TIỀN XỬ LÝ HOÀN TẤT TRONG {elapsed:.2f} GIÂY!")
        print("=" * 70)
        print("DANH SÁCH FILE ĐÃ SẴN SÀNG CHO BƯỚC TRAIN MODEL:")
        print(f" 1. Ma trận tương tác thưa:    {p_dir / 'train_user_item_matrix.npz'}")
        print(f" 2. Bảng ánh xạ định danh:     {p_dir / 'id_mappings.pkl'}")
        print(f" 3. Dữ liệu đánh giá Test:     {p_dir / 'test_ground_truth.pkl'}")
        print(f" 4. Bảng tương tác chi tiết:   {p_dir / 'train_interactions.csv'}")
        if build_features:
            print(f" 5. Metadata sản phẩm:         {p_dir / 'item_metadata.csv'}")
            print(f" 6. Ma trận đặc trưng sản phẩm:{p_dir / 'item_features_matrix.npz'}")
        print("=" * 70 + "\n")

    return {
        "user_item_matrix": user_item_matrix,
        "mapping_payload": mapping_payload,
        "ground_truth": ground_truth,
    }


if __name__ == "__main__":
    run_data_pipeline()
