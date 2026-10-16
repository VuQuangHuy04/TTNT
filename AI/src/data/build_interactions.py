"""Module xử lý ma trận tương tác và mã hóa ID cho AI / Recommender System.

Các bước cốt lõi:
1. Tổng hợp phản hồi ngầm (Implicit Feedback):
   - Đếm số lần view, addtocart, transaction của mỗi cặp (visitorid, itemid).
   - Gán trọng số tương tác: view=1, addtocart=3, transaction=5.
   - Chuẩn hóa phi tuyến (log1p) để giảm ảnh hưởng của bot/outlier.
2. Lọc K-Core (tùy chọn):
   - Cho phép lọc bớt các user hoặc item có quá ít tương tác (giảm độ thưa từ 99.999%).
3. Mã hóa định danh liên tục (ID Index Encoding):
   - Model AI/CF (ALS, BPR, LightFM, PyTorch) yêu cầu ID liên tục [0, N-1].
   - Fit ánh xạ user2idx, item2idx TRÊN TẬP TRAIN và lưu id_mappings.pkl.
4. Xuất ma trận thưa chuẩn cho Model:
   - train_user_item_matrix.npz dạng scipy.sparse.csr_matrix (dtype float32).
5. Xây dựng bộ dữ liệu đánh giá Test Ground Truth:
   - Ánh xạ tập test theo index của train (chỉ giữ Warm Users và Known Items).
   - Lưu test_ground_truth.pkl phục vụ đo lường Precision@K, Recall@K, NDCG@K.
"""

from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
from scipy import sparse

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Trọng số phản hồi ngầm chuẩn trong E-commerce
DEFAULT_EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}


def filter_k_core(df, min_user_interactions=1, min_item_interactions=1, verbose=True):
    """Lọc dữ liệu theo ngưỡng tương tác tối thiểu của User và Item."""
    if min_user_interactions <= 1 and min_item_interactions <= 1:
        return df

    curr_df = df.copy()
    iteration = 0
    while True:
        iteration += 1
        prev_len = len(curr_df)

        if min_user_interactions > 1:
            user_counts = curr_df["visitorid"].value_counts()
            valid_users = user_counts[user_counts >= min_user_interactions].index
            curr_df = curr_df[curr_df["visitorid"].isin(valid_users)]

        if min_item_interactions > 1:
            item_counts = curr_df["itemid"].value_counts()
            valid_items = item_counts[item_counts >= min_item_interactions].index
            curr_df = curr_df[curr_df["itemid"].isin(valid_items)]

        if len(curr_df) == prev_len:
            break

    if verbose:
        print(f"K-Core Filter (min_user={min_user_interactions}, min_item={min_item_interactions}): "
              f"Sau {iteration} vòng lặp, giữ lại {len(curr_df):,}/{len(df):,} dòng tương tác.")

    return curr_df


def build_train_interactions(
    train_file=None,
    output_dir=None,
    event_weights=None,
    min_user_interactions=1,
    min_item_interactions=1,
    verbose=True,
):
    t_file = Path(train_file) if train_file else DATA_DIR / "train_events.csv"
    out_dir = Path(output_dir) if output_dir else DATA_DIR
    weights = event_weights or DEFAULT_EVENT_WEIGHTS

    if not t_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file train: {t_file}")

    if verbose:
        print("\n" + "=" * 60)
        print("BƯỚC 4: XÂY DỰNG TƯƠNG TÁC TẬP TRAIN & MÃ HÓA ID")
        print("=" * 60)

    train_df = pd.read_csv(t_file, usecols=["visitorid", "itemid", "event"])
    train_df = train_df[train_df["event"].isin(weights)].copy()

    # Tổng hợp số lần mỗi sự kiện
    interactions = (
        train_df.groupby(["visitorid", "itemid", "event"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for event in weights:
        if event not in interactions.columns:
            interactions[event] = 0

    interactions = interactions.rename(
        columns={
            "view": "view_count",
            "addtocart": "addtocart_count",
            "transaction": "transaction_count",
        }
    )

    # Tính điểm tin cậy tương tác (Implicit Feedback Score)
    interactions["interaction_score"] = (
        interactions["view_count"] * weights["view"]
        + interactions["addtocart_count"] * weights["addtocart"]
        + interactions["transaction_count"] * weights["transaction"]
    )
    interactions["log_interaction_score"] = np.log1p(interactions["interaction_score"]).astype(np.float32)

    # Lọc K-Core nếu được cấu hình
    interactions = filter_k_core(
        interactions,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        verbose=verbose,
    )

    # MÃ HÓA ĐỊNH DANH LIÊN TỤC [0, N-1] DỰA TRÊN TẬP TRAIN
    unique_users = np.sort(interactions["visitorid"].unique())
    unique_items = np.sort(interactions["itemid"].unique())

    user2idx = {int(uid): int(i) for i, uid in enumerate(unique_users)}
    idx2user = {int(i): int(uid) for i, uid in enumerate(unique_users)}
    item2idx = {int(iid): int(i) for i, iid in enumerate(unique_items)}
    idx2item = {int(i): int(iid) for i, iid in enumerate(unique_items)}

    interactions["user_idx"] = interactions["visitorid"].map(user2idx).astype(np.int32)
    interactions["item_idx"] = interactions["itemid"].map(item2idx).astype(np.int32)

    # Sắp xếp các cột đầu ra
    cols = [
        "visitorid", "itemid", "user_idx", "item_idx",
        "view_count", "addtocart_count", "transaction_count",
        "interaction_score", "log_interaction_score"
    ]
    interactions = interactions[cols]

    # XÂY DỰNG MA TRẬN THƯA CSR MATRIX (np.float32) CHO MODEL TRAIN
    n_users = len(user2idx)
    n_items = len(item2idx)
    user_item_matrix = sparse.csr_matrix(
        (
            interactions["log_interaction_score"].values,
            (interactions["user_idx"].values, interactions["item_idx"].values),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    # Lưu ma trận thưa
    matrix_file = out_dir / "train_user_item_matrix.npz"
    sparse.save_npz(matrix_file, user_item_matrix)

    # Lưu bảng mapping ID
    mapping_payload = {
        "user2idx": user2idx,
        "idx2user": idx2user,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "num_users": n_users,
        "num_items": n_items,
    }
    mapping_file = out_dir / "id_mappings.pkl"
    with open(mapping_file, "wb") as f:
        pickle.dump(mapping_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Lưu bảng tương tác csv
    csv_file = out_dir / "train_interactions.csv"
    interactions.to_csv(csv_file, index=False)

    if verbose:
        sparsity = 1.0 - (len(interactions) / (n_users * n_items))
        print(f"Tổng số tương tác (Non-zero entries): {len(interactions):,}")
        print(f"Số người dùng duy nhất (N_users): {n_users:,}")
        print(f"Số sản phẩm duy nhất (N_items): {n_items:,}")
        print(f"Kích thước ma trận: {n_users:,} x {n_items:,} (Độ thưa Sparsity: {sparsity:.4%})")
        print(f"Kiểu dữ liệu ma trận: {user_item_matrix.dtype} (CSR Matrix)")
        print(f"Đã lưu ma trận thưa tại: {matrix_file}")
        print(f"Đã lưu bảng ánh xạ ID tại: {mapping_file}")
        print(f"Đã lưu bảng tương tác tại: {csv_file}")

    return interactions, mapping_payload, user_item_matrix


def build_test_ground_truth(test_file=None, output_dir=None, mapping_payload=None, verbose=True):
    t_file = Path(test_file) if test_file else DATA_DIR / "test_events.csv"
    out_dir = Path(output_dir) if output_dir else DATA_DIR

    if not t_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file test: {t_file}")

    if verbose:
        print("\n" + "=" * 60)
        print("BƯỚC 5: XÂY DỰNG GROUND TRUTH ĐÁNH GIÁ (TẬP TEST)")
        print("=" * 60)

    user2idx = mapping_payload["user2idx"]
    item2idx = mapping_payload["item2idx"]

    test_df = pd.read_csv(t_file, usecols=["visitorid", "itemid", "event"])

    # Chỉ giữ lại các tương tác của Warm Users và Known Items
    test_df["is_warm_user"] = test_df["visitorid"].isin(user2idx)
    test_df["is_known_item"] = test_df["itemid"].isin(item2idx)

    warm_test = test_df[test_df["is_warm_user"] & test_df["is_known_item"]].copy()
    warm_test["user_idx"] = warm_test["visitorid"].map(user2idx).astype(np.int32)
    warm_test["item_idx"] = warm_test["itemid"].map(item2idx).astype(np.int32)

    # Tạo từ điển Ground Truth theo user_idx phục vụ vectorized evaluation
    # user_idx -> {'all': set(item_idx), 'purchases': set(item_idx)}
    ground_truth = {}
    for (u_idx, i_idx, evt) in warm_test[["user_idx", "item_idx", "event"]].itertuples(index=False):
        if u_idx not in ground_truth:
            ground_truth[u_idx] = {"all": set(), "purchases": set()}
        ground_truth[u_idx]["all"].add(i_idx)
        if evt == "transaction":
            ground_truth[u_idx]["purchases"].add(i_idx)

    # Lưu file nhị phân pkl để model đánh giá siêu tốc
    gt_pkl_file = out_dir / "test_ground_truth.pkl"
    with open(gt_pkl_file, "wb") as f:
        pickle.dump(ground_truth, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Lưu file CSV thống kê để người dùng dễ đọc
    warm_summary = (
        warm_test.groupby(["visitorid", "itemid", "user_idx", "item_idx"])
        .agg(
            events=("event", lambda x: list(x.unique())),
            has_transaction=("event", lambda x: int("transaction" in set(x))),
        )
        .reset_index()
    )
    gt_csv_file = out_dir / "test_ground_truth_warm.csv"
    warm_summary.to_csv(gt_csv_file, index=False)

    if verbose:
        num_warm_users = len(ground_truth)
        users_with_purchase = sum(1 for v in ground_truth.values() if len(v["purchases"]) > 0)
        print(f"Tổng số tương tác test của Warm Users & Known Items: {len(warm_test):,}")
        print(f"Số Warm Users có tương tác trong Test: {num_warm_users:,}")
        print(f"Số Warm Users có giao dịch mua (transaction) trong Test: {users_with_purchase:,}")
        print(f"Đã lưu Ground Truth nhị phân tại: {gt_pkl_file}")
        print(f"Đã lưu bảng Ground Truth CSV tại: {gt_csv_file}")

    return ground_truth


def main():
    _, mapping, _ = build_train_interactions()
    build_test_ground_truth(mapping_payload=mapping)


if __name__ == "__main__":
    main()
