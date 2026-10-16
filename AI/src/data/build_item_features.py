"""Module tiền xử lý đặc trưng sản phẩm (Item Features & Category Hierarchy).

Chức năng:
- Đọc và phân tích cây danh mục (category_tree.csv): xác định danh mục cha (parent) và danh mục gốc (root).
- Quét nhanh theo khối (chunks) hai file thuộc tính (item_properties_part1 & part2):
  + Trích xuất danh mục (categoryid) và trạng thái tồn kho (available: 0 hoặc 1).
  + Chống Feature Leakage: Chỉ lấy các snapshot thuộc tính có timestamp <= train_cutoff.
  + Lấy giá trị cập nhật mới nhất cho từng itemid.
- Kết hợp tạo bảng metadata sản phẩm: item_metadata.csv.
- Hỗ trợ xây dựng ma trận đặc trưng thưa (Item Feature Sparse Matrix npz) phục vụ
  cho các mô hình Hybrid (như LightFM, Two-Tower DSSM, Content-Based).
"""

from pathlib import Path
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
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_category_hierarchy(category_file=None):
    cat_path = Path(category_file) if category_file else RAW_DIR / "category_tree.csv"
    if not cat_path.exists():
        return {}, {}

    df_cat = pd.read_csv(cat_path)
    parent_map = {}
    for _, row in df_cat.dropna(subset=["categoryid"]).iterrows():
        cat_id = int(row["categoryid"])
        p_id = int(row["parentid"]) if pd.notna(row["parentid"]) and int(row["parentid"]) >= 0 else None
        parent_map[cat_id] = p_id

    # Tìm root category cho mỗi category
    root_map = {}
    for cat_id in parent_map:
        curr = cat_id
        visited = set()
        while parent_map.get(curr) is not None and curr not in visited:
            visited.add(curr)
            curr = parent_map[curr]
        root_map[cat_id] = curr

    return parent_map, root_map


def extract_item_properties(raw_dir=None, cutoff_timestamp=None, verbose=True):
    r_dir = Path(raw_dir) if raw_dir else RAW_DIR
    files = [r_dir / "item_properties_part1.csv", r_dir / "item_properties_part2.csv"]

    records = []
    for f in files:
        if not f.exists():
            if verbose:
                print(f"Cảnh báo: Không tìm thấy file {f.name}")
            continue

        if verbose:
            print(f"Đang quét thuộc tính từ {f.name}...")

        for chunk in pd.read_csv(
            f,
            chunksize=1_000_000,
            usecols=["timestamp", "itemid", "property", "value"],
            dtype={"timestamp": "int64", "itemid": "int64", "property": "string", "value": "string"},
        ):
            subset = chunk[chunk["property"].isin(["categoryid", "available"])].copy()
            if cutoff_timestamp is not None:
                subset = subset[subset["timestamp"] <= cutoff_timestamp]

            if not subset.empty:
                records.append(subset)

    if not records:
        return pd.DataFrame(columns=["itemid", "categoryid", "available"])

    df_props = pd.concat(records, ignore_index=True)
    # Sắp xếp theo timestamp và giữ snapshot mới nhất của mỗi cặp (itemid, property)
    df_props = df_props.sort_values("timestamp").drop_duplicates(subset=["itemid", "property"], keep="last")

    # Pivot thành bảng phẳng
    pivoted = df_props.pivot(index="itemid", columns="property", values="value").reset_index()

    # Chuẩn hóa kiểu dữ liệu
    if "available" in pivoted.columns:
        pivoted["available"] = pd.to_numeric(pivoted["available"], errors="coerce").fillna(0).astype(int)
    else:
        pivoted["available"] = 1

    if "categoryid" in pivoted.columns:
        pivoted["categoryid"] = pd.to_numeric(pivoted["categoryid"], errors="coerce")
    else:
        pivoted["categoryid"] = np.nan

    return pivoted


def build_item_features(
    raw_dir=None,
    output_dir=None,
    cutoff_timestamp=None,
    item2idx=None,
    verbose=True,
):
    r_dir = Path(raw_dir) if raw_dir else RAW_DIR
    out_dir = Path(output_dir) if output_dir else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 60)
        print("BƯỚC 3: XỬ LÝ ĐẶC TRƯNG SẢN PHẨM & CÂY DANH MỤC")
        print("=" * 60)

    # 1. Phân cấp cây danh mục
    parent_map, root_map = build_category_hierarchy(r_dir / "category_tree.csv")

    # 2. Trích xuất snapshot thuộc tính sản phẩm
    df_items = extract_item_properties(r_dir, cutoff_timestamp=cutoff_timestamp, verbose=verbose)

    if not df_items.empty:
        df_items["parent_categoryid"] = df_items["categoryid"].map(parent_map)
        df_items["root_categoryid"] = df_items["categoryid"].map(root_map)

        meta_file = out_dir / "item_metadata.csv"
        df_items.to_csv(meta_file, index=False)
        if verbose:
            print(f"Tổng số sản phẩm có metadata: {len(df_items):,}")
            print(f"Số sản phẩm có categoryid hợp lệ: {df_items['categoryid'].notna().sum():,}")
            print(f"Đã lưu item metadata tại: {meta_file}")

        # 3. Nếu có item2idx (tập train items), xây dựng ma trận đặc trưng thưa (CSR)
        if item2idx is not None:
            build_sparse_item_feature_matrix(df_items, item2idx, out_dir / "item_features_matrix.npz", verbose=verbose)

    return df_items


def build_sparse_item_feature_matrix(df_items, item2idx, output_file, verbose=True):
    """Tạo ma trận thưa One-Hot Category cho các items trong tập Train."""
    n_items = len(item2idx)
    # Lọc items có trong tập train
    train_items_df = df_items[df_items["itemid"].isin(item2idx)].copy()
    train_items_df["item_idx"] = train_items_df["itemid"].map(item2idx)

    # Lấy các category duy nhất
    valid_cats = train_items_df["categoryid"].dropna().unique()
    cat2idx = {cat: i for i, cat in enumerate(valid_cats)}

    rows = []
    cols = []
    for _, row in train_items_df.iterrows():
        idx = int(row["item_idx"])
        cat = row["categoryid"]
        if pd.notna(cat) and cat in cat2idx:
            rows.append(idx)
            cols.append(cat2idx[cat])

    if rows:
        data = np.ones(len(rows), dtype=np.float32)
        n_features = len(cat2idx)
        feature_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_items, n_features), dtype=np.float32)
        sparse.save_npz(output_file, feature_matrix)
        if verbose:
            print(f"Đã tạo ma trận đặc trưng sản phẩm: shape {feature_matrix.shape}, lưu tại {output_file}")


if __name__ == "__main__":
    build_item_features()
