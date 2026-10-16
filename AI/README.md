# RetailRocket Recommender System (AI/RecSys) - Data Preprocessing Pipeline

Tài liệu kỹ thuật toàn diện về quy trình xử lý dữ liệu từ dữ liệu thô (Raw Data) đến dữ liệu sẵn sàng cho mô hình huấn luyện AI (Model-Ready Data), tuân thủ tiêu chuẩn MLOps và Recommender Systems.

---

## MỤC LỤC
1. [Tổng quan dự án & Tập dữ liệu](#1-tổng-quan-dự-án--tập-dữ-liệu)
2. [Đánh giá hiện trạng: Đã hoàn thành gì & Điểm cần bổ sung](#2-đánh-giá-hiện-trạng-đã-hoàn-thành-gì--điểm-cần-bổ-sung)
3. [Quy trình 5 bước từ Raw Data đến Model-Ready](#3-quy-trình-5-bước-từ-raw-data-đến-model-ready)
4. [Các chỉnh sửa, dọn dẹp và tái cấu trúc mã nguồn](#4-các-chỉnh-sửa-dọn-dẹp-và-tái-cấu-trúc-mã-nguồn)
5. [Cấu trúc thư mục dự án chuẩn hóa](#5-cấu-trúc-thư-mục-dự-án-chuẩn-hóa)
6. [Đặc tả các tệp dữ liệu đầu ra cho Model Training](#6-đặc-tả-các-tệp-dữ-liệu-đầu-ra-cho-model-training)
7. [Hướng dẫn sử dụng & Tham số dòng lệnh](#7-hướng-dẫn-sử-dụng--tham-số-dòng-lệnh)
8. [Hướng dẫn bước tiếp theo: Nạp dữ liệu vào Model AI](#8-hướng-dẫn-bước-tiếp-theo-nạp-dữ-liệu-vào-model-ai)

---

## 1. Tổng quan dự án & Tập dữ liệu
Dự án sử dụng tập dữ liệu **RetailRocket E-commerce Dataset** gồm 3 nguồn dữ liệu thô (`data/raw/`):
- `events.csv` (~94 MB, 2.75 triệu dòng): Ghi nhận hành vi của người dùng gồm `view`, `addtocart`, `transaction`.
- `category_tree.csv` (~14 KB, 1,669 dòng): Quan hệ phân cấp danh mục hàng hóa (`categoryid`, `parentid`).
- `item_properties_part1.csv` & `part2.csv` (~892 MB, ~20 triệu dòng): Các thuộc tính biến đổi theo thời gian của sản phẩm (`categoryid`, `available`, hash giá trị thuộc tính).

Mục tiêu kỹ thuật: Xây dựng pipeline tự động biến đổi dữ liệu thô thành các ma trận thưa và cấu trúc nhị phân chuẩn hóa, sẵn sàng huấn luyện các mô hình lọc cộng tác (Collaborative Filtering: ALS, BPR), mô hình lai (Hybrid: LightFM) hoặc mạng nơ-ron (Two-Tower, NeuMF).
---

## 2. Đánh giá hiện trạng: Đã hoàn thành gì & Điểm cần bổ sung

### A. Những bước code ban đầu đã hoàn thành tốt:
1. **Khám phá và phân tích dữ liệu (EDA & Data Profiling)**:
   - Thống kê phân bố sự kiện (`view`: 96.68%, `addtocart`: 2.50%, `transaction`: 0.81%).
   - Phát hiện tính chất cốt lõi: dữ liệu cực kỳ thưa (độ thưa Sparsity > 99.999%), hơn 90% người dùng chỉ có đúng 1 tương tác view.
   - Kiểm tra cây danh mục (`category_tree.csv`) và tính toàn vẹn của thuộc tính sản phẩm (`item_properties`).
2. **Làm sạch sự kiện (Data Cleaning)**:
   - Khử trùng lặp hoàn toàn, loại bỏ dòng thiếu ID cốt lõi, ép kiểu số nguyên dương cho `visitorid` và `itemid`.
   - Lọc 3 hành vi hợp lệ (`view`, `addtocart`, `transaction`) và chuẩn hóa thời gian UTC.
3. **Phân chia theo thời gian (Temporal Split 80/20)**:
   - Chia train/test theo mốc thời gian (Quantile 0.8) thay vì chia ngẫu nhiên. Đây là điểm **hoàn toàn chính xác theo chuẩn AI/RecSys** nhằm chống hiện tượng **Rò rỉ dữ liệu theo thời gian (Data Leakage)**.
4. **Tính điểm phản hồi ngầm cơ bản (Implicit Scoring)**:
   - Gán trọng số tương tác: view = 1, addtocart = 3, transaction = 5 kèm $\log(1 + x)$.

### B. Những điểm còn thiếu trước đó (Nguyên nhân chưa thể train model):
- ❌ **Chưa có mã hóa ID liên tục (Contiguous 0-indexed ID Mapping)**: Mô hình AI/RecSys yêu cầu ID là số nguyên liên tục `[0, N-1]`. Giữ ID gốc 64-bit rời rạc sẽ làm hỏng kích thước ma trận embedding/CSR.
- ❌ **Chưa có Ma trận thưa đúng kiểu dữ liệu (`scipy.sparse.csr_matrix` dtype `float32`)**: Các thư viện như `implicit`, `scipy` yêu cầu đầu vào trực tiếp là CSR matrix, không thể truyền DataFrame CSV phẳng.
- ❌ **Chưa tiền xử lý thuộc tính sản phẩm**: Gần 900 MB `item_properties` và `category_tree` mới chỉ dừng ở script đọc kiểm tra, chưa trích xuất thành metadata và ma trận đặc trưng cho các mô hình Hybrid/Content-Based.
- ❌ **Chưa tối ưu Ground Truth cho Test**: Dữ liệu test lưu danh sách dạng string gây nghẽn tốc độ khi tính các metric `Precision@K`, `Recall@K`, `NDCG@K`.
- ❌ **Cấu trúc mã nguồn bị phân mảnh**: Rải rác hơn 10 file nhỏ trong nhiều thư mục con, có thư mục rỗng và file cũ lỗi thời gây nhầm lẫn.

---

## 3. Quy trình 5 bước từ Raw Data đến Model-Ready

```
[Raw Data: events.csv, item_properties, category_tree]
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ BƯỚC 1: Ingestion & Làm sạch dữ liệu (Data Cleaning)   │
│ - Lọc bản ghi trùng lặp và thiếu ID cốt lõi            │
│ - Chuẩn hóa kiểu ID (int64 >= 0) và event hợp lệ       │
│ - Chuyển Unix ms sang Datetime UTC                     │
│ ➔ cleaned_events.csv                                   │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ BƯỚC 2: Phân chia theo thời gian (Temporal Split)      │
│ - Cắt mốc thời gian 80% (Quantile 0.8)                 │
│ - Chống Data Leakage (Quá khứ dự đoán Tương lai)       │
│ ➔ train_events.csv, test_events.csv                    │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ BƯỚC 3: Tính điểm tương tác & Lọc K-Core (Sparsity)    │
│ - Điểm Implicit: score = 1*view + 3*cart + 5*buy       │
│ - Chuẩn hóa phi tuyến: log1p(score) giảm nhiễu bot     │
│ - Hỗ trợ lọc K-Core giảm độ thưa (sparsity)            │
│ ➔ train_interactions.csv                               │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ BƯỚC 4: Mã hóa ID liên tục & Tạo Ma trận thưa CSR      │
│ - Fit trên tập Train: ánh xạ [0..N_u-1] và [0..N_i-1]  │
│ - Xuất scipy.sparse.csr_matrix (dtype float32)         │
│ - Trích xuất Ground Truth tập Test cho Warm Users      │
│ ➔ train_user_item_matrix.npz, id_mappings.pkl          │
│ ➔ test_ground_truth.pkl                                │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ BƯỚC 5: Xử lý đặc trưng sản phẩm & Cây danh mục       │
│ - Phân tích category_tree (parentid, root_categoryid)  │
│ - Quét chunk item_properties với timestamp <= cutoff   │
│ - Tạo metadata & ma trận One-Hot Category CSR          │
│ ➔ item_metadata.csv, item_features_matrix.npz          │
└────────────────────────────────────────────────────────┘
```

### Chi tiết kỹ thuật từng bước:
1. **Bước 1 (Làm sạch)**: Lọc các ID âm, loại bỏ sự kiện ngoài `view`, `addtocart`, `transaction`.
2. **Bước 2 (Chia Train/Test)**: Mốc cắt thời gian (Cutoff) đảm bảo mọi sự kiện xảy ra trước mốc dùng để train, mọi sự kiện sau mốc dùng để đánh giá. Không bao giờ chia ngẫu nhiên (random split) đối với dữ liệu tương tác theo thời gian.
3. **Bước 3 (Tính điểm & K-Core)**: Điểm tương tác ngầm phản ánh độ tin cậy (confidence). Hàm $\log(1 + x)$ kìm hãm các tương tác quá lớn từ bot hoặc đại lý mua buôn. Lọc K-Core (tùy chọn) loại bỏ các user/item có quá ít tương tác để mô hình hội tụ tốt hơn.
4. **Bước 4 (Mã hóa ID & CSR Matrix)**: Ánh xạ 1.12 triệu `visitorid` và 212.8 nghìn `itemid` thành chỉ số nguyên liên tục `user_idx` $[0, N_u-1]$ và `item_idx` $[0, N_i-1]$. Lưu từ điển mapping để phục vụ suy luận thực tế (inference).
5. **Bước 5 (Đặc trưng sản phẩm)**: Chống Feature Leakage bằng cách chỉ lấy snapshot thuộc tính có `timestamp <= cutoff`. Kết hợp với cây danh mục để xác định danh mục cha (`parent_categoryid`) và danh mục gốc (`root_categoryid`).

---

## 4. Các chỉnh sửa, dọn dẹp và tái cấu trúc mã nguồn

### A. Tệp và thư mục đã xóa / dọn dẹp:
- ❌ **Xóa** thư mục rác tạo nhầm: `src/results/`.
- ❌ **Xóa** `data/processed/user_item_interactions.csv`: Tệp cũ tạo trước khi chia train/test (dễ gây nhầm lẫn và rò rỉ dữ liệu).
- ❌ **Xóa** các script kiểm tra tạm thời: `check_cleanevents.py`, `check_item_properties.py`, `analyze_item_properties.py`.
- ❌ **Thay thế** `train_user_item_interactions.csv` bằng `train_interactions.csv` (chuẩn hóa có thêm `user_idx`, `item_idx`).

### B. Tách biệt kiến trúc Production và EDA:
- **Tập trung hóa Pipeline** trong `src/data/`: Chỉ chứa các module nghiệp vụ tiền xử lý sản xuất, chạy tự động, không chứa script in console tạm thời.
- **Tập trung hóa Phân tích** trong `src/eda/`: Lưu trữ toàn bộ mã nguồn phân tích dữ liệu, khám phá và kiểm tra chất lượng dữ liệu để thuận tiện xem lại.
- **Bổ sung CLI điều phối**: Tạo `run_pipeline.py` tại thư mục gốc giúp chạy toàn bộ luồng bằng 1 dòng lệnh.

---

## 5. Cấu trúc thư mục dự án chuẩn hóa

```
D:\AI\
├── data\
│   ├── raw\                               # Dữ liệu gốc (Bất biến)
│   │   ├── events.csv                     # Sự kiện tương tác
│   │   ├── category_tree.csv              # Cây phả hệ danh mục
│   │   ├── item_properties_part1.csv      # Thuộc tính sản phẩm (phần 1)
│   │   └── item_properties_part2.csv      # Thuộc tính sản phẩm (phần 2)
│   └── processed\                         # Dữ liệu đã làm sạch & tiền xử lý chuẩn AI
│       ├── cleaned_events.csv             # Sự kiện đã làm sạch & sắp xếp thời gian
│       ├── train_events.csv               # Sự kiện tập train (80% đầu)
│       ├── test_events.csv                # Sự kiện tập test (20% sau)
│       ├── train_interactions.csv         # Bảng tương tác chi tiết kèm user_idx, item_idx
│       ├── train_user_item_matrix.npz     # Ma trận thưa CSR Matrix (n_users x n_items) dtype float32
│       ├── id_mappings.pkl                # Từ điển ánh xạ: user2idx, idx2user, item2idx, idx2item
│       ├── item_metadata.csv              # Bảng metadata sản phẩm (category, parent, root, available)
│       ├── item_features_matrix.npz       # Ma trận đặc trưng danh mục thưa CSR Matrix (One-hot)
│       ├── test_ground_truth.pkl          # Ground truth nhị phân phục vụ tính metric siêu tốc
│       └── test_ground_truth_warm.csv     # Bảng tương tác chi tiết của Warm Users ở tập Test
├── src\
│   ├── data\                              # Các module tiền xử lý dữ liệu chính (Production)
│   │   ├── clean_events.py                # Bước 1: Làm sạch dữ liệu sự kiện
│   │   ├── split_data.py                  # Bước 2: Chia Train/Test theo thời gian
│   │   ├── build_interactions.py          # Bước 3 & 4: Tính implicit score, mã hóa ID, tạo CSR Matrix
│   │   ├── build_item_features.py         # Bước 5: Phân cấp danh mục & tạo ma trận đặc trưng
│   │   └── pipeline.py                    # Orchestrator điều phối toàn bộ quy trình
│   └── eda\                               # Các kịch bản phân tích khám phá dữ liệu (EDA)
│       ├── eda_events.py                  # Phân bố sự kiện và hành vi người dùng
│       ├── check_category_tree.py         # Kiểm tra tính toàn vẹn cây danh mục
│       ├── profile_item_properties.py     # Quét và thống kê các loại thuộc tính sản phẩm
│       ├── check_data_overlap.py          # Đối chiếu độ phủ sản phẩm giữa events và properties
│       ├── analyze_split.py               # Phân tích tỷ lệ người dùng mới / sản phẩm mới ở test
│       └── analyze_evaluation_data.py     # Thống kê nhóm Warm Users và giao dịch mua ở test
├── results\
│   └── eda\                               # Kết quả bảng biểu và báo cáo EDA
├── run_pipeline.py                        # Script thực thi 1-Click toàn bộ pipeline từ dòng lệnh
├── requirements.txt                       # Danh mục thư viện Python phụ thuộc
└── README.md                              # Tài liệu kỹ thuật dự án
```

---

## 6. Đặc tả các tệp dữ liệu đầu ra cho Model Training

Toàn bộ các tệp tại `data/processed/` đều đã được kiểm tra tính toàn vẹn và chuẩn hóa kiểu dữ liệu:

| Tên Tệp | Định dạng | Kích thước / Shape | Kiểu dữ liệu | Mô tả & Cách sử dụng cho Mô hình AI |
| :--- | :--- | :--- | :--- | :--- |
| **`train_user_item_matrix.npz`** | Scipy CSR Matrix | `(1,122,936, 212,853)` | `np.float32` | **Đầu vào chính** cho các mô hình Collaborative Filtering (Implicit ALS, BPR, Logistic Matrix Factorization). Chứa điểm tin cậy tương tác $\log(1 + score)$. |
| **`id_mappings.pkl`** | Pickle Dict | 1.12M users, 212.8K items | Python Dict | Chứa `user2idx`, `idx2user`, `item2idx`, `idx2item`. Dùng để dịch ngược chỉ số ma trận về ID thực tế khi gợi ý cho người dùng. |
| **`test_ground_truth.pkl`** | Pickle Dict | 23,450 Warm Users | Python Dict | Cấu trúc `{user_idx: {'all': set(item_idx), 'purchases': set(item_idx)}}`. Phục vụ tính toán song song các metric `Precision@K`, `Recall@K`, `NDCG@K`. |
| **`item_features_matrix.npz`** | Scipy CSR Matrix | `(212,853, 1,080)` | `np.float32` | Ma trận đặc trưng One-hot danh mục của các items trong Train. Đầu vào cho các mô hình Hybrid (LightFM) hoặc Two-Tower DSSM. |
| **`item_metadata.csv`** | CSV Tabular | 412,882 sản phẩm | Tabular | Chứa `itemid`, `categoryid`, `parent_categoryid`, `root_categoryid`, `available`. Dùng giải quyết bài toán Cold-Start cho sản phẩm mới. |
| **`train_interactions.csv`** | CSV Tabular | 1,711,719 tương tác | Tabular | Bảng tương tác chi tiết kèm `user_idx` và `item_idx`, sẵn sàng nạp vào PyTorch Dataset / DataLoader cho các mô hình Deep Learning. |
| **`test_ground_truth_warm.csv`**| CSV Tabular | 71,212 tương tác | Tabular | Chi tiết tương tác của Warm Users trong Test để người dùng dễ đọc và kiểm tra thủ công. |

---

## 7. Hướng dẫn sử dụng & Tham số dòng lệnh

Chạy toàn bộ pipeline từ thư mục gốc dự án:

```bash
# 1. Chạy mặc định toàn bộ quy trình tiền xử lý:
python run_pipeline.py
```

### Các tùy chọn dòng lệnh nâng cao:

1. **Lọc K-Core (Khuyến nghị để huấn luyện nhanh và giảm độ thưa)**:
   Mặc định ma trận có độ thưa 99.999% do hàng triệu người chỉ click 1 lần. Bạn có thể lọc người dùng có tối thiểu 3 tương tác và sản phẩm có tối thiểu 2 tương tác:
   ```bash
   python run_pipeline.py --min-user-interactions 3 --min-item-interactions 2
   ```

2. **Thay đổi tỷ lệ chia Train/Test**:
   ```bash
   python run_pipeline.py --split-ratio 0.85
   ```

3. **Bỏ qua bước trích xuất item features (nếu chỉ huấn luyện mô hình Collaborative Filtering cơ bản)**:
   ```bash
   python run_pipeline.py --skip-features
   ```

---

## 8. Hướng dẫn bước tiếp theo: Nạp dữ liệu vào Model AI

Dưới đây là mã nguồn mẫu minh họa cách nạp dữ liệu từ các file đã tiền xử lý để huấn luyện mô hình **Implicit ALS (Alternating Least Squares)** và đánh giá chất lượng gợi ý:

```python
import pickle
from scipy import sparse
import implicit
import numpy as np

# 1. Nạp ma trận tương tác User-Item
train_matrix = sparse.load_npz("data/processed/train_user_item_matrix.npz")
print(f"Ma trận huấn luyện: {train_matrix.shape}, dtype: {train_matrix.dtype}")

# 2. Nạp bảng ánh xạ ID
with open("data/processed/id_mappings.pkl", "rb") as f:
    mappings = pickle.load(f)
user2idx = mappings["user2idx"]
idx2item = mappings["idx2item"]

# 3. Nạp Ground Truth đánh giá tập Test
with open("data/processed/test_ground_truth.pkl", "rb") as f:
    ground_truth = pickle.load(f)

# 4. Khởi tạo và huấn luyện mô hình ALS
# Alpha là hệ số phóng đại độ tin cậy trong phản hồi ngầm
model = implicit.als.AlternatingLeastSquares(
    factors=64,
    regularization=0.05,
    iterations=20,
    random_state=42
)
model.fit(train_matrix * 40.0)

# 5. Gợi ý Top 10 sản phẩm cho một người dùng bất kỳ
sample_user_idx = list(ground_truth.keys())[0]
recommendations, scores = model.recommend(
    userid=sample_user_idx,
    user_items=train_matrix[sample_user_idx],
    N=10
)

original_user_id = mappings["idx2user"][sample_user_idx]
recommended_item_ids = [idx2item[i] for i in recommendations]
print(f"Gợi ý Top 10 cho người dùng {original_user_id}: {recommended_item_ids}")
```

# 2. Nạp bảng ánh xạ ID
with open("data/processed/id_mappings.pkl", "rb") as f:
    mappings = pickle.load(f)
user2idx = mappings["user2idx"]
idx2item = mappings["idx2item"]

# 3. Nạp Ground Truth đánh giá tập Test
with open("data/processed/test_ground_truth.pkl", "rb") as f:
    ground_truth = pickle.load(f)

# 4. Khởi tạo và huấn luyện mô hình ALS
# Alpha là hệ số phóng đại độ tin cậy trong phản hồi ngầm
model = implicit.als.AlternatingLeastSquares(
    factors=64,
    regularization=0.05,
    iterations=20,
    random_state=42
)
model.fit(train_matrix * 40.0)

# 5. Gợi ý Top 10 sản phẩm cho một người dùng bất kỳ
sample_user_idx = list(ground_truth.keys())[0]
recommendations, scores = model.recommend(
    userid=sample_user_idx,
    user_items=train_matrix[sample_user_idx],
    N=10
)

original_user_id = mappings["idx2user"][sample_user_idx]
recommended_item_ids = [idx2item[i] for i in recommendations]
print(f"Gợi ý Top 10 cho người dùng {original_user_id}: {recommended_item_ids}")
```

