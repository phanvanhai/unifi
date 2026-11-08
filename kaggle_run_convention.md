# Quy ước chạy trên Kaggle (SSHAR ESP)

Mọi mô hình trong benchmark chạy trên Kaggle theo **cùng một cell mẫu, cùng bộ biến cấu hình, cùng bộ
tham số dòng lệnh và cùng cấu trúc thư mục kết quả**. Như vậy kết quả của các mô hình đặt cạnh nhau là
so sánh được ngay. Thiết lập train và cách chia dữ liệu xem ở [SSHAR_ESP_protocol.md](SSHAR_ESP_protocol.md);
note này chỉ quy định cách gọi.

## 1. Bộ biến cấu hình (đầu cell)

| Biến | Giá trị | Ý nghĩa |
|---|---|---|
| `DATASET_DIR` | đường dẫn dataset trên Kaggle | thư mục dataset đã add vào notebook; cell tự dò thư mục chứa `room_02` |
| `MODEL` | tên ngắn của mô hình, vd. `AMCS_TCN_FC` | dùng trong tên run |
| `SPLIT` | `"user"` hoặc `"repeat"` | `user`: cross-user (**mặc định**). `repeat`: chia theo lần lặp |
| `USER_CASE` | `0` đến `4` | case chia theo người (mục 4), chỉ dùng khi `SPLIT = "user"`. Hiện chạy **case 0** |
| `PATIENCE` | số nguyên, mặc định `10` | dừng sớm sau N epoch liên tiếp val accuracy không cải thiện; `0` = tắt (chạy đủ số epoch) |
| `SEEDS` | chuỗi seed cách nhau bởi dấu cách, mặc định `"42"` | vd. `"42 3407 1024 2024 0"` khi chạy nhiều seed |
| `FILTERS` | `"none"` hoặc tên bộ lọc | tiền xử lý tuỳ mô hình; mô hình nào không có thì bỏ biến này |

Mỗi run đều: train, chọn checkpoint theo val (val chỉ dùng để chọn checkpoint và dừng sớm), rồi
báo cáo trên **test** bằng checkpoint đó, kèm kết quả val của cùng checkpoint. Không có chế độ dev.

Không đổi các thiết lập chung (batch 32, tối đa 100 epoch, Adam, LR theo bài gốc) trong cell. Nếu buộc
phải đổi (vd. hết bộ nhớ GPU phải giảm batch), truyền thêm tham số tương ứng và ghi rõ trong báo cáo.

## 2. Tên run

```
RUN = f"SSHAR_ESP_{MODEL}_{split_tag}_p{PATIENCE}_seed{SEEDS.replace(' ', '-')}"
# split_tag = "user0" ... "user4" (cross-user, case k) hoặc "repeat"
```

Ví dụ: `SSHAR_ESP_AMCS_TCN_FC_user0_p10_seed42`. Tên run là tên thư mục kết quả và tên file zip,
nên đổi bất kỳ biến nào ở mục 1 thì tên run cũng đổi theo, không ghi đè kết quả cũ.

## 3. Tham số dòng lệnh mà script train của mỗi mô hình phải nhận

Script chính của mỗi mô hình (AMCS-TCN: `run_all.py`) phải nhận đúng các tham số sau, cùng tên, cùng
ý nghĩa:

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `--data_path` | thư mục chứa `room_02` | |
| `--split` | `user` \| `repeat` | mặc định `user` |
| `--user_case` | `0`..`4` | mặc định `0` |
| `--patience` | số nguyên | mặc định `10`; `0` = tắt dừng sớm |
| `--seeds` | danh sách số nguyên | mặc định `42` |
| `--results_dir` | thư mục gốc kết quả | trên Kaggle dùng `results` |
| `--run_name` | tên run (mục 2) | |
| `--batch_size`, `--epochs`, `--lr` | ghi đè thiết lập chung | chỉ dùng khi buộc phải đổi, và ghi vào báo cáo |

## 4. Năm case chia theo người (ESP32, 10 người)

Lấy từ `USER_CASES['sshar']` của benchmark (8 người train + val, 2 người còn lại là test). Người val
tách từ 8 người đó: case 0 giữ người cuối (12); case 1 đến 4 chọn ngẫu nhiên (`random.Random(42)`, lần
lượt từng case, không trùng người val của các case trước) để năm case có năm người val khác nhau.
Mỗi người có 640 mẫu, nên case nào cũng ra **train 4480 / val 640 / test 1280**.

| Case | Train (7 người) | Val (1 người) | Test (2 người) |
|---|---|---|---|
| **0 (mặc định)** | **01, 02, 03, 04, 09, 10, 11** | **12** | **13, 14** |
| 1 | 01, 03, 04, 09, 11, 12, 14 | 13 | 02, 10 |
| 2 | 02, 04, 09, 10, 12, 13, 14 | 01 | 03, 11 |
| 3 | 03, 04, 10, 11, 12, 13, 14 | 02 | 01, 09 |
| 4 | 01, 02, 04, 09, 11, 13, 14 | 10 | 03, 12 |

Người 03 nằm trong test ở 2 case (2 và 4), người 04 không nằm trong test ở case nào. Nếu báo cáo
trung bình qua nhiều case thì ghi chú điều này.

`SPLIT = "repeat"` không dùng `USER_CASE`: người nào cũng có ở cả ba tập, chia theo lần lặp
(xem mục 3B của protocol).

## 5. Cấu trúc kết quả

```
results/<RUN>/
  summary.txt        # cấu hình (split, case, người từng tập, patience, ...) + accuracy / P / R / F1
  summary.csv
  config.json
  <MODEL>/
    per_seed.csv
    seed_<s>/
      metrics.json          # accuracy, precision, recall, f1 (macro), best_epoch, epochs_trained, ...
      history.csv           # mỗi epoch: train loss/acc, val loss/acc
      curve.png             # loss / accuracy theo epoch (train vs val)
      cm.png, cm_normalized.png, confusion_matrix.csv, per_class_metrics.csv, predictions.csv
      val/                  # cùng các file trên nhưng đo trên val (cùng checkpoint)
```

Tối thiểu mỗi mô hình phải có `summary.txt`, `metrics.json` (đủ 4 metric macro, best epoch, số epoch
đã chạy), `history.csv`, `curve.png` và confusion matrix. File ở `seed_<s>/` là kết quả trên test,
còn `seed_<s>/val/` là trên val.

## 6. Cell mẫu

Chép nguyên cell, chỉ sửa phần **CẤU HÌNH** và phần **MÔ HÌNH** (repo, nhánh, script) cho mô hình của
mình. Ví dụ dưới đây điền sẵn cho AMCS-TCN.

```python
# ===================== CẤU HÌNH (chung cho mọi mô hình) =====================
DATASET_DIR = "/kaggle/input/datasets/phanvanhai/csi-room-02-0707"
SPLIT     = "user"    # "user": cross-user (mặc định) | "repeat": theo lần lặp
USER_CASE = 0         # 0..4 (xem bảng 5 case); case 0: train 01,02,03,04,09,10,11 | val 12 | test 13,14
PATIENCE  = 10        # dừng sớm sau N epoch val acc không cải thiện; 0 = tắt
SEEDS     = "42"      # vd. "42 3407 1024 2024 0"
FILTERS   = "none"    # tiền xử lý riêng của AMCS-TCN ("none" | "lowpass hamming kalman")

# ===================== MÔ HÌNH (mỗi mô hình sửa phần này) =====================
MODEL     = "AMCS_TCN_FC"
REPO_URL  = "https://github.com/phanvanhai/amcs_tcn.git"
BRANCH    = "sshar_esp"
REPO_DIR  = "amcs_tcn"
ENTRY     = "run_all.py"
EXTRA_ARGS = f"--dataset SSHAR_ESP --filters {FILTERS}"   # tham số riêng của mô hình

# ===================== TÊN RUN =====================
split_tag = f"user{USER_CASE}" if SPLIT == "user" else "repeat"
RUN = f"SSHAR_ESP_{MODEL}_{split_tag}_p{PATIENCE}_seed{SEEDS.replace(' ', '-')}"

# ===================== LẤY CODE =====================
import os, glob
%cd /kaggle/working
if os.path.isdir(REPO_DIR):
    %cd {REPO_DIR}
    !git fetch origin && git checkout {BRANCH} && git reset --hard origin/{BRANCH}
else:
    !git clone -b {BRANCH} {REPO_URL} {REPO_DIR}
    %cd {REPO_DIR}
!git log --oneline -1
!pip install -q -r requirements.txt

# ===================== DỮ LIỆU =====================
# DATA = thư mục CHỨA room_02 (tự dò, kể cả khi có lớp sshar_amp_interp/ ở giữa)
hits = glob.glob(f"{DATASET_DIR}/**/room_*/esp/rx_00", recursive=True)
assert hits, f"Không thấy room_*/esp/rx_00 trong {DATASET_DIR}"
DATA = hits[0].rsplit('/room_', 1)[0]
print(f"DATA = {DATA}\nRUN  = {RUN}")

# ===================== TRAIN -> CHỌN CHECKPOINT THEO VAL -> ĐÁNH GIÁ =====================
!python -u {ENTRY} {EXTRA_ARGS} --split {SPLIT} --user_case {USER_CASE} --patience {PATIENCE} --seeds {SEEDS} --data_path "{DATA}" --results_dir results --run_name {RUN}

# ===================== KẾT QUẢ + HÌNH =====================
from IPython.display import Image, display, Markdown
!cat results/{RUN}/summary.txt
for seed_dir in sorted(glob.glob(f"results/{RUN}/*/seed_*")):
    display(Markdown(f"### {seed_dir}"))
    figs = [("Loss / accuracy theo epoch (train vs val)", "curve.png"),
            ("Confusion matrix trên TEST (chuẩn hoá theo hàng)", "cm_normalized.png"),
            ("Confusion matrix trên TEST (số mẫu)", "cm.png"),
            ("Confusion matrix trên VAL (cùng checkpoint, chuẩn hoá)", "val/cm_normalized.png")]
    for title, name in figs:
        path = os.path.join(seed_dir, name)
        if os.path.exists(path):
            display(Markdown(f"**{title}**"))
            display(Image(filename=path, width=700))

!cd results && zip -qr /kaggle/working/{RUN}.zip {RUN}
print(f"Đã nén: /kaggle/working/{RUN}.zip")
```

Trong log, kiểm tra hai dòng sau trước khi để chạy tiếp (với AMCS-TCN; mô hình khác nên in thông tin
tương đương):

```
👥 Split user case 0 | test: 13, 14 | train: 01, 02, 03, 04, 09, 10, 11 | val: 12
📦 sizes: {'train': 4480, 'val': 640, 'test': 1280} | ...
```

## 7. Ghi chú

- Nếu báo `CUDA out of memory`: thêm `--batch_size 16` vào `EXTRA_ARGS`, và ghi lại trong báo cáo vì
  đây là lệch khỏi thiết lập chung.
- Một phiên Kaggle tối đa 12 giờ. Khi chạy nhiều seed hoặc nhiều case, chia thành nhiều run (mỗi run
  một tên riêng theo mục 2).
- Clone vào `/kaggle/working` để kết quả còn lại sau khi phiên kết thúc; tải file zip về.
