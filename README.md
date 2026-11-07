# UniFi — Nhận dạng cử chỉ Wi-Fi đa góc nhìn (Widar 3.0)

Cài đặt lại mô hình **UniFi** (Liu et al., Proc. ACM IMWUT 2023) bằng PyTorch, huấn luyện trên dữ liệu CSI của Widar 3.0 (1 Tx, 6 Rx = 6 view).

Tài liệu này hướng dẫn chạy trên **Kaggle Notebook** (GPU). Ở cuối có phần chạy trên máy cá nhân.

---

## 1. Cấu trúc thư mục

```
unifi/
├── main.py                 # Điểm vào: cấu hình, đọc dữ liệu, chia train/test, train
├── unifi_preprocessing.py  # CSI ratio → DFS + pha → ảnh 3×224×224
├── unifi_intra_view.py     # ResNet18 trích đặc trưng trong từng view
├── unifi_inter_view.py     # Attention giữa các view + spatial mask
├── unifi_fusion.py         # Gộp đặc trưng các view
├── unifi_model.py          # Ghép thành mô hình UniFi
├── unifi_loss_train.py     # Loss (global + view + proxy contrastive), vòng train, ghi log
├── plot_logs.py            # Vẽ loss/acc, confusion matrix… từ log của một lần chạy
├── plot_samples.py         # Vẽ thử ảnh DFS/pha của vài mẫu
├── unifi_cache.py          # Cache tiền xử lý (DFS/pha) để train nhanh
├── kaggle_unifi.ipynb      # Notebook chạy trên Kaggle (clone repo → train → xem kết quả)
├── requirements.txt
└── widar/                  # Dữ liệu (không đưa lên git, xem mục 2)
```

## 2. Dữ liệu

Tên file theo định dạng của Widar 3.0:

```
widar/<user>/<user>-<gesture>-<location>-<orientation>-<repetition>-r<rx>.dat
ví dụ: widar/user5/user5-1-1-1-1-r1.dat
```

- 1 mẫu = 6 file `r1`…`r6`, mỗi file ứng với một bộ thu (view).
- Thư mục user có thể nằm sâu hơn một cấp, ví dụ `widar/20181130_user15_16_17/user15/`. Code vẫn tự tìm thấy.
- Code tự bỏ qua mẫu thiếu bộ thu hoặc có file quá ngắn (< 256 packet) và in danh sách các mẫu bị bỏ.

---

## 3. Chạy trên Kaggle

Quy trình: code nằm trên GitHub → notebook Kaggle clone về → chạy `main.py` với đường dẫn dataset → kết quả ghi vào `/kaggle/working/runs/`.

### Bước 1 — Đưa code lên GitHub

```bash
cd unifi
git init && git add . && git commit -m "UniFi"
git branch -M main
git remote add origin https://github.com/<tài-khoản>/<repo>.git
git push -u origin main
```

`.gitignore` đã loại `widar/`, `*.zip`, `*.pdf`, `DFSExtractionCode/`, `runs/`, `cache/`, `*.pth`.
Repo private: tạo GitHub token (quyền đọc repo), thêm vào Kaggle **Add-ons → Secrets** tên `GITHUB_TOKEN`, rồi đặt `PRIVATE_REPO = True` trong notebook.

### Bước 2 — Upload dữ liệu thành Kaggle Dataset

Nén `widar/` thành `widar.zip`, vào **Datasets → New Dataset** và upload. Kaggle sẽ tự giải nén. Để chế độ **Private** nếu dữ liệu không được phép chia sẻ. Không đưa `.zip` lồng nhau vào trong.

### Bước 3 — Notebook

1. **File → Import Notebook** và chọn `kaggle_unifi.ipynb` trong repo, hoặc tạo notebook mới rồi chép các ô của file này sang.
2. **Add Input**: chọn dataset ở bước 2.
3. **Settings**: Accelerator chọn **GPU T4 x2** (hoặc P100), bật **Internet On**.
4. Ở ô *Cấu hình*, sửa `REPO_URL` và `ARGS`. `DATA_ROOT` có thể để trống: notebook sẽ tự tìm thư mục chứa `userX/*.dat`.
5. Bấm **Save Version → Save & Run All (Commit)**. Notebook chạy nền, kể cả khi bạn đóng trình duyệt.
6. Khi chạy xong, mở version đó → tab **Output** để tải `runs/<tên run>.zip`.

Lệnh mà notebook chạy:

```bash
python -u main.py --data-root /kaggle/input/<dataset>/widar --output-dir /kaggle/working --epochs 100 --split random
```

Ví dụ `ARGS` khác:

```bash
--split location --test-values 3          # cross-location, test ở vị trí 3
--split user --test-values user5 user10   # cross-user
--split orientation --test-values 1       # cross-orientation
--max-samples 120 --epochs 2              # chạy thử vài phút để kiểm tra mọi thứ
--batch-size 16                           # nếu thiếu VRAM
```

### Tốc độ trên Kaggle

- **Cache tiền xử lý** (bật mặc định): code đọc `.dat` và tính HPF + PCA + STFT cho mọi mẫu một lần, lưu vào `/tmp/unifi_cache`. Với ~6700 mẫu (9 user), cache chiếm **5.7 GB** và mất **~170 phút** trên 4 CPU của Kaggle. Các epoch sau chỉ còn colormap và resize. `/tmp` bị xoá sau mỗi phiên, nên hãy build cache **một lần** thành Kaggle Dataset (xem bên dưới).

### Build cache một lần thành Kaggle Dataset (khuyên dùng)

1. Ở ô *Cấu hình* của notebook, đặt `ARGS = CACHE_ARGS`, tức là:
   ```
   --cache-only --cache-dir /kaggle/working/unifi_cache --no-zip
   ```
   Thêm `--users ...` / `--gestures ...` nếu muốn cache khác mặc định. Tắt GPU cũng được (chỉ dùng CPU). **Save Version → Save & Run All**, chạy ~3 giờ. Output khoảng 6 GB (giới hạn `/kaggle/working` là 20 GB).
2. Khi version chạy xong: mở version → tab **Output** → **New Dataset**, đặt tên ví dụ `unifi-cache`, chế độ Private.
3. Notebook train: **Add Input** → dataset `unifi-cache`, trả `ARGS` về lệnh train. `main.py` tự tìm trong `/kaggle/input` (`--cache-search`) một cache có cùng tham số tiền xử lý và chứa đủ các mẫu cần dùng, log in `[cache] Dùng cache có sẵn ...`. Dữ liệu trong `/kaggle/input` không tính vào 20 GB output.

Cache build từ 9 user dùng được cho mọi cách chia (`random`, `user`, `location`, `orientation`) và cho tập con user/gesture. Đổi tham số tiền xử lý (`--no-highpass`, ...) hoặc thêm user ngoài cache thì code tự build lại vào `/tmp`.
- **AMP** (fp16) và **DataParallel** trên 2 GPU được bật tự động. Tắt bằng `--no-amp` hoặc `--single-gpu`.
- **Giới hạn thời gian**: trên Kaggle mặc định `--time-limit-hours 11.5`. Nếu epoch kế tiếp có thể vượt mốc này, code dừng sớm nhưng vẫn ghi đủ checkpoint, CSV, `summary.json`, hình và file zip.

### Log

Mọi thứ in ra màn hình đều được ghi vào `runs/<run>/train.log`, kèm timestamp:
- thông tin môi trường (GPU, VRAM, CUDA, RAM, git commit), toàn bộ cấu hình, phân bố train/test theo lớp, user, location và orientation, spatial mask, số tham số của từng module;
- tiến độ build cache: số mẫu/giây, dung lượng, thời gian còn lại;
- mỗi `--log-interval` batch: các thành phần loss, accuracy, số mẫu/giây, % thời gian chờ dữ liệu, thời gian còn lại của epoch, VRAM, RAM của container và `/dev/shm`;
- mỗi epoch: loss và accuracy của train/test, macro-F1, accuracy theo từng view và từng lớp, thời gian, ETA;
- cuối run: best/last accuracy và confusion matrix; nếu có lỗi thì ghi cả traceback.

### Chạy nhiều thí nghiệm trong một notebook

```python
for loc in [1, 2, 3, 4, 5]:
    !cd {CODE_DIR} && python -u main.py --data-root "{DATA_ROOT}" --split location --test-values {loc} --epochs 100
```

Cache được dùng chung giữa các lần chạy vì khoá cache phụ thuộc vào danh sách mẫu, không phụ thuộc cách chia train/test. Tổng thời gian cả notebook vẫn bị giới hạn 12 giờ, nên với chạy nhiều thí nghiệm hãy giảm `--time-limit-hours`.

---

## 4. Cấu hình

Mọi biến ở đầu `main.py` đều ghi đè được bằng tham số dòng lệnh (`python main.py --help`):

| Tham số | Biến | Ý nghĩa |
|---|---|---|
| `--data-root` | `DATA_ROOT` | Thư mục chứa `userX/*.dat` (có thể lồng sâu hơn) |
| `--output-dir` | `OUTPUT_DIR` | Nơi tạo `runs/` (Kaggle: `/kaggle/working`) |
| `--users`, `--gestures` | `USERS`, `GESTURES` | User và gesture id được dùng (nhãn = vị trí trong list) |
| `--split`, `--test-values`, `--test-ratio` | `SPLIT_BY`, `TEST_VALUES`, `TEST_RATIO` | `random` (in-domain, 20% test) / `user` / `location` / `orientation` |
| `--epochs`, `--batch-size`, `--lr`, `--beta1`, `--beta2` | | Theo bài báo: 100*, 32, 0.005, 1.0, 0.05 (*bài không nêu số epoch) |
| `--weight-decay`, `--grad-clip` | | Không có trong bài, mặc định tắt |
| `--no-mask`, `--no-highpass` | | Tắt spatial mask / tắt lọc thông cao 2 Hz trước DFS |
| `--workers` | `NUM_WORKERS` | Số tiến trình đọc dữ liệu (mặc định ≤ 4) |
| `--no-amp`, `--single-gpu`, `--no-cache`, `--cache-dir` | | Tốc độ, xem mục 3 |
| `--cache-only` | `CACHE_ONLY` | Chỉ build cache rồi thoát (để tạo Kaggle Dataset) |
| `--cache-search` | `CACHE_SEARCH` | Thư mục tìm cache build sẵn, chỉ đọc (Kaggle: `/kaggle/input`) |
| `--time-limit-hours` | `TIME_LIMIT_H` | Dừng sớm và vẫn ghi kết quả (Kaggle: 11.5, `0` = không giới hạn) |
| `--max-samples` | `MAX_SAMPLES` | Chỉ lấy N mẫu, chia đều theo lớp, để chạy thử |
| `--run-name`, `--log-interval`, `--no-plots`, `--zip/--no-zip` | | Tên run, tần suất log, vẽ hình, nén run |

**Chia dữ liệu:** chỉ có train và test, như bài UniFi và WiGRUNT. Checkpoint được chọn theo **test accuracy cao nhất**, nên con số này hơi lạc quan. Kết quả của epoch cuối (`*_last`) vẫn được lưu song song để báo cáo.

Biến môi trường (không bắt buộc): `UNIFI_DATA_ROOT`, `UNIFI_OUTPUT_DIR`.

---

## 5. Kết quả của một lần chạy

Mỗi lần chạy tạo thư mục `runs/<ngày_giờ>_<kiểu chia>/`, ví dụ `runs/20261104_231508_location-3/`:

| File | Nội dung |
|---|---|
| `train.log` | Toàn bộ nội dung in ra console |
| `config.json` | Cấu hình, môi trường (GPU, git commit, lệnh chạy), spatial mask, số mẫu train/test |
| `split.csv` | Mẫu nào thuộc train, mẫu nào thuộc test (kèm user/location/orientation) |
| `history.csv` | Mỗi epoch: train/test loss (tổng, G, V, P), acc, macro-F1, acc TB các view, lr, thời gian, thời gian chờ data, mẫu/s, VRAM |
| `test_predictions_best.csv` | Từng mẫu test tại epoch tốt nhất: nhãn, dự đoán, dự đoán của từng view, xác suất |
| `test_predictions_last.csv` | Như trên, tại epoch cuối |
| `confusion_matrix_best.csv`, `confusion_matrix_last.csv` | Ma trận nhầm lẫn |
| `best_model.pth`, `last_model.pth` | Checkpoint |
| `summary.json` | Best/last test acc + macro-F1, acc theo view/lớp, best epoch, lý do dừng, tổng thời gian, git commit |
| `<run>.zip` | Toàn bộ thư mục run nén lại (mặc định trên Kaggle) |
| `plots/` | Hình do `plot_logs.py` tạo: loss, acc, confusion matrix, acc theo view/location/orientation, `report.txt` (precision/recall/F1) |

Nạp lại checkpoint:

```python
import torch, main
from unifi_model import UniFi
from unifi_inter_view import build_spatial_mask

Mask  = build_spatial_mask(main.V, main.RX_DIRECTIONS)
model = UniFi(V=main.V, D=main.D, num_classes=len(main.GESTURES), Mask=Mask)
model.load_state_dict(torch.load('runs/<tên_run>/best_model.pth', map_location='cpu'))
```

---

## 6. Chạy trên máy cá nhân

```bash
# Có GPU: cài torch + torchvision bản CUDA theo https://pytorch.org/get-started/locally/ trước
pip install -r requirements.txt

python main.py           # train
python plot_logs.py      # vẽ kết quả của run mới nhất
python plot_samples.py   # (tuỳ chọn) xem ảnh DFS/pha của vài mẫu → figures/
```

Đặt dữ liệu ở `unifi/widar/`, hoặc đặt biến môi trường `UNIFI_DATA_ROOT`. Trên Windows nên chạy `python main.py` từ terminal: DataLoader có `num_workers > 0` có thể bị treo khi chạy trong Jupyter trên Windows.

---

## 7. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Không có mẫu nào` / `Không thấy thư mục userX` | Kiểm tra `--data-root` (xem dòng `cfg.data_root` trong log) và `--users` |
| `Tập rỗng: train=…, test=0` | `TEST_VALUES` không khớp dữ liệu, ví dụ location ghi `'3'` thay vì `3` (phải là số nguyên; user thì là chuỗi) |
| `Read-only file system` | Đang ghi vào `/kaggle/input`. Đặt `--output-dir /kaggle/working` |
| `No space left on device` khi build cache | `--cache-dir` sang ổ khác, hoặc `--no-cache` |
| `git clone` lỗi trên Kaggle | Chưa bật *Internet On*, hoặc repo private mà chưa có secret `GITHUB_TOKEN` |
| Run bị dừng với `stop_reason: time_limit` | Hết `--time-limit-hours`: giảm `--epochs`, hoặc tách thí nghiệm ra nhiều notebook |
| `CUDA out of memory` | `--batch-size 16` |
| `DataLoader worker … killed` / `Your notebook tried to allocate more memory than is available` | Xem cột `ram …/31G shm …G` trong log; giảm `--workers 2` (hoặc 0) |
| `Device: cpu` trên Kaggle | Chưa bật GPU trong *Settings → Accelerator* |

---

## Tham khảo

- Liu et al., *UniFi: A Unified Framework for Generalizable Gesture Recognition with Wi-Fi Signals Using Consistency-guided Multi-View Networks*, Proc. ACM IMWUT 7(4), 2023. DOI: 10.1145/3631429
- Zheng et al., *Widar3.0: Zero-Effort Cross-Domain Gesture Recognition with Wi-Fi*, MobiSys 2019 / TPAMI 2021. Dataset: https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset
