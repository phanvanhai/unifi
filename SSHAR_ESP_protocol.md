# SSHAR ESP: dữ liệu và quy ước train chung

Note này cố định **thông số thiết bị, cấu trúc dữ liệu, cách chia train/val/test và thiết lập train**
cho dataset SSHAR ESP. Mọi mô hình trong benchmark (AMCS-TCN và các mô hình khác) đều phải theo đúng
note này để kết quả so sánh được với nhau. Code tham chiếu: `datasets/csi_dataset.py`
(`sshar_split_of`, `_sshar_index`), nhánh `sshar_esp`.

## 1. Thiết bị và tín hiệu

| Mục | Giá trị |
|---|---|
| Thiết bị | ESP32 (thư mục `esp`) |
| Số receiver | 3 (`rx_00`, `rx_01`, `rx_02`) |
| Số anten mỗi receiver | 1 |
| Số subcarrier | 56 |
| Tín hiệu | biên độ CSI (`amp_*.npy`, thư mục `sshar_amp_interp`), float32, giá trị khoảng 1 đến 57 |
| Độ dài mỗi bản ghi | 1000 điểm = 5 s (200 Hz) |
| Shape một file (một receiver) | `(1, 56, 1000)` = (anten, subcarrier, thời gian) |
| Phòng | `room_02` (chỉ có một phòng) |

Một bản ghi là các file cùng tên ở cả 3 receiver, nên một mẫu có shape `(3, 56, 1000)`
(3 receiver × 1 anten = 3 luồng).

## 2. Cấu trúc thư mục và tên file

```
sshar_amp_interp/
  room_02/
    esp/
      rx_00/ subject_01/ amp_act01_pos01_dir00_rep01.npy ...
      rx_01/ subject_01/ ...      # cùng tên file với rx_00
      rx_02/ subject_01/ ...
```

Tên file: `amp_act{AA}_pos{BB}_dir{CC}_rep{DD}.npy`

| Trường | Ý nghĩa | Giá trị |
|---|---|---|
| `act` | hành động, là nhãn | 01 đến 08, ánh xạ sang chỉ số lớp 0 đến 7 (`act - 1`) |
| `pos` | vị trí | 01 đến 08 |
| `dir` | hướng | `00` = không có hướng; `01`, `02` = hai hướng ngược nhau |
| `rep` | lần lặp | 1 đến 10 với dir00, 1 đến 5 với dir01/dir02 |

10 người (subject): `01, 02, 03, 04, 09, 10, 11, 12, 13, 14`.

### Hành động và hướng

Các hành động chia làm hai nhóm. Đây là lý do cách chia tập khác nhau giữa hai nhóm:

| Hành động | Hướng | Số lần lặp | Số bản ghi mỗi người, mỗi hành động |
|---|---|---|---|
| act01, act02, act03, act08 | chỉ `dir00` | 10 | 8 vị trí × 10 lần = **80** |
| act04, act05, act06, act07 | `dir01` và `dir02` | 5 mỗi hướng | 8 vị trí × 2 hướng × 5 lần = **80** |

Tổng: 10 người × 8 hành động × 80 = **6400 bản ghi** (800 mỗi lớp, cân bằng).
Tên hành động chưa chốt, báo cáo tạm dùng `act01` … `act08`.

## 3. Chia train / val / test (7:1:2, theo lần lặp)

Chia **theo lần lặp**: người nào, vị trí nào cũng có mặt ở cả ba tập, và một bản ghi (cả 3 receiver)
luôn nằm trọn trong một tập. Không xáo trộn ngẫu nhiên từng mẫu rồi mới chia, vì như vậy các mẫu gần
như trùng nhau sẽ rơi vào cả train lẫn test.

Với **mỗi cặp (người, hành động)**, 80 bản ghi được chia đúng **56 / 8 / 16**.

### Nhóm 1: `dir00` (act01, 02, 03, 08), 10 lần lặp mỗi vị trí

| Tập | Quy tắc | Mỗi (người, hành động) |
|---|---|---|
| train | rep 1 đến 7 | 8 vị trí × 7 = 56 |
| val | rep 8 | 8 vị trí × 1 = 8 |
| test | rep 9 và 10 | 8 vị trí × 2 = 16 |

### Nhóm 2: `dir01` / `dir02` (act04, 05, 06, 07), 5 lần lặp mỗi hướng mỗi vị trí

| Tập | Quy tắc | Mỗi (người, hành động) |
|---|---|---|
| test | rep 5 của **cả hai** hướng | 8 vị trí × 2 hướng × 1 = 16 |
| val | rep 4 của **một** hướng: **dir01 nếu pos lẻ, dir02 nếu pos chẵn** | 8 vị trí × 1 = 8 |
| train | phần còn lại: rep 1 đến 3 của cả hai hướng + rep 4 của hướng kia | 8 × 2 × 3 + 8 = 56 |

Vì sao val luân phiên hướng: nếu lấy rep 4 của cả hai hướng thì val có 16 mẫu (2 phần chứ không phải
1). Nếu chỉ lấy cố định một hướng thì hướng còn lại không có mặt trong val. Luân phiên theo vị trí
giữ đúng tỉ lệ 7:1:2, và hai hướng có mặt trong val như nhau.

Ví dụ act04, subject_01, pos03 (lẻ):

| rep | dir01 | dir02 |
|---|---|---|
| 1 đến 3 | train | train |
| 4 | **val** | train |
| 5 | test | test |

Với pos04 (chẵn), ở rep 4: dir01 thuộc train, dir02 thuộc **val**.

### Quy tắc dạng code (bản tham chiếu, chép nguyên văn)

```python
def sshar_split_of(pos, direction, rep):
    if direction == 0:                      # 10 lần lặp
        return 'test' if rep > 8 else 'val' if rep == 8 else 'train'
    if rep == 5:                            # dir01 / dir02, 5 lần lặp
        return 'test'
    val_dir = 1 if pos % 2 else 2           # pos lẻ -> dir01, pos chẵn -> dir02
    return 'val' if rep == 4 and direction == val_dir else 'train'
```

### Số mẫu phải ra (kiểm tra sau khi load)

| Tập | Tổng | Mỗi lớp | Theo hướng |
|---|---|---|---|
| train | 4480 | 560 | |
| val | 640 | 80 | dir00 320, dir01 160, dir02 160 |
| test | 1280 | 160 | |

So với cách chia cũ của benchmark (80/20: train = dir00 rep ≤ 8, dir01/02 rep ≤ 4): **tập test giữ
nguyên**, tập val được tách ra từ tập train cũ.

## 4. Thiết lập train (chung cho mọi mô hình)

| Thiết lập | Giá trị |
|---|---|
| Batch size | 32 |
| Epochs | tối đa 100 |
| Optimizer | Adam |
| Learning rate | theo bài báo gốc của từng mô hình (AMCS-TCN: 0.001, Table 1 của bài) |
| Chọn model | checkpoint có **val accuracy** cao nhất |
| Metric | accuracy, precision, recall, F1 (precision/recall/F1 lấy trung bình **macro** trên 8 lớp) |
| Seed | hiện chỉ dùng 42 để đánh giá nhanh, sau này có thể thêm seed |

Hai giai đoạn:

1. **Development:** train trên train, chọn checkpoint theo val, **báo cáo trên val**. Mọi so sánh
   và mọi quyết định điều chỉnh đều dựa vào giai đoạn này, không nhìn vào test. Hiện chạy seed 42.
   Khi có thêm seed thì báo cáo mean ± std qua các seed.
2. **Final test:** chạy một lần duy nhất ở bước cuối, seed 42. Train như cũ, rồi đánh giá checkpoint
   (chọn theo val) trên **test**; cùng checkpoint đó cũng được đo trên val để đối chiếu, tất cả trong
   một lần chạy. Sau khi xem kết quả test thì không điều chỉnh gì thêm.

Những mục thuộc về từng mô hình (ghi rõ trong báo cáo của mô hình đó, không phải thiết lập chung):
early stopping, LR scheduler, augmentation, class weights, chuẩn hoá input và cách reshape input.
Để tham khảo, AMCS-TCN dùng early stop patience 10 theo val accuracy, StepLR(20, 0.5), nhiễu Gaussian
std 0.02, min-max theo min/max toàn cục của tập train, và giữ đủ T = 1000 với input `(3, 1000, 56)`.

## 5. Checklist khi thêm một mô hình mới

- [ ] Đọc đủ 3 receiver, và bản ghi nào cũng có đủ ở cả ba
- [ ] Nhãn `act01..08` ánh xạ sang `0..7`
- [ ] Dùng đúng `sshar_split_of` ở trên, số mẫu khớp **4480 / 640 / 1280** (mỗi lớp 560 / 80 / 160)
- [ ] Thống kê chuẩn hoá (nếu có) chỉ tính trên **train**
- [ ] Batch 32, tối đa 100 epoch, Adam, LR theo bài gốc
- [ ] Chọn checkpoint theo val accuracy cao nhất
- [ ] Kết quả development báo cáo trên val (seed 42); test chỉ đánh giá một lần (seed 42)
- [ ] Báo cáo accuracy và precision / recall / F1 (macro)
