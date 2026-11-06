# Phương án tiền xử lý cho phần cứng 1 anten (SISO)

> Trạng thái: **đã chốt phương án, chưa code** (2026-11-04).
> Bối cảnh: phần cứng thực tế có Tx 1 anten, mỗi Rx 1 anten → không tính được CSI ratio như UniFi / WiGRUNT / ABG.
> Kiến trúc mô hình (Module 2, 3 trong [unifi_architecture.md](unifi_architecture.md)) **giữ nguyên**; chỉ thay Module 1.

---

## 1. Tóm tắt quyết định

| View đầu vào | Pipeline ratio (hiện tại) | Pipeline SISO (chốt) |
|---|---|---|
| **X_f — DFS** | H_r phức → trừ mean → (HPF) → PCA PC1 → STFT 2 phía → 121 bin | **\|H\|** → Hampel → trừ mean → **HPF 2 Hz (bắt buộc)** → PCA **PC1** → STFT **1 phía** → **61 bin** (0…60 Hz) |
| **X_t — miền thời gian** | Pha ∠H_r [30, T] | **Biên độ thô \|H\|** [30, T] → Hampel → low-pass. **Không bù AGC, không trừ tĩnh, không PCA** |
| Visualize | min-max toàn ảnh → parula → resize 224×224 | Giữ nguyên |

---

## 2. Vì sao chọn như vậy

**Không dùng được ratio.** Ratio hoạt động vì nhiễu AGC và lệch pha (CFO/SFO/STO) giống nhau giữa các anten trên cùng một card, nên phép chia triệt tiêu chúng. Với SISO không có cặp anten nào, kể cả phía Tx. Chia giữa các Rx khác nhau cũng không được vì mỗi Rx có đồng hồ riêng.

**Không dùng pha.** Với SISO, chỉ khử được lệch pha bằng cách fit tuyến tính theo subcarrier. Cách này cũng xóa luôn phần pha chung do chuyển động gây ra. UniFi (Fig. 12) cho thấy pha thô khi đổi phòng chỉ ngang đoán ngẫu nhiên.

**Dùng biên độ.** Biên độ không bị lệch pha ảnh hưởng. UniFi (Fig. 12) cho thấy biên độ thô trong cùng miền đạt gần bằng biên độ của ratio. Nhược điểm của biên độ là kém khi đổi phòng, do mang theo phản xạ tĩnh của môi trường. Dữ liệu chỉ có 1 phòng nên điểm này ít ảnh hưởng.

**DFS lấy từ biên độ.** `|H|² = |Hs|² + |Hd|² + 2·Re(Hs·Hd*)`. Số hạng chéo dao động ở f_D và chỉ phụ thuộc biên độ. Khi |Hs| ≫ |Hd| thì |H| cũng xấp xỉ tuyến tính theo số hạng này, nên dùng |H| là đủ và dùng chung được với nhánh X_t.

**Chỉ lấy PC1.** Khớp với Widar3.0 (`get_doppler_spectrum.m`, dòng 69–70: `conj_mult * pca_coef(:,1)`) và WiFE (*"the first component presents the best performance"*). UniFi, WiGRUNT và ABG không dùng PCA.

**Hai view bổ sung cho nhau.** X_f mang thông tin tần số/tốc độ, đã gộp các subcarrier qua PCA. X_t giữ nguyên 30 subcarrier, mang thông tin hình dạng theo thời gian và kiểu fading khác nhau giữa các subcarrier.

---

## 3. Chi tiết pipeline

### 3.0 Yêu cầu khi thu dữ liệu
- Tần số gói **fs ≈ 1000 Hz**, không thấp hơn 500 Hz. Doppler của cử chỉ tay khoảng ≤ 60 Hz.
- Ghi timestamp cho từng gói, sau đó **nội suy về lưới thời gian đều**. Gói rơi hoặc giãn cách lệch sẽ làm nhòe STFT.
- Mỗi file chứa 1 Rx, nên mỗi Rx là 1 view, giống Widar3.0.

### 3.1 Bước chung: lọc cơ bản
```
H [K=30, T] complex
  → A = |H|                                  [30, T]
  → Hampel theo trục thời gian (từng subcarrier) → bỏ xung nhiễu
```
- Tham số Hampel khởi điểm: cửa sổ khoảng 0.1 s, ngưỡng 3σ (MAD). Chỉnh lại sau khi xem dữ liệu thật.

### 3.2 Nhánh DFS → X_f
```
A [30, T]
  → trừ mean theo thời gian (từng subcarrier)            bỏ thành phần tĩnh
  → Butterworth high-pass 2 Hz, bậc 3                   BẮT BUỘC (khác bản ratio đang tắt)
  → PCA theo subcarrier: SVD của X = Aᵀ [T, 30]
      signal = X · v₁                                    [T] thực, chỉ PC1
  → STFT: Hann, nperseg=256, noverlap=248 (hop 8), nfft=1000 (1 Hz/bin),
          return_onesided=True, boundary=None, padded=False
  → giữ 0 ≤ f ≤ 60 Hz, lấy |·|                          F_DFS [61, L]
```
- Dấu của v₁ không quan trọng vì chỉ lấy độ lớn của phổ.
- So với bản ratio: dữ liệu vào PCA là số thực, STFT chỉ một phía, F = 61 thay vì 121.

### 3.3 Nhánh biên độ → X_t
```
A [30, T] (đã qua Hampel)
  → Butterworth low-pass ~80 Hz, zero-phase (filtfilt)  làm mượt, không lệch thời gian
  → T_amp [30, T]
```
- **Không** bù AGC, **không** trừ mean, **không** PCA. Giữ đúng tinh thần "biên độ thô" giống ảnh biên độ trong UniFi Fig. 11.

### 3.4 Visualize
Giữ nguyên `to_rgb_image`: min-max trên toàn ảnh → parula → [3, H, W] → resize 224×224 trong Dataset.

---

## 4. Khi bắt đầu code (ánh xạ vào `unifi_preprocessing.py`)

| Hàm hiện có | Việc cần làm |
|---|---|
| `csi_ratio` | Thêm hàm song song lấy \|H\| của 1 anten và chạy Hampel (bước 3.1). Giữ `csi_ratio` cho baseline. |
| `extract_dfs` | Thêm chế độ cho input thực: bật HPF, `return_onesided=True`, không `fftshift`, giữ 0…60 Hz. |
| `extract_phase` | Thêm `extract_amplitude` (bước 3.3) để thay thế. |
| `preprocess_one_view` / `preprocess_all_views` | Thêm cờ chọn chế độ `ratio` hoặc `siso`. |
| `test_preprocessing`, `visualize_one_view` | Kiểm tra lại shape: X_f [3, 61, L] trước khi resize. |

---

## 5. Kiểm chứng trên Widar3.0 trước khi chạy phần cứng thật

1. **Giả lập SISO:** mỗi Rx của Widar3.0 có 3 anten → chỉ lấy anten 0 và chạy pipeline SISO.
2. **Baseline:** pipeline ratio hiện tại (DFS + pha), chạy trên cùng tập dữ liệu và cùng cách chia train/test.
3. **So sánh:** độ chính xác tổng, và ma trận nhầm lẫn của các cặp cử chỉ ngược chiều (push/pull, slide trái/phải).
4. **Kiểm tra bằng mắt:** vẽ X_f và X_t của cùng một mẫu theo cả hai pipeline, đối chiếu với UniFi Fig. 11.

---

## 6. Hạn chế đã biết

- **DFS không có dấu.** Chỉ biết |vận tốc|, không biết lại gần hay ra xa. Các cặp cử chỉ ngược chiều có thể bị nhầm. Kỳ vọng 6 Rx đặt ở các hướng khác nhau và hình dạng theo thời gian sẽ bù lại.
- **Hai view trùng thông tin hơn so với bản ratio**, vì cả hai cùng lấy từ |H|.
- **Không bù AGC.** Nếu chip thay đổi gain theo từng gói, X_t sẽ có các vạch dọc (nhảy đồng thời trên mọi subcarrier). Xem mục 7.
- **Biên độ thô chứa phản xạ tĩnh** → nhạy với thay đổi môi trường. Hiện chấp nhận được vì chỉ dùng 1 phòng.

---

## 7. Hướng dự phòng / ablation (chưa dùng mặc định)

| Thay đổi | Khi nào thử |
|---|---|
| Bù AGC: chia \|H(f,t)\| cho trung bình các subcarrier tại t | X_t có vạch dọc, hoặc DFS có năng lượng dàn trải đột ngột trên mọi tần số |
| Trừ tĩnh / high-pass cho X_t | Khi mở rộng sang nhiều phòng |
| Bỏ PC1, dùng PC2–PC3 hoặc tổng phổ của PC1–3 | PC1 có vẻ chủ yếu là nhiễu. Theo CARM (Wang et al., MobiCom 2015), **cần đọc lại bài gốc để xác nhận chi tiết** |
| Chọn vài subcarrier có phương sai lớn nhất thay cho PCA | Ablation đơn giản để đối chứng với PCA |
| Dùng \|H\|² thay cho \|H\| ở nhánh DFS | Ablation nhỏ |
| Thêm pha đã sanitize (fit tuyến tính theo subcarrier, trừ khuôn pha tĩnh) làm view thứ 3 | Nếu D+A chưa đủ: so D, D+A, D+P, D+A+P |

---

## 8. Nguồn

- **UniFi** — Liu et al., IMWUT 2023: §5.1 (ratio, DFS, visualize), Fig. 12 (có/không ratio), Fig. 13 (D / DP / DPA).
- **WiFE** — Gu et al.: PCA để khử nhiễu xung, chỉ dùng PC1, sau đó DWT.
- **WiGRUNT**, **ABG (Attention-Based Gesture Recognition)**: dùng ratio; ảnh pha (và biên độ) đưa vào ResNet18.
- **Widar3.0** — `DFSExtractionCode/get_doppler_spectrum.m`: PCA lấy PC1 sau nhân liên hợp giữa 2 anten.
