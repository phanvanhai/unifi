# Kiến trúc mô hình UniFi

> Liu et al., "UniFi: A Unified Framework for Generalizable Gesture Recognition with Wi-Fi Signals Using Consistency-guided Multi-View Networks." *Proc. ACM IMWUT*, vol. 7, no. 4, Article 168, December 2023. DOI: 10.1145/3631429

---

## Luồng dữ liệu tổng quát

```
Input CSI  H ∈ ℂᴹˣᴺˣᴷˣᵀ   M=1 TX, N=3 RX anten, K=30 sub-carrier
        ↓
  ┌────────────────────────────────────────────────────────┐
  │  Module 1: Signal Preprocessing                        │
  │    Khử nhiễu:  Hᵣ = H_{j₁} / H_{j₂}  ∈ ℂᴷˣᵀ         │
  │       ↓                                                │
  │    Biến đổi:   trừ mean → HPF 2 Hz → PCA → STFT        │
  │                → DFS ∈ ℝ¹²¹ˣᴸ                         │
  │                ∠Hᵣ → T_pha ∈ ℝᴷˣᵀ                    │
  │       ↓                                                │
  │    Visualize:  min-max + parula → resize 224×224       │
  │                Xⁱ_f ∈ ℝ³ˣ²²⁴ˣ²²⁴  (ảnh DFS)          │
  │                Xⁱ_t ∈ ℝ³ˣ²²⁴ˣ²²⁴  (ảnh pha)          │
  └──────────────────────── ↓ ─────────────────────────────┘
    × V=6 receivers → V bộ (Xⁱ_f, Xⁱ_t)
        ↓
  ┌────────────────────────────────────────────────────────┐
  │  Module 2: Multi-view Deep Network                     │
  │  [Intra-view]                                          │
  │    ResNet18(Xⁱ_t) + ResNet18(Xⁱ_f) → concat          │
  │    Channel Attention + Conv1D                          │
  │    → Fⁱ_intra ∈ ℝᴰˣᵀ,  D=64                          │
  │       ↓                                                │
  │  [Inter-view]                                          │
  │    Stack V views → F_intra ∈ ℝᵀˣᵛˣᴰ                   │
  │    Temporal-aware avg → F̃_intra                        │
  │    Spatial-aware mask → Mask ∈ {0,1}ᵛˣᵛ              │
  │    Masked Cross-view Attention (tại từng t)            │
  │    → F_inter ∈ ℝᵀˣᵛˣᴰ                                 │
  │       ↓                                                │
  │  [Fusion]                                              │
  │    Multi-scale Conv1D + GlobalAvgPool → pⁱₖ ∈ ℝᴰ     │
  │    Segment-level fusion → zⁱ ∈ ℝᴰ                     │
  │    View-level fusion → z_global ∈ ℝᴰ                  │
  └──────────────────────── ↓ ─────────────────────────────┘
        ↓
  ┌────────────────────────────────────────────────────────┐
  │  Module 3: Generalizable Representation Learning       │
  │    Proxy w_c = trọng số FC cuối (class proxy)          │
  │    L_pcon = proxy-based contrastive loss               │
  │    L = L_global + β₁·Σ Lⁱ_view + β₂·L_pcon           │
  │            β₁=1, β₂=0.05                              │
  └──────────────────────── ↓ ─────────────────────────────┘
        ↓
  Classifier: FC(z_global) + Softmax → ℝ|ᴬ|
  |A| = 9 (Widar room 1) hoặc 6 (room 2, 3)
```

---

## Chi tiết từng thành phần

### Module 1 — Signal Preprocessing

Code: `unifi_preprocessing.py` (biến đổi + ảnh), `main.py` (đọc `.dat`).

UniFi chỉ mô tả ngắn gọn module này và dẫn sang hai bài khác:
- **[12]** Gu et al., *Attention-Based Gesture Recognition Using Commodity WiFi Devices*, IEEE Sensors J. 2023 — ảnh pha/biên độ CSI ratio, resize 224×224×3.
- **[13]** Gu et al., *WiFE: WiFi and Vision Based Unobtrusive Emotion Recognition…* — phương pháp vẽ heatmap.

Tham số STFT không được báo cáo ở bài nào → chọn theo code gốc Widar3.0 (`DFSExtractionCode/get_doppler_spectrum.m`: fs = 1000, giữ ±60 Hz) và **kiểm tra trực quan với Fig. DFS của UniFi** (trục −60…60 Hz, 0…1.2 s, colorbar 0…1).

#### 1a. Đọc CSI + Khử nhiễu (CSI Ratio)

**Input:** 6 file `userX-<gesture>-<loc>-<ori>-<rep>-r<1..6>.dat` (Intel 5300), 1 file = 1 bộ thu = 1 view

**Đọc file** (`read_bf_file` trong `main.py`, port từ `read_bfee.c` của Linux 802.11n CSI Tool):
```
record = [2 byte độ dài big-endian][1 byte code][payload]
code 187 (0xBB) → CSI: Nrx=3, Ntx=1, 30 sub × 3 anten × (real, imag) int8
hoán vị anten theo antenna_sel
→ H ∈ ℂ¹ˣ³ˣ³⁰ˣᵀ   (T = số packet, 1000 packet/s, khác nhau giữa mẫu VÀ giữa bộ thu)
```
Không cần `get_scaled_csi`: hệ số scale giống nhau cho mọi anten trong 1 packet → triệt tiêu khi chia.

**Công thức** ([12] eq. 3 — hai anten **liền kề** cùng bộ thu, dùng j₁=1, j₂=2):
```
Hᵣ = H_{i,j₁} / H_{i,j₂}
   = [Aₙ·e^{-jθₙ}·(H_s1 + H_d1)] / [Aₙ·e^{-jθₙ}·(H_s2 + H_d2)]
   = (H_s1 + H_d1) / (H_s2 + H_d2)
```

**Output:** Hᵣ ∈ ℂᴷˣᵀ — nhiễu biên độ Aₙ và lệch pha θₙ được triệt tiêu

#### 1b. Biến đổi (Transform)

**Input:** Hᵣ ∈ ℂᴷˣᵀ,  K = 30

**f₁ — DFS** (`extract_dfs`):
```
① Bỏ thành phần tĩnh   H_d = Hᵣ − mean_t(Hᵣ)                    ∈ ℂ³⁰ˣᵀ
② Lọc thông cao        H_d ← HPF(H_d)   Butterworth bậc 3, 2 Hz, nhân quả, từng sub-carrier
                       (bật/tắt: DFS_HIGHPASS trong main.py, mặc định bật)
③ PCA                  s = H_dᵀ · v₁   (v₁ = thành phần chính 1)  ∈ ℂᵀ
④ STFT                 Hann, nperseg = 256, hop = 8, nfft = 1000, không pad
                       → 1 Hz/bin, hai phía (giữ tần số âm)
⑤ Cắt dải              |f| ≤ 60 Hz,  F_DFS = |STFT(s)|           ∈ ℝ¹²¹ˣᴸ
                       F = 121,  L = (T − 256)/8 + 1
```

| Bước | Lý do |
|---|---|
| ① trừ mean | Đường truyền tĩnh (LoS, tường) là hằng số theo thời gian → dồn ~70% năng lượng vào \|f\| ≤ 2 Hz; không trừ thì ảnh chỉ còn một vạch ngang ở 0 Hz, không thấy vệt Doppler như Fig. của UniFi. |
| ② HPF 2 Hz | Theo `get_doppler_spectrum.m` của Widar3.0 (`butter(3, 2/500, 'high')` + `filter`, áp lên từng sub-carrier trước PCA). Trừ mean chỉ bỏ DC tuyệt đối; trôi chậm của kênh vẫn để lại dải sáng ±5 Hz suốt mẫu, kể cả lúc đứng yên → min-max toàn ảnh làm dải này sáng nhất, vệt Doppler của cử chỉ (±20–60 Hz) bị tối. Lọc 2 Hz xóa dải ở đoạn tĩnh, giữ phần tần số thấp lúc đang chuyển động (so sánh: `figures/input_g*.png`, hàng 1 vs 2). Lọc nhân quả giống Widar → năng lượng trễ ~0.05–0.1 s, như nhau với mọi mẫu. Low-pass 60 Hz của Widar không cần vì bước ⑤ đã cắt \|f\| ≤ 60 Hz. |
| ③ PCA | Cử chỉ tác động tương quan lên 30 sub-carrier; PC1 giữ phần chuyển động (Widar3.0 và WiFE [13] đều dùng PC1). Ảnh gần như giống hệt trung bình \|STFT\| của 30 sub nhưng chỉ cần 1 STFT → nhanh ~30 lần. |
| ④ không pad | Giống `spectrogram` MATLAB: mất ½ cửa sổ mỗi đầu → mẫu T = 1467 (1.47 s) cho trục DFS ≈ 1.2 s, đúng như Fig. của UniFi. |
| ⑤ ±60 Hz | Theo Widar3.0 và trục của Fig. UniFi. |

**f₂ — Pha** (`extract_phase`):
```
T_pha = ∠Hᵣ ∈ [−π, π]     ∈ ℝ³⁰ˣᵀ     (không unwrap)
```
Fig. 3 của [12] có dải pha nhảy ±π (vàng lẫn đốm xanh đậm) → dùng pha gốc.

**Output:** F_DFS ∈ ℝ¹²¹ˣᴸ,   T_pha ∈ ℝ³⁰ˣᵀ

#### 1c. Visualize (Heatmap → Ảnh RGB)

**Input:** F_DFS ∈ ℝᶠˣᴸ,   T_pha ∈ ℝᴷˣᵀ

**Công thức** (`to_rgb_image` — tương đương `imagesc` của MATLAB):
```
① Chuẩn hóa      M̂ = (M − min M) / (max M − min M)      min-max trên TOÀN ảnh
② Colormap       Xⁱ = parula(M̂)  → ℝ³ˣᴴˣᵂ              (RGB, bỏ alpha)
③ Resize         bilinear + antialias → 3 × 224 × 224   (WiDarDataset)

f₁ (DFS image):  Xⁱ_f = resize(parula(minmax(F_DFS)))   →  ℝ³ˣ²²⁴ˣ²²⁴
f₂ (pha image):  Xⁱ_t = resize(parula(minmax(T_pha)))   →  ℝ³ˣ²²⁴ˣ²²⁴
```

| Bước | Căn cứ |
|---|---|
| ① min-max toàn ảnh | Heatmap trong [12], [13] và Fig. DFS của UniFi: độ sáng thay đổi theo thời gian/dải → không chuẩn hóa theo cột (khác Widar3.0) hay theo dải. |
| ② parula | Cả 3 bài ([12] Fig. 3, [13] Fig. 3–4, UniFi Fig. DFS) đều dùng parula (xanh tím → xanh lơ → vàng). Matplotlib không có sẵn → `PARULA` dựng từ 17 điểm mốc lấy mẫu trên colorbar Fig. DFS của UniFi. |
| ③ 224×224 | [12]: *"resize the dimensions of the input image to 224×224×3"*; [13]: *"unified each CSI data into a same size image"*. Bắt buộc vì T, L khác nhau giữa mẫu và giữa 6 bộ thu của cùng một mẫu → DataLoader mới stack được batch. Resize làm **một lần** ở `WiDarDataset`, intra-view nhận ảnh đã đúng kích thước. |

**Output:** Xⁱ_f ∈ ℝ³ˣ²²⁴ˣ²²⁴,   Xⁱ_t ∈ ℝ³ˣ²²⁴ˣ²²⁴   (C=3 kênh RGB),  × V=6 views

> **Tự chọn (không có trong bài):** cửa sổ Hann 256 / hop 8 (Widar dùng Gauss 251 / hop 1; đã thử Gauss 251 với hop 8 → ảnh gần như giống nhau), trừ mean + HPF 2 Hz của Widar (bỏ low-pass 60 Hz vì đã cắt dải sau STFT), PCA thay vì trung bình sub-carrier. Ảnh biên độ T_amp (DPA trong UniFi Fig. 13) chưa dùng — model hiện chỉ nhận DFS + pha (DP).

---

### Module 2 — Multi-view Deep Network

#### 2a. Intra-view Feature Extraction

**Input:** Xⁱ_t ∈ ℝ³ˣ²²⁴ˣ²²⁴,   Xⁱ_f ∈ ℝ³ˣ²²⁴ˣ²²⁴   (mỗi view i độc lập, đã resize ở Module 1)

**Công thức:**
```
f₁ (time backbone):  Fⁱ_t = ResNet18_t(Xⁱ_t; Θ_t)     shared across views
f₂ (freq backbone):  Fⁱ_f = ResNet18_f(Xⁱ_f; Θ_f)     shared across views
f₃ (concat):         Fⁱ_tf = [Fⁱ_t ; Fⁱ_f]             nối theo chiều kênh
f₄ (channel attn):   α = softmax(FC(Fⁱ_tf))
                      Fⁱ_weighted = α ⊙ Fⁱ_tf
f₅ (conv):           Fⁱ_intra = Conv1D(Fⁱ_weighted)
```

**Output:** Fⁱ_intra ∈ ℝᴰˣᵀ,   D=64

#### 2b. Inter-view: Stack + Temporal-aware Avg

**Input:** Fⁱ_intra ∈ ℝᴰˣᵀ   (V views)

**Công thức:**
```
f₁ (stack):   F_intra = stack([F¹_intra,...,Fᵛ_intra])   →  ℝᵀˣᵛˣᴰ

f₂ (temporal-aware avg):
  F̃^{i,t} = Σ_{j=−S}^{S} wⱼ · F^{i,t+j}_intra
  wⱼ ∝ 1/(1+|j|)²  (normalized)                         →  F̃_intra ∈ ℝᵀˣᵛˣᴰ
```

**Output:** F_intra ∈ ℝᵀˣᵛˣᴰ (dùng làm Q),   F̃_intra ∈ ℝᵀˣᵛˣᴰ (dùng làm K, V)

> **Tham số chưa rõ:** S — cửa sổ temporal-aware avg (không được báo cáo).

#### 2c. Inter-view: Spatial-aware Mask

**Input:** V views, thông tin vị trí thiết bị (Fresnel zone)

**Công thức:**
```
Với mỗi view i:
  1. Chọn m candidate views đặt ở hướng khác nhau (Fresnel complementary)
  2. Random chọn r target views từ m candidates
  3. Mask[i, j] = 0 nếu j không được chọn, 1 nếu được chọn
```

**Output:** Mask ∈ {0,1}ᵛˣᵛ

> **Tham số chưa rõ:** m và r — số views candidate và target (không được báo cáo).

#### 2d. Inter-view: Masked Cross-view Attention

**Input:** F_intra ∈ ℝᵀˣᵛˣᴰ (Q),   F̃_intra ∈ ℝᵀˣᵛˣᴰ (K, V),   Mask ∈ {0,1}ᵛˣᵛ

**Công thức (tính độc lập tại từng bước thời gian t):**
```
Qₕ = F_intra · Wqₕ      ∈ ℝᵀˣᵛˣdₖ
Kₕ = F̃_intra · Wkₕ     ∈ ℝᵀˣᵛˣdₖ
Vₕ = F̃_intra · Wvₕ     ∈ ℝᵀˣᵛˣdᵥ

Hₕ = Softmax( Mask(Qₕ·Kₕᵀ / √D) ) · Vₕ    ma trận attn: V×V per t

F_inter = Concat[H₁,...,Hₕ] · Wₒ           ∈ ℝᵀˣᵛˣᴰ
```

**Output:** F_inter ∈ ℝᵀˣᵛˣᴰ

> **Tham số chưa rõ:** h — số attention heads; dₖ=dᵥ=D/h=64/h (báo cáo D=64).

#### 2e. Segment-level Fusion

**Input:** Fⁱ_inter ∈ ℝᵀˣᴰ   (tách per view)

**Công thức:**
```
f₁ (multi-scale conv):  pⁱₖ = GlobalAvgPool( Conv1D(Fⁱ_inter, kernel=pₖ) )   ∈ ℝᴰ
f₂ (scale attn):        α^filter = softmax( ReLU( FC(concat[pⁱₖ]) ) )
f₃ (weighted sum):      zⁱ = Σₖ α^filter_k · pⁱₖ                              ∈ ℝᴰ
```

**Output:** zⁱ ∈ ℝᴰ   (T biến mất sau GlobalAvgPool)

> **Tham số chưa rõ:** tập kernel size pₖ (không được báo cáo).

#### 2f. View-level Fusion

**Input:** zⁱ ∈ ℝᴰ   (V views)

**Công thức:**
```
f₁ (view attn):   α^view_i = softmax( ReLU( FC(zⁱ) ) )
f₂ (weighted sum): z_global = Σᵢ α^view_i · zⁱ           ∈ ℝᴰ
```

**Output:** z_global ∈ ℝᴰ

---

### Module 3 — Generalizable Representation Learning

#### 3a. Proxy-based Contrastive Loss

**Input:** zᵢ ∈ ℝᴰ (sample embedding),   wc ∈ ℝᴰ (class proxy = trọng số FC cuối lớp c)

**Công thức:**
```
L_pcon = −(1/N) · Σᵢ log [
    e^(ŵcᵀ·žᵢ / τ)
    ──────────────────────────────────────────────
    e^(ŵcᵀ·žᵢ/τ) + Σ_{k≠c} e^(ŵkᵀ·žᵢ/τ) + Σ_{j≠i} e^(žᵢᵀ·žⱼ/τ)
]
```

Mẫu số gồm 2 loại negative:
- Proxy negative: ŵk với k ≠ c (proxy lớp khác)
- Sample negative: žⱼ với j ≠ i (mẫu khác trong batch)

> **Tham số chưa rõ:** τ — temperature (không được báo cáo).

#### 3b. Hàm mất mát tổng

```
L = L_global + β₁·Σᵢ Lⁱ_view + β₂·L_pcon

  L_global  = CE(y, ŷ)          từ z_global     [dùng khi inference]
  Lⁱ_view   = CE(y, ŷⁱ)         từ zⁱ per view  [chỉ dùng khi training]
  L_pcon    = proxy contrastive  từ zᵢ
  β₁ = 1,  β₂ = 0.05
```

---

### Classifier

**Input:** z_global ∈ ℝᴰ
**Công thức:** ŷ = softmax( FC(z_global) )
**Output:** ŷ ∈ ℝ|ᴬ|   (|A|=9 hoặc 6)

---

## Cài đặt huấn luyện

| Thiết lập | Giá trị |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.005 |
| Batch size | 32 |
| Backbone | ResNet18 |
| D (feature dim) | 64 |
| β₁ | 1 |
| β₂ | 0.05 |
| V (số views) | 6 |
| K (sub-carrier) | 30 |

---

## Sơ đồ PlantUML

### Sơ đồ tổng thể

```plantuml
@startuml UniFi_Overall
skinparam defaultFontSize 12
skinparam backgroundColor white
skinparam shadowing false
skinparam ArrowColor #555555
skinparam ArrowThickness 1.5

skinparam rectangle {
  BorderThickness 1.5
  RoundCorner 10
  BackgroundColor #F1EFE8
  BorderColor #5F5E5A
  FontColor #2C2C2A
}

skinparam package {
  BorderThickness 1.5
  RoundCorner 10
}

skinparam package<<m1>> {
  BackgroundColor #E1F5EE
  BorderColor #0F6E56
  FontColor #04342C
}

skinparam package<<m2>> {
  BackgroundColor #EEEDFE
  BorderColor #534AB7
  FontColor #26215C
}

skinparam package<<m3>> {
  BackgroundColor #FAECE7
  BorderColor #993C1D
  FontColor #4A1B0C
}

title UniFi — Kiến trúc tổng thể

rectangle "Input CSI\n─────────────────\nin:   H ∈ ℂᴹˣᴺˣᴷˣᵀ,  K=30\nout:  H ∈ ℂᴹˣᴺˣᴷˣᵀ" as INPUT

package "Module 1: Signal Preprocessing" <<m1>> as M1 {
  rectangle "Khử nhiễu (CSI Ratio)\n─────────────────\nin:   H ∈ ℂᴹˣᴺˣᴷˣᵀ\nf:    Hᵣ = H_{j₁} / H_{j₂}\nout:  Hᵣ ∈ ℂᴷˣᵀ" as DENOISE

  rectangle "Transform (STFT + Phase)\n─────────────────\nin:   Hᵣ ∈ ℂᴷˣᵀ\nf₁:   F_DFS = |STFT(PCA₁(HPF₂Hz(Hᵣ − mean)))|, |f| ≤ 60 Hz → ℝ¹²¹ˣᴸ\nf₂:   T_pha = ∠Hᵣ         → ℝᴷˣᵀ\nout:  F_DFS, T_pha" as TRANSFORM

  rectangle "Visualize → RGB image\n─────────────────\nin:   F_DFS ∈ ℝᶠˣᴸ,  T_pha ∈ ℝᴷˣᵀ\nf₁:   Xⁱ_f = resize(parula(minmax(F_DFS)))  → ℝ³ˣ²²⁴ˣ²²⁴\nf₂:   Xⁱ_t = resize(parula(minmax(T_pha)))  → ℝ³ˣ²²⁴ˣ²²⁴\nout:  Xⁱ_f, Xⁱ_t   × V=6 views" as VIZ
}

package "Module 2: Multi-view Deep Network" <<m2>> as M2 {
  rectangle "Intra-view Extraction\n─────────────────\nin:   Xⁱ_t, Xⁱ_f ∈ ℝ³ˣ²²⁴ˣ²²⁴\nf:    ResNet18 × 2 (shared) → concat\n      Channel Attention → Conv1D\nout:  Fⁱ_intra ∈ ℝᴰˣᵀ,  D=64" as INTRA

  rectangle "Inter-view Attention\n─────────────────\nin:   F_intra ∈ ℝᵀˣᵛˣᴰ\nf₁:   Temporal-aware avg → F̃_intra\nf₂:   Spatial-aware mask → Mask\nf₃:   Masked Cross-view Attn (per t)\nout:  F_inter ∈ ℝᵀˣᵛˣᴰ" as INTER

  rectangle "Feature Fusion\n─────────────────\nin:   F_inter ∈ ℝᵀˣᵛˣᴰ\nf₁:   Multi-scale Conv + AvgPool → pⁱₖ ∈ ℝᴰ\nf₂:   Segment-level attn → zⁱ ∈ ℝᴰ\nf₃:   View-level attn → z_global ∈ ℝᴰ\nout:  zⁱ ∈ ℝᴰ,  z_global ∈ ℝᴰ" as FUSION
}

package "Module 3: Generalizable Repr. Learning" <<m3>> as M3 {
  rectangle "Proxy Contrastive Loss\n─────────────────\nin:   zᵢ (sample),  wc (class proxy)\nf:    L_pcon = proxy-based contrastive\nout:  scalar loss" as PCON

  rectangle "Hàm mất mát tổng\n─────────────────\nf:    L = L_global + β₁·Σ Lⁱ_view + β₂·L_pcon\n      β₁=1,  β₂=0.05" as LOSS
}

rectangle "Classifier\n─────────────────\nin:   z_global ∈ ℝᴰ\nf:    FC + Softmax\nout:  ŷ ∈ ℝ|ᴬ|,  |A|=9 hoặc 6" as CLS

INPUT --> DENOISE
DENOISE --> TRANSFORM
TRANSFORM --> VIZ
VIZ --> INTRA : × V=6 views
INTRA --> INTER
INTER --> FUSION
FUSION --> PCON
PCON --> LOSS
FUSION --> CLS : z_global

@enduml
```

### Sơ đồ chi tiết Module 2 — Intra-view

```plantuml
@startuml UniFi_Intra
skinparam defaultFontSize 12
skinparam backgroundColor white
skinparam shadowing false
skinparam ArrowColor #555555
skinparam ArrowThickness 1.5

skinparam rectangle {
  BorderThickness 1.5
  RoundCorner 8
  BackgroundColor #CECBF6
  BorderColor #534AB7
  FontColor #26215C
}

title Intra-view Feature Extraction  (mỗi view i độc lập)

rectangle "Input  Xⁱ_t, Xⁱ_f\n─────────────────\nin:   Xⁱ_t ∈ ℝ³ˣ²²⁴ˣ²²⁴  (ảnh pha)\n      Xⁱ_f ∈ ℝ³ˣ²²⁴ˣ²²⁴  (ảnh DFS)" as IN

rectangle "ResNet18 Backbone (shared)\n─────────────────\nin:   Xⁱ_t,  Xⁱ_f\nf₁:   Fⁱ_t = ResNet18_t(Xⁱ_t; Θ_t)\nf₂:   Fⁱ_f = ResNet18_f(Xⁱ_f; Θ_f)\nout:  Fⁱ_t,  Fⁱ_f" as RESNET

note right of RESNET
  Θ_t, Θ_f dùng chung
  cho tất cả V views
end note

rectangle "Channel Attention\n─────────────────\nin:   Fⁱ_tf = [Fⁱ_t ; Fⁱ_f]  (concat kênh)\nf₁:   α = softmax( FC(Fⁱ_tf) )\nf₂:   Fⁱ_weighted = α ⊙ Fⁱ_tf\nout:  Fⁱ_weighted" as CHATTN

rectangle "Conv1D\n─────────────────\nin:   Fⁱ_weighted\nf:    Conv1D(Fⁱ_weighted)\nout:  Fⁱ_intra ∈ ℝᴰˣᵀ,  D=64" as CONV

rectangle "Output  Fⁱ_intra\n─────────────────\nout:  Fⁱ_intra ∈ ℝᴰˣᵀ  (T còn nguyên)" as OUT

IN --> RESNET
RESNET --> CHATTN
CHATTN --> CONV
CONV --> OUT

@enduml
```

### Sơ đồ chi tiết Module 2 — Inter-view Attention

```plantuml
@startuml UniFi_Inter
skinparam defaultFontSize 12
skinparam backgroundColor white
skinparam shadowing false
skinparam ArrowColor #555555
skinparam ArrowThickness 1.5

skinparam rectangle {
  BorderThickness 1.5
  RoundCorner 8
  BackgroundColor #CECBF6
  BorderColor #534AB7
  FontColor #26215C
}

title Inter-view Masked Cross-view Attention

rectangle "Input  F_intra\n─────────────────\nin:   Fⁱ_intra ∈ ℝᴰˣᵀ  × V views" as IN

rectangle "Stack + Temporal-aware Avg\n─────────────────\nin:   Fⁱ_intra × V\nf₁:   F_intra = stack(...)           →  ℝᵀˣᵛˣᴰ\nf₂:   F̃^{i,t} = Σⱼ wⱼ·F^{i,t+j}    →  F̃_intra ∈ ℝᵀˣᵛˣᴰ\n      wⱼ ∝ 1/(1+|j|)²  (normalized)\nout:  F_intra (dùng Q),  F̃_intra (dùng K, V)" as TAWA

note right of TAWA
  Tham số chưa rõ:
  · S — cửa sổ avg
  Mục đích: bù lệch thời gian nhỏ
  giữa các thiết bị
end note

rectangle "Spatial-aware View Mask\n─────────────────\nin:   V views + thông tin vị trí\nf:    chọn r trong m candidate views\n      theo hướng Fresnel bổ sung\nout:  Mask ∈ {0,1}ᵛˣᵛ" as MASK

note right of MASK
  Tham số chưa rõ:
  · m — số candidate views
  · r — số target views
end note

rectangle "Masked Cross-view Attention\n─────────────────\nin:   F_intra (Q),  F̃_intra (K, V),  Mask\nf₁:   Qₕ = F_intra · Wqₕ        ∈ ℝᵀˣᵛˣdₖ\nf₂:   Kₕ = F̃_intra · Wkₕ       ∈ ℝᵀˣᵛˣdₖ\nf₃:   Vₕ = F̃_intra · Wvₕ       ∈ ℝᵀˣᵛˣdᵥ\nf₄:   Hₕ = Softmax(Mask(QₕKₕᵀ/√D))·Vₕ\nf₅:   F_inter = Concat[H₁..Hₕ]·Wₒ\nout:  F_inter ∈ ℝᵀˣᵛˣᴰ" as ATTN

note right of ATTN
  Attention tính tại từng t độc lập
  Ma trận attn: V×V per timestep
  (không phải T×T)
  Tham số chưa rõ:
  · h — số heads  (dₖ=dᵥ=D/h=64/h)
end note

rectangle "Output  F_inter\n─────────────────\nout:  F_inter ∈ ℝᵀˣᵛˣᴰ" as OUT

IN --> TAWA
TAWA --> MASK
TAWA --> ATTN : Q (F_intra gốc)
MASK --> ATTN : K, V (F̃_intra có mask)
ATTN --> OUT

@enduml
```

### Sơ đồ chi tiết Module 2 — Feature Fusion

```plantuml
@startuml UniFi_Fusion
skinparam defaultFontSize 12
skinparam backgroundColor white
skinparam shadowing false
skinparam ArrowColor #555555
skinparam ArrowThickness 1.5

skinparam rectangle {
  BorderThickness 1.5
  RoundCorner 8
  BackgroundColor #CECBF6
  BorderColor #534AB7
  FontColor #26215C
}

title Feature Fusion — Segment-level + View-level

rectangle "Input  F_inter\n─────────────────\nin:   F_inter ∈ ℝᵀˣᵛˣᴰ\n      → tách per view: Fⁱ_inter ∈ ℝᵀˣᴰ" as IN

rectangle "Multi-scale Conv1D  (per view)\n─────────────────\nin:   Fⁱ_inter ∈ ℝᵀˣᴰ\nf:    pⁱₖ = GlobalAvgPool( Conv1D(Fⁱ_inter, kernel=pₖ) )\nout:  pⁱₖ ∈ ℝᴰ  (T biến mất, per scale k)" as MSCNN

note right of MSCNN
  Tham số chưa rõ:
  · Tập kernel size pₖ
  Kernel nhỏ: nắm đoạn ngắn
  Kernel lớn: nắm xu hướng dài
end note

rectangle "Segment-level Fusion\n─────────────────\nin:   pⁱₖ ∈ ℝᴰ  (per scale)\nf₁:   α^filter = softmax(ReLU(FC(concat[pⁱₖ])))\nf₂:   zⁱ = Σₖ α^filter_k · pⁱₖ\nout:  zⁱ ∈ ℝᴰ  (per view)" as SEGFUS

rectangle "View-level Fusion\n─────────────────\nin:   zⁱ ∈ ℝᴰ  × V views\nf₁:   α^view_i = softmax(ReLU(FC(zⁱ)))\nf₂:   z_global = Σᵢ α^view_i · zⁱ\nout:  z_global ∈ ℝᴰ" as VIEWFUS

rectangle "Output\n─────────────────\nout:  zⁱ ∈ ℝᴰ    → view-wise loss (training)\n      z_global ∈ ℝᴰ → classifier (inference)" as OUT

IN --> MSCNN
MSCNN --> SEGFUS
SEGFUS --> VIEWFUS : zⁱ
VIEWFUS --> OUT : z_global
SEGFUS --> OUT : zⁱ

@enduml
```

---

*Ghi chú: Widar 3.0 dataset — 15,375 mẫu, 9 cử chỉ, 3 phòng, 16 người, 5 vị trí, 5 hướng, 1 TX + 6 RX. Tham số chưa rõ: S (temporal window), m, r (spatial mask), h (attention heads), pₖ (kernel sizes), τ (temperature).*
