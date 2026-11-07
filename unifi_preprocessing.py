"""
UniFi — Module 1: Signal Preprocessing
=======================================
Liu et al., Proc. ACM IMWUT, vol. 7, no. 4, 2023.
DOI: 10.1145/3631429

Pipeline (chi tiết + nguồn tham khảo: unifi_architecture.md, mục 1a–1c):
    H [M, R, N, K, T] complex
        ↓ csi_ratio                       anten liền kề j1/j2
    H_r [K, T] complex
        ↓ extract_dfs                     trừ mean → HPF 2 Hz → PCA → STFT → |f| ≤ 60 Hz
        ↓ extract_phase                   ∠H_r (không unwrap)
    F_DFS [F=121, L] float,  T_pha [K=30, T] float
        ↓ to_rgb_image                    min-max toàn ảnh → parula
    X_f [3, F, L] float32,  X_t [3, K, T] float32
        ↓ resize (WiDarDataset)
    X_f, X_t [3, 224, 224]

Cài đặt:
    pip install numpy scipy matplotlib torch
"""

# =============================================================================
# Import
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import stft, butter, sosfilt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Bước 1 — CSI Ratio
# =============================================================================

def csi_ratio(H, rx_idx=0, ant1=0, ant2=1):
    """
    Chia tín hiệu hai anten cùng bộ thu để triệt nhiễu biên độ và lệch pha.

    Tham số:
        H      : ndarray complex [M, R, N, K, T]
                 M = số TX (thường = 1)
                 R = số bộ thu (1 file .dat = 1 bộ thu → R = 1)
                 N = số RX anten (Intel 5300 = 3)
                 K = số sub-carrier (= 30)
                 T = số timestep
        rx_idx : chỉ số bộ thu đang xử lý
        ant1   : chỉ số anten j1 (tử số)
        ant2   : chỉ số anten j2 (mẫu số)

    Trả về:
        H_r    : ndarray complex [K, T]
    """
    # H[0]          → [N, K, T]  bỏ chiều M (chỉ có 1 TX)
    # H[0, rx_idx]  → [K, T]    bỏ chiều N (chọn 1 RX)
    # Tương tự C: H[0][rx_idx][ant1][k][t]
    H_j1 = H[0, rx_idx, ant1, :, :]  # [K, T] complex
    H_j2 = H[0, rx_idx, ant2, :, :]  # [K, T] complex

    # Chia phức element-wise, +1e-8 để tránh chia 0
    H_r = H_j1 / (H_j2 + 1e-8)      # [K, T] complex

    return H_r


# =============================================================================
# Bước 2a — Extract DFS (Doppler Frequency Shift)
# =============================================================================

def extract_dfs(H_r, fs=1000, nperseg=256, noverlap=248,
                nfft=1000, f_max=60, highpass=False, hp_cutoff=2, hp_order=3):
    """
    Tính phổ Doppler theo thời gian bằng STFT.

    UniFi không báo cáo tham số STFT → chọn theo Widar3.0 (fs, ±60 Hz)
    và kiểm tra trực quan với Fig. DFS của UniFi.

    Tham số:
        H_r     : ndarray complex [K, T]
        fs      : sampling rate (Hz), Widar 3.0 dùng 1000
        nperseg : độ dài cửa sổ STFT (256 mẫu = 0.256 s)
        noverlap: số điểm chồng lấp giữa hai cửa sổ
                  hop = nperseg - noverlap = 8
        nfft    : số điểm FFT (zero-pad lên nfft trước FFT)
                  → phân giải tần số = fs / nfft = 1 Hz/bin
        f_max   : ngưỡng tần số Doppler giữ lại (Hz)
                  cử chỉ người tạo ra Doppler ~ ±20–60 Hz
        highpass: bật lọc Butterworth thông cao trước PCA
        hp_cutoff, hp_order : tần số cắt (Hz) và bậc bộ lọc
                  mặc định 2 Hz, bậc 3 như get_doppler_spectrum.m (Widar3.0)

    Trả về:
        F_DFS   : ndarray float [F, L]
                  F = 2*f_max + 1 = 121 bin trong [-f_max, f_max]
                  L = (T - nperseg) // hop + 1 cửa sổ thời gian
    """
    # ── Bỏ thành phần tĩnh ────────────────────────────────────────────
    # Đường truyền tĩnh (LoS, tường) là hằng số theo thời gian → 0 Hz.
    # Không trừ thì ~70% năng lượng nằm ở |f| ≤ 2 Hz và ảnh chỉ còn
    # một vạch ngang ở 0 Hz, mất hẳn vệt Doppler của cử chỉ.
    # keepdims=True: mean [K, 1] để broadcast với H_r [K, T]
    H_d = H_r - H_r.mean(axis=1, keepdims=True)    # [K, T]

    # ── Lọc thông cao (tùy chọn) ─────────────────────────────────────
    # Trừ mean chỉ bỏ phần tĩnh tuyệt đối; trôi chậm của kênh vẫn để lại
    # dải sáng ±5 Hz suốt mẫu, lấn át vệt Doppler khi min-max.
    # Lọc nhân quả (không filtfilt) giống filter() trong code Widar3.0.
    if highpass:
        sos = butter(hp_order, hp_cutoff, btype='high', fs=fs, output='sos')
        H_d = sosfilt(sos, H_d, axis=1)              # [K, T]

    # ── PCA: gộp K sub-carrier thành 1 chuỗi ─────────────────────────
    # Cử chỉ tác động tương quan lên các sub-carrier → thành phần chính
    # thứ nhất giữ phần chuyển động, bỏ bớt nhiễu (giống Widar3.0, WiFE).
    # Cho ảnh gần như giống hệt mean |STFT| của 30 sub nhưng chỉ cần
    # 1 lần STFT thay vì 30 → nhanh ~30 lần.
    # SVD của X [T, K]: hàng đầu của Vh = hướng chính [K]
    X = H_d.T                                       # [T, K]
    _, _, Vh = np.linalg.svd(X, full_matrices=False)
    signal = X @ Vh[0].conj()                       # [T] complex

    # ── STFT ──────────────────────────────────────────────────────────
    # return_onesided=False: giữ cả tần số âm (tín hiệu phức → hướng
    #                        lại gần / ra xa bộ thu)
    # boundary=None, padded=False: không pad 2 đầu, giống spectrogram
    #   của MATLAB → mẫu 1.47 s cho trục DFS ~1.2 s như Fig. của UniFi
    f, t, Zxx = stft(signal,
                     fs=fs,
                     window='hann',
                     nperseg=nperseg,
                     noverlap=noverlap,
                     nfft=nfft,
                     return_onesided=False,
                     boundary=None,
                     padded=False)

    # FFT output: [0, 1, ..., nfft/2, -nfft/2, ..., -1] Hz (lộn xộn)
    # fftshift sắp xếp lại về [-nfft/2, ..., 0, ..., nfft/2] Hz
    f   = np.fft.fftshift(f)             # [nfft]
    Zxx = np.fft.fftshift(Zxx, axes=0)  # [nfft, L]

    # Lọc chỉ giữ |f| <= f_max
    mask  = np.abs(f) <= f_max
    F_DFS = np.abs(Zxx[mask])            # [F, L] float

    return F_DFS


# =============================================================================
# Bước 2b — Extract Phase
# =============================================================================

def extract_phase(H_r):
    """
    Lấy góc pha của tín hiệu CSI ratio.

    Không unwrap: ảnh pha trong [12] (Gu et al., IEEE Sensors 2023)
    có pha nhảy ±π (dải vàng lẫn đốm xanh đậm) → dùng pha gốc.

    Tham số:
        H_r   : ndarray complex [K, T]

    Trả về:
        T_pha : ndarray float [K, T] trong [-π, π]
    """
    # np.angle(a + bj) = arctan(b / a) = atan2(b, a)
    T_pha = np.angle(H_r)  # [K, T]

    return T_pha


# =============================================================================
# Bước 3 — Visualize (Heatmap → Ảnh RGB)
# =============================================================================

# Colormap parula (MATLAB ≥ R2017a) — matplotlib không có sẵn.
# UniFi [13], [12] đều vẽ heatmap bằng parula (xanh tím → xanh lơ → vàng).
# 17 điểm mốc lấy mẫu trực tiếp từ colorbar Fig. DFS của bài UniFi.
PARULA = LinearSegmentedColormap.from_list('parula', [
    (0.243, 0.158, 0.691),   # 0.0000
    (0.271, 0.227, 0.860),   # 0.0625
    (0.280, 0.306, 0.947),   # 0.1250
    (0.261, 0.395, 0.987),   # 0.1875
    (0.189, 0.487, 0.983),   # 0.2500
    (0.168, 0.562, 0.938),   # 0.3125
    (0.125, 0.640, 0.887),   # 0.3750
    (0.051, 0.696, 0.831),   # 0.4375
    (0.066, 0.740, 0.724),   # 0.5000
    (0.203, 0.777, 0.603),   # 0.5625
    (0.344, 0.796, 0.472),   # 0.6250
    (0.560, 0.792, 0.299),   # 0.6875
    (0.753, 0.759, 0.171),   # 0.7500
    (0.921, 0.727, 0.197),   # 0.8125
    (0.992, 0.775, 0.205),   # 0.8750
    (0.967, 0.863, 0.157),   # 0.9375
    (0.969, 0.963, 0.101),   # 1.0000
], N=256)


def to_rgb_image(data_2d, colormap='parula'):
    """
    Chuyển ma trận 2D thành ảnh RGB bằng colormap.
    Tương đương imagesc của MATLAB: min-max tuyến tính trên TOÀN ảnh.

    Tham số:
        data_2d  : ndarray float [H, W]
        colormap : 'parula' (mặc định, giống bài) hoặc tên colormap
                   matplotlib khác ('jet', 'viridis', ...)

    Trả về:
        rgb      : ndarray float32 [3, H, W] trong [0, 1]
    """
    # --- Normalize về [0, 1] ---
    d_min = data_2d.min()
    d_max = data_2d.max()
    diff  = d_max - d_min

    if diff < 1e-8:
        # Toàn bộ giá trị bằng nhau → ảnh đen
        normalized = np.zeros_like(data_2d, dtype=np.float32)
    else:
        normalized = (data_2d - d_min) / diff  # [H, W] trong [0, 1]

    # --- Áp colormap ---
    # plt.get_cmap(name) trả về object colormap
    # cmap(array) → [H, W, 4] (RGBA, mỗi giá trị trong [0,1])
    cmap = PARULA if colormap == 'parula' else plt.get_cmap(colormap)
    rgba = cmap(normalized)        # [H, W, 4] float64

    # Bỏ kênh Alpha (index 3), giữ RGB (index 0,1,2)
    rgb = rgba[:, :, :3]           # [H, W, 3]

    # Chuyển [H, W, 3] → [3, H, W] (PyTorch convention)
    # transpose(2, 0, 1): trục 2→0, trục 0→1, trục 1→2
    rgb = rgb.transpose(2, 0, 1)   # [3, H, W]
    rgb = rgb.astype(np.float32)

    return rgb


# =============================================================================
# Pipeline hoàn chỉnh cho một view
# =============================================================================

def normalize01(data_2d):
    """Min-max trên toàn ảnh về [0, 1] (ảnh hằng → 0). Giống bước đầu của to_rgb_image."""
    d_min, d_max = data_2d.min(), data_2d.max()
    if d_max - d_min < 1e-8:
        return np.zeros_like(data_2d, dtype=np.float32)
    return ((data_2d - d_min) / (d_max - d_min)).astype(np.float32)


def extract_dfs_amp(A, fs=200, nperseg=64, noverlap=60, nfft=200, f_max=60,
                    highpass=False, hp_cutoff=2, hp_order=3):
    """
    Phổ Doppler từ BIÊN ĐỘ CSI (thiết bị chỉ có |H|, 1 anten → không có CSI ratio).

    Giống extract_dfs (trừ mean → HPF → PCA → STFT) nhưng tín hiệu thực:
    dao động biên độ chỉ cho |f_D| (không phân biệt lại gần / ra xa) → phổ
    đối xứng, chỉ giữ nửa dương 0..f_max (cách làm của CARM).

    A : ndarray float [K, T] → F_DFS [f_max * nfft / fs + 1, L]
    """
    A_d = A - A.mean(axis=1, keepdims=True)
    if highpass:
        sos = butter(hp_order, hp_cutoff, btype='high', fs=fs, output='sos')
        A_d = sosfilt(sos, A_d, axis=1)
    X = A_d.T                                       # [T, K]
    _, _, Vh = np.linalg.svd(X, full_matrices=False)
    signal = X @ Vh[0]                              # [T] thực
    f, _, Zxx = stft(signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
                     nfft=nfft, return_onesided=True, boundary=None, padded=False)
    return np.abs(Zxx[f <= f_max])                  # [F, L]


def compute_maps(H_r, fs=1000, nperseg=256, noverlap=248,
                 nfft=1000, f_max=60, highpass=False, signal='csi_ratio'):
    """
    Phần tốn CPU của preprocessing (HPF + PCA + STFT), không phụ thuộc colormap.
    Kết quả đã min-max về [0, 1] → có thể cache (xem unifi_cache.py).

    signal = 'csi_ratio' (Widar, H_r complex [K, T]): ảnh DFS + ảnh pha ∠H_r
    signal = 'amp'       (SSHAR ESP, H_r = |H| float [K, T]): ảnh DFS từ biên độ
                         + ảnh biên độ [K, T] thay cho ảnh pha (không có pha)

    Trả về:
        dfs01 : float32 [F, L] trong [0, 1]
        pha01 : float32 [K, T] trong [0, 1]   (ảnh pha hoặc ảnh biên độ)
    """
    if signal == 'amp':
        A = np.asarray(H_r, dtype=np.float64)
        F_DFS = extract_dfs_amp(A, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                nfft=nfft, f_max=f_max, highpass=highpass)
        return normalize01(F_DFS), normalize01(A)
    if signal != 'csi_ratio':
        raise ValueError(f"signal không hợp lệ: {signal!r}")
    F_DFS = extract_dfs(H_r, fs=fs, nperseg=nperseg, noverlap=noverlap,
                        nfft=nfft, f_max=f_max, highpass=highpass)   # [F, L]
    T_pha = extract_phase(H_r)                                       # [K, T]
    return normalize01(F_DFS), normalize01(T_pha)


def maps_to_tensors(dfs01, pha01, colormap='parula', img_size=(224, 224)):
    """
    Colormap + resize 2 bản đồ [0, 1] của một view → tensor đầu vào mô hình.

    Trả về:
        X_f, X_t : Tensor float32 [3, H, W]
    """
    X_f = torch.from_numpy(to_rgb_image(dfs01, colormap))[None]   # [1, 3, F, L]
    X_t = torch.from_numpy(to_rgb_image(pha01, colormap))[None]   # [1, 3, K, T]
    # antialias=True: trục T của ảnh pha bị thu nhỏ ~5–9 lần
    # (T ~1000–2000 → 224); bilinear thường chỉ lấy 4 điểm lân cận
    # → bỏ sót mẫu, răng cưa ở các chỗ pha nhảy ±π. Antialias lọc
    # thông thấp trước khi thu nhỏ, giống imresize (MATLAB) / PIL
    # mà WiGRUNT dùng.
    X_f = F.interpolate(X_f, size=img_size, mode='bilinear',
                        align_corners=False, antialias=True)[0]
    X_t = F.interpolate(X_t, size=img_size, mode='bilinear',
                        align_corners=False, antialias=True)[0]
    return X_f, X_t


def preprocess_one_view(H_r,
                        fs=1000,
                        nperseg=256,
                        noverlap=248,
                        nfft=1000,
                        f_max=60,
                        colormap='parula',
                        highpass=False):
    """
    Xử lý toàn bộ preprocessing cho một view (một bộ thu).

    Tham số:
        H_r      : ndarray complex [K, T]

    Trả về:
        X_f      : ndarray float32 [3, F, L]  ảnh DFS
        X_t      : ndarray float32 [3, K, T]  ảnh pha
    """
    dfs01, pha01 = compute_maps(H_r, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                nfft=nfft, f_max=f_max, highpass=highpass)
    X_f = to_rgb_image(dfs01, colormap)  # [3, F, L]
    X_t = to_rgb_image(pha01, colormap)  # [3, K, T]

    return X_f, X_t


def preprocess_all_views(H_all_views,
                         fs=1000,
                         nperseg=256,
                         noverlap=248,
                         nfft=1000,
                         f_max=60,
                         colormap='parula',
                         highpass=False):
    """
    Xử lý tất cả V views của một mẫu.

    Tham số:
        H_all_views : list V phần tử
                      mỗi phần tử là ndarray complex [K, T]

    Trả về:
        X_f_views   : ndarray float32 [V, 3, F, L]
        X_t_views   : ndarray float32 [V, 3, K, T]
    """
    X_f_list = []
    X_t_list = []

    for H_r in H_all_views:
        X_f, X_t = preprocess_one_view(H_r,
                                       fs=fs,
                                       nperseg=nperseg,
                                       noverlap=noverlap,
                                       nfft=nfft,
                                       f_max=f_max,
                                       colormap=colormap,
                                       highpass=highpass)
        X_f_list.append(X_f)  # [3, F, L]
        X_t_list.append(X_t)  # [3, K, T]

    X_f_views = np.stack(X_f_list, axis=0)  # [V, 3, F, L]
    X_t_views = np.stack(X_t_list, axis=0)  # [V, 3, K, T]

    return X_f_views, X_t_views


# =============================================================================
# Dataset class cho PyTorch
# =============================================================================

class WiDarDataset(Dataset):
    """
    PyTorch Dataset cho Widar 3.0.

    PyTorch yêu cầu implement 3 hàm:
        __init__    : khởi tạo, nhận data và label
        __len__     : trả về số mẫu
        __getitem__ : trả về 1 mẫu theo index
    """

    def __init__(self, data_list, label_list,
                 fs=1000, nperseg=256, noverlap=248,
                 nfft=1000, f_max=60, colormap='parula',
                 img_size=(224, 224), highpass=False, signal='csi_ratio'):
        """
        Tham số:
            data_list  : list N phần tử
                         mỗi phần tử là list V mảng [K, T] complex
            label_list : list N nhãn (int), bắt đầu từ 0
            img_size   : (H, W) resize ảnh về cùng kích thước
                         vì T (và L) khác nhau giữa các mẫu
                         → DataLoader không stack được nếu không resize
        """
        self.data_list  = data_list
        self.label_list = label_list
        self.img_size   = img_size

        # Lưu tham số preprocessing
        self.fs       = fs
        self.nperseg  = nperseg
        self.noverlap = noverlap
        self.nfft     = nfft
        self.f_max    = f_max
        self.colormap = colormap
        self.highpass = highpass
        self.signal   = signal

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Trả về 1 mẫu.
        PyTorch tự gọi hàm này khi lấy batch từ DataLoader.
        """
        H_views = self.data_list[idx]   # list V mảng [K, T]
        label   = self.label_list[idx]  # int

        # Các bộ thu của cùng 1 mẫu cũng có T khác nhau
        # → resize từng view về img_size rồi mới stack
        X_f_list = []
        X_t_list = []
        for H_r in H_views:
            dfs01, pha01 = compute_maps(
                H_r,
                fs=self.fs,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                f_max=self.f_max,
                highpass=self.highpass,
                signal=self.signal
            )
            X_f, X_t = maps_to_tensors(dfs01, pha01, self.colormap, self.img_size)
            # X_f, X_t: [3, H, W]

            X_f_list.append(X_f)
            X_t_list.append(X_t)

        X_f = torch.stack(X_f_list, dim=0)   # [V, 3, H, W]
        X_t = torch.stack(X_t_list, dim=0)   # [V, 3, H, W]

        y   = torch.tensor(label, dtype=torch.long)    # scalar

        return X_f, X_t, y


# =============================================================================
# Test và visualize
# =============================================================================

def test_preprocessing(V=6, K=30, T=1200):
    """
    Chạy thử toàn bộ pipeline với dữ liệu giả.
    Gọi hàm này để kiểm tra code chạy đúng không.
    """
    print("=== Test preprocessing ===")

    # Tạo dữ liệu giả V views
    H_views = []
    for _ in range(V):
        H_r = (np.random.randn(K, T)
               + 1j * np.random.randn(K, T)).astype(np.complex64)
        H_views.append(H_r)

    # Chạy preprocessing
    X_f_views, X_t_views = preprocess_all_views(H_views)

    print(f"X_f_views shape: {X_f_views.shape}")
    # Kỳ vọng: (6, 3, 121, 119)
    # F = 2*f_max+1 = 121, L = (1200-256)//8 + 1 = 119 (không pad)

    print(f"X_t_views shape: {X_t_views.shape}")
    # Kỳ vọng: (6, 3, 30, 1200)

    print(f"X_f dtype: {X_f_views.dtype}")   # float32
    print(f"X_f min/max: {X_f_views.min():.3f} / {X_f_views.max():.3f}")
    # Kỳ vọng: trong [0, 1] (parula: ~0.05 / ~0.99, không chạm 0 hay 1)


def visualize_one_view(H_r, fs=1000, nperseg=256,
                       noverlap=248, nfft=1000, f_max=60):
    """
    Hiển thị ảnh DFS và pha của một view.
    Dùng để kiểm tra trực quan.
    """
    X_f, X_t = preprocess_one_view(H_r, fs=fs,
                                   nperseg=nperseg,
                                   noverlap=noverlap,
                                   nfft=nfft,
                                   f_max=f_max)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # X_f: [3, F, L] → transpose về [F, L, 3] để plt.imshow
    axes[0].imshow(X_f.transpose(1, 2, 0), aspect='auto',
                   origin='lower')
    axes[0].set_title('DFS image  [F × L]')
    axes[0].set_xlabel('Time bins (L)')
    axes[0].set_ylabel(f'Doppler bins (F, ±{f_max} Hz)')

    # X_t: [3, K, T] → transpose về [K, T, 3]
    axes[1].imshow(X_t.transpose(1, 2, 0), aspect='auto',
                   origin='lower')
    axes[1].set_title('Phase image  [K × T]')
    axes[1].set_xlabel('Time (T)')
    axes[1].set_ylabel('Sub-carrier (K=30)')

    plt.tight_layout()
    plt.show()


# =============================================================================
# Chạy thử khi chạy file trực tiếp
# =============================================================================

if __name__ == '__main__':
    # Chỉ chạy khi gọi: python unifi_preprocessing.py
    # Không chạy khi file được import từ file khác

    test_preprocessing()

    # Visualize một view giả
    K, T = 30, 1200
    H_r_fake = (np.random.randn(K, T)
                + 1j * np.random.randn(K, T)).astype(np.complex64)
    visualize_one_view(H_r_fake)