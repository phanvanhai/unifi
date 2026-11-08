# main.py
"""
Train UniFi trên Widar 3.0.

Chạy (local hoặc Kaggle):
    python main.py --data-root /kaggle/input/widar3/widar --output-dir /kaggle/working
    python main.py --split location --test-values 3 --epochs 100
    python main.py --max-samples 120 --epochs 1          # chạy thử nhanh
    python main.py --help                                 # xem mọi tuỳ chọn

Hoặc từ notebook (sửa biến module rồi gọi main()):
    import main
    main.SPLIT_BY, main.TEST_VALUES = 'location', [3]
    main.main()

Cấu trúc dữ liệu:
    widar/<user>/<user>-<gesture>-<location>-<orientation>-<repetition>-r<rx>.dat
    ví dụ: widar/user3/user3-1-1-1-1-r1.dat

    1 mẫu = 6 file r1..r6 (6 bộ thu = 6 views)
    Mỗi file: CSI Intel 5300, 1 TX × 3 anten × 30 sub-carrier
"""
import os
import sys
import csv
import json
import math
import time
import random
import re
import shutil
import logging
import argparse
import platform
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from unifi_preprocessing import WiDarDataset, csi_ratio
from unifi_inter_view    import build_spatial_mask
from unifi_model         import UniFi
from unifi_loss_train    import train
from unifi_cache         import build_or_load_cache, CachedWiDarDataset

# In tiếng Việt không lỗi trên console Windows (cp1252).
# Trong Jupyter/Kaggle notebook sys.stdout không có reconfigure → bỏ qua.
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

log = logging.getLogger('unifi')


# =============================================================================
# Môi trường: local / Kaggle
# =============================================================================

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or Path('/kaggle/input').is_dir()

# Thư mục chứa code (khi chạy từng cell trong notebook không có __file__)
CODE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

# Kết quả (checkpoint, hình) — /kaggle/input chỉ đọc nên phải ghi vào /kaggle/working
OUTPUT_DIR = Path(os.environ.get('UNIFI_OUTPUT_DIR',
                                 '/kaggle/working' if IS_KAGGLE else CODE_DIR))


def find_data_root():
    """
    Tìm thư mục 'widar' theo thứ tự:
      1. biến môi trường UNIFI_DATA_ROOT
      2. <CODE_DIR>/widar           (local, hoặc upload cả folder unifi lên Kaggle)
      3. thư mục tên 'widar' bất kỳ trong /kaggle/input (sâu tối đa 5 cấp)
      4. thư mục cha chung của mọi thư mục userX/*.dat trong /kaggle/input
         (trường hợp upload nội dung widar/ làm dataset)
    (--data-root trên dòng lệnh luôn được ưu tiên hơn cả 4 cách trên.)
    """
    env = os.environ.get('UNIFI_DATA_ROOT')
    if env:
        return Path(env)

    local = CODE_DIR / 'widar'
    if local.is_dir():
        return local

    kin = Path('/kaggle/input')
    if kin.is_dir():
        user_dirs = []
        for depth in range(1, 7):
            for p in sorted(kin.glob('/'.join(['*'] * depth))):
                if p.is_dir() and p.name == 'widar':
                    return p
                if p.is_dir() and p.name.startswith('user') and next(p.glob('*.dat'), None):
                    user_dirs.append(str(p))
        if user_dirs:
            # thư mục cha chung của mọi userX (kể cả khi lồng trong thư mục ngày thu)
            return Path(os.path.commonpath(user_dirs))

    return local   # không tìm thấy → để scan_samples báo lỗi rõ ràng


# =============================================================================
# Cấu hình (giá trị mặc định — ghi đè được bằng tham số dòng lệnh)
# =============================================================================

# Dataset: 'widar' (Widar 3.0, mặc định) | 'sshar' (SSHAR ESP, theo SSHAR_ESP_protocol.md)
DATASET   = 'widar'
DATA_ROOT = find_data_root()
USERS     = ['user5', 'user10', 'user11', 'user12', 'user13', 'user14', 'user15', 'user16', 'user17']
GESTURES  = [1, 2, 3, 4, 5, 6] # gesture id trong tên file (nhãn = vị trí trong list)
V         = 6                  # số bộ thu

# Tên cử chỉ để ghi vào log / vẽ confusion matrix (thiếu → 'g<id>').
# Lưu ý: ở Widar 3.0 ý nghĩa gesture id khác nhau theo ngày thu — điền theo đúng ngày.
GESTURE_NAMES = {1: 'Push&Pull', 2: 'Sweep', 3: 'Clap', 4: 'Slide', 5: 'Draw-O', 6: 'Draw-Zigzag'}

# Chia 2 tập train/test (như UniFi, WiGRUNT). Checkpoint lấy theo test acc cao nhất.
#   'random'      : chia ngẫu nhiên theo TEST_RATIO (in-domain)
#   'user'        : TEST_VALUES là list user để test   (cross-user)
#   'location'    : TEST_VALUES là list vị trí torso   (cross-location)
#   'orientation' : TEST_VALUES là list hướng mặt      (cross-orientation)
SPLIT_BY    = 'random'
TEST_VALUES = []
TEST_RATIO  = 0.2
# Widar: val = VAL_RATIO của phần train+val, mỗi lớp (UniFi mục 6.1 lấy 80% làm "training and
# validation set" nhưng không nêu tỉ lệ val) → 72 / 8 / 20. 0 = không val, chọn theo test.
# SSHAR không dùng biến này: val theo SSHAR_ESP_protocol.md.
VAL_RATIO   = 0.1

# Tọa độ (m) theo sơ đồ Widar 3.0: gốc tại Tx, x sang trái, y xuống dưới
# Vùng cảm biến 2m × 2m: x ∈ [0, 2], y ∈ [0, 2]
RX_POS = [(0.5, -0.5),   # Rx1
          (1.4, -0.5),   # Rx2
          (2.0,  0.0),   # Rx3
          (-0.5, 0.5),   # Rx4
          (-0.5, 1.4),   # Rx5
          (0.0,  2.0)]   # Rx6
CENTER = (1.0, 1.0)      # Loc5 — tâm vùng cảm biến

# Góc nhìn từng bộ thu từ tâm (radian) cho spatial mask
# Chỉ thứ tự độ lệch góc là quan trọng → dùng tâm cho mọi location
# (đã kiểm tra: cặp gần hướng nhất gần như không đổi giữa 5 location)
# Đặt None → không dùng mask (mọi view attend lẫn nhau)
RX_DIRECTIONS = [math.atan2(y - CENTER[1], x - CENTER[0]) for x, y in RX_POS]

# ── SSHAR ESP (SSHAR_ESP_protocol.md) ──────────────────────────────────────
# ESP32: 3 receiver x 1 anten, 56 subcarrier, chỉ có BIÊN ĐỘ, 1000 điểm = 5 s @ 200 Hz.
# Không có CSI ratio / pha → nhánh "pha" của UniFi nhận ảnh biên độ [56, T], nhánh DFS
# tính từ biên độ (unifi_preprocessing.extract_dfs_amp). Chưa có toạ độ receiver →
# không dùng spatial mask. Chia 7:1:2 theo lần lặp, chọn checkpoint theo val acc.
SSHAR_DEVICE   = 'esp'
SSHAR_RX       = ['rx_00', 'rx_01', 'rx_02']
SSHAR_SUBJECTS = ['subject_01', 'subject_02', 'subject_03', 'subject_04', 'subject_09',
                  'subject_10', 'subject_11', 'subject_12', 'subject_13', 'subject_14']
SSHAR_ACTS     = [1, 2, 3, 4, 5, 6, 7, 8]
SSHAR_EXPECTED = {'train': 4480, 'val': 640, 'test': 1280}   # đủ 10 người x 8 hành động (cả 2 kiểu chia)
# Cross-user (--split user --user_case N), kaggle_run_convention.md mục 4: train+val = USER_CASES["sshar"][N]
# của CodeForPaper/benchmark/datasets.py (8 người), test = 2 người còn lại. Val: case 0 = người cuối (12);
# case 1..4 chọn bằng random.Random(42), lần lượt, không trùng val của các case trước.
SSHAR_USER_CASES = [   # (train, val, test) theo số subject
    ([1, 2, 3, 4, 9, 10, 11],    [12], [13, 14]),
    ([1, 3, 4, 9, 11, 12, 14],   [13], [2, 10]),
    ([2, 4, 9, 10, 12, 13, 14],  [1],  [3, 11]),
    ([3, 4, 10, 11, 12, 13, 14], [2],  [1, 9]),
    ([1, 2, 4, 9, 11, 13, 14],   [10], [3, 12]),
]
USER_CASE      = 0
MODEL_NAME     = 'UniFi'   # thư mục results/<RUN>/<MODEL_NAME>/
SSHAR_PATTERN  = re.compile(r'act(\d+)_pos(\d+)_dir(\d+)_rep(\d+)')


def find_sshar_root():
    """Thư mục chứa room_XX/<device>/rx_YY/... (vd. sshar_amp_interp)."""
    env = os.environ.get('UNIFI_SSHAR_ROOT')
    if env:
        return Path(env)
    cands = [CODE_DIR / 'sshar_amp_interp',
             CODE_DIR.parent.parent / 'CodeForPaper' / 'benchmark' / 'data' / 'sshar_amp_interp']
    kin = Path('/kaggle/input')
    if kin.is_dir():
        for depth in range(0, 7):
            for p in sorted(kin.glob('/'.join(['*'] * depth + ['room_*', SSHAR_DEVICE]))):
                cands.append(p.parent.parent)
    for c in cands:
        if c.is_dir() and any(c.glob(f'room_*/{SSHAR_DEVICE}')):
            return c
    return cands[0]


# Siêu tham số (theo bài báo)
D          = 64
BATCH_SIZE = 32
NUM_EPOCHS = 100
LR         = 0.005
BETA1      = 1.0
BETA2      = 0.05

# Không có trong bài báo — mặc định tắt / theo bài
WEIGHT_DECAY = 0.0
GRAD_CLIP    = None       # ví dụ 5.0 nếu loss bị NaN

NUM_WORKERS = min(4, os.cpu_count() or 1)   # đọc dữ liệu song song (Kaggle: 4 CPU)
SEED        = 42

# Số packet tối thiểu: STFT cần T >= nperseg = 256
MIN_PACKETS = 256

# Lọc thông cao 2 Hz trước PCA khi tính DFS (như Widar3.0)
DFS_HIGHPASS = True

# Tốc độ / môi trường
AMP          = True       # mixed precision fp16 trên GPU (nhanh ~2×, ít VRAM)
# DataParallel (2 GPU): leak ~0.9 MB RAM/batch ở tiến trình chính (~0.2 GB/epoch, đo trên
# Kaggle T4 x2) và còn chậm hơn 1 GPU (31 vs 50 mẫu/s, model nhỏ + batch 32) → mặc định tắt
MULTI_GPU    = False
USE_CACHE    = True       # cache bản đồ DFS/pha (xem unifi_cache.py)
CACHE_DIR    = Path('/tmp/unifi_cache') if IS_KAGGLE else CODE_DIR / 'cache'
# Nơi tìm cache build sẵn (vd. Kaggle Dataset tạo từ output của --cache-only)
CACHE_SEARCH = [Path('/kaggle/input')] if IS_KAGGLE else []
CACHE_ONLY   = False      # chỉ build cache rồi thoát (không train)
PIN_MEMORY   = False      # tắt: nghi gây leak RAM ~0.2 GB/epoch ở tiến trình chính (DataParallel + AMP)
MAX_SAMPLES  = None       # giới hạn số mẫu (chạy thử nhanh), None = tất cả
TIME_LIMIT_H = 11.5 if IS_KAGGLE else None   # Kaggle cắt phiên ở 12 giờ
# Early stopping theo val acc (không có trong bài UniFi): dừng khi PATIENCE epoch liên tiếp
# không có val acc mới cao nhất; checkpoint tốt nhất vẫn là epoch có val acc cao nhất. 0 = tắt
PATIENCE     = 10
LOG_INTERVAL = 20         # in tiến độ mỗi N batch (0 = tắt)
RUN_NAME     = None       # tên thư mục run (None → <thời gian>_<kiểu chia>)
RUN_DIR      = None       # đặt sẵn thư mục run (chế độ --results_dir), bỏ qua OUTPUT_DIR/runs/RUN_NAME
MAKE_PLOTS   = True       # vẽ hình (plot_logs.py) khi train xong
ZIP_RUN      = IS_KAGGLE  # nén thư mục run thành .zip để tải về 1 lần


# =============================================================================
# Đọc file .dat Intel 5300 (port từ read_bf_file.m / read_bfee.c)
# =============================================================================

def read_bf_file(path):
    """
    Đọc CSI từ file .dat của Linux 802.11n CSI Tool (Intel 5300).

    Trả về:
        csi : ndarray complex64 [T, 30, Nrx]   (chỉ lấy TX 0)
              T = số packet hợp lệ
    """
    raw = Path(path).read_bytes()

    payloads = []   # list bytes payload [192] của packet hợp lệ
    ant_sels = []   # antenna_sel của từng packet

    pos = 0
    n   = len(raw)
    while pos + 3 <= n:
        # Mỗi record: 2 byte độ dài (big-endian) + 1 byte code
        field_len = int.from_bytes(raw[pos:pos + 2], 'big')
        code      = raw[pos + 2]
        start     = pos + 3
        pos       = start + field_len - 1
        if pos > n:
            break                    # record cuối bị cắt dở

        if code != 187:              # 187 = 0xBB = beamforming (CSI)
            continue

        b   = raw[start:pos]
        Nrx = b[8]
        Ntx = b[9]
        if Nrx != 3 or Ntx != 1:
            continue                 # Widar: 1 TX, 3 anten; bỏ packet lạ

        csi_len  = int.from_bytes(b[16:18], 'little')
        calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 7) // 8
        if csi_len != calc_len or len(b) < 20 + calc_len:
            continue                 # packet hỏng

        payloads.append(np.frombuffer(b, dtype=np.uint8, count=calc_len, offset=20))
        ant_sels.append(b[15])

    if not payloads:
        return np.zeros((0, 30, 3), dtype=np.complex64)

    P  = np.stack(payloads).astype(np.uint16)  # [T, 192]
    NN = 3                                     # Nrx * Ntx

    # Vị trí bit của (sub-carrier i, anten j):
    #   mỗi sub-carrier: bỏ 3 bit, rồi NN cặp (real, imag) 8 bit
    i = np.arange(30)[:, None]
    j = np.arange(NN)[None, :]
    bit  = 3 * (i + 1) + 16 * (i * NN + j)     # [30, NN]
    byte = bit // 8
    rem  = (bit % 8).astype(np.uint16)

    def take(k):
        # Ghép 2 byte liên tiếp rồi dịch bit → 8 bit có dấu
        v = (P[:, byte + k] >> rem) | (P[:, byte + k + 1] << (8 - rem))
        return (v & 0xFF).astype(np.uint8).view(np.int8).astype(np.float32)

    csi = (take(0) + 1j * take(1)).astype(np.complex64)   # [T, 30, NN]

    # Hoán vị anten theo antenna_sel (giống read_bf_file.m)
    # read_bf_file.m xét TỪNG packet: perm hợp lệ → hoán vị, không hợp lệ → giữ nguyên.
    ant_sel = np.array(ant_sels, dtype=np.uint8)
    perm    = np.stack([(ant_sel >> (2 * k)) & 0x3 for k in range(3)], axis=1)  # [T, 3]
    valid   = (np.sort(perm, axis=1) == np.arange(3)).all(axis=1)              # [T]
    perm[~valid] = np.arange(3)                                                 # giữ nguyên
    out = np.empty_like(csi)
    np.put_along_axis(out, np.broadcast_to(perm[:, None, :], csi.shape).astype(np.int64),
                      csi, axis=2)

    return out   # [T, 30, 3]


def load_view(path):
    """
    1 file của 1 receiver:
      .dat (Widar)  → H_r [K=30, T] complex (CSI ratio 2 anten)
      .npy (SSHAR)  → biên độ [anten * K, T] float32 (ESP: [56, 1000])
    """
    if str(path).endswith('.npy'):
        a = np.load(path).astype(np.float32)          # (anten, K, T)
        return a.reshape(-1, a.shape[-1])
    csi = read_bf_file(path)                 # [T, 30, 3]
    H   = csi.transpose(2, 1, 0)[None, None] # [M=1, R=1, N=3, K=30, T]
    return csi_ratio(H, rx_idx=0, ant1=0, ant2=1)


class LazyCSIList:
    """
    Đóng vai trò data_list cho WiDarDataset nhưng đọc file khi cần.
    Toàn bộ CSI quá lớn để nạp hết vào RAM.

    self[idx] → list V mảng [K, T] complex
    """

    def __init__(self, samples):
        self.samples = samples   # list N phần tử, mỗi phần tử list V path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return [load_view(p) for p in self.samples[idx]]


# =============================================================================
# Liệt kê mẫu + chia train/test
# =============================================================================

def find_user_dir(user):
    """
    widar/<user> hoặc lồng sâu hơn, ví dụ widar/20181130_user15_16_17/user15.
    Lấy thư mục đầu tiên tên <user> có chứa file .dat.
    """
    direct = DATA_ROOT / user
    if direct.is_dir():
        return direct
    for p in sorted(DATA_ROOT.rglob(user)):
        if p.is_dir() and any(p.glob(f'{user}-*.dat')):
            return p
    return None


def scan_samples():
    """
    Trả về list dict: {'paths', 'label', 'user', 'location', 'orientation', 'id'}
    Bỏ mẫu thiếu bộ thu hoặc có file quá ngắn.
    """
    # Ước lượng kích thước tối thiểu: 1 packet 3x1 = 3 + 212 byte
    min_bytes = MIN_PACKETS * 215

    samples, skipped = [], []
    for user in USERS:
        user_dir = find_user_dir(user)
        if user_dir is None:
            log.warning(f"Không thấy thư mục {user} trong {DATA_ROOT}")
            continue

        n_user = 0
        for f in sorted(user_dir.glob(f'{user}-*-r1.dat')):
            # user3-1-1-1-1-r1 → [user3, 1, 1, 1, 1]
            parts = f.stem.split('-')[:-1]
            try:
                gesture, location, orientation = int(parts[1]), int(parts[2]), int(parts[3])
            except (IndexError, ValueError):
                skipped.append((f.stem, 'tên file lạ'))
                continue
            if gesture not in GESTURES:
                continue

            base  = '-'.join(parts)
            paths = [user_dir / f'{base}-r{r}.dat' for r in range(1, V + 1)]
            bad   = [p.name for p in paths if not p.exists() or p.stat().st_size < min_bytes]
            if bad:
                skipped.append((base, 'thiếu/ngắn: ' + ','.join(bad)))
                continue

            samples.append({
                'id'         : base,
                'paths'      : [str(p) for p in paths],
                'label'      : GESTURES.index(gesture),
                'user'       : user,
                'location'   : location,
                'orientation': orientation,
            })
            n_user += 1
        log.info(f"  {user:>7}: {n_user:5d} mẫu  ({user_dir})")

    log.info(f"Tìm thấy {len(samples)} mẫu, bỏ {len(skipped)} mẫu lỗi/thiếu file")
    for s, why in skipped:
        log.info(f"  - bỏ {s} ({why})")
    return samples


def sshar_split_of(pos, direction, rep):
    """SSHAR_ESP_protocol.md mục 3 (chép nguyên văn bản tham chiếu)."""
    if direction == 0:                      # 10 lần lặp
        return 'test' if rep > 8 else 'val' if rep == 8 else 'train'
    if rep == 5:                            # dir01 / dir02, 5 lần lặp
        return 'test'
    val_dir = 1 if pos % 2 else 2           # pos lẻ -> dir01, pos chẵn -> dir02
    return 'val' if rep == 4 and direction == val_dir else 'train'


def scan_sshar():
    """
    <root>/room_XX/<device>/rx_YY/subject_ZZ/amp_actAA_posBB_dirCC_repDD.npy
    Mỗi mẫu = file cùng tên ở mọi receiver trong SSHAR_RX (thiếu 1 receiver → lỗi).
    """
    root, samples = Path(DATA_ROOT), []
    for room in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('room')):
        first = room / SSHAR_DEVICE / SSHAR_RX[0]
        if not first.is_dir():
            continue
        for subj in sorted(p for p in first.iterdir() if p.is_dir() and p.name in USERS):
            n_subj = 0
            for f in sorted(subj.glob('*.npy')):
                m = SSHAR_PATTERN.search(f.name)
                if m is None:
                    continue
                act, pos, direction, rep = map(int, m.groups())
                if act not in GESTURES:
                    continue
                paths = [room / SSHAR_DEVICE / rx / subj.name / f.name for rx in SSHAR_RX]
                miss = [str(q) for q in paths if not q.is_file()]
                if miss:
                    raise FileNotFoundError(f"{f.name} thiếu ở receiver: {miss}")
                samples.append({
                    'id'         : f"{room.name}/{subj.name}/{f.stem}",
                    'paths'      : [str(q) for q in paths],
                    'label'      : GESTURES.index(act),
                    'user'       : subj.name,
                    'location'   : pos,
                    'orientation': direction,
                    'rep'        : rep,
                    'split'      : sshar_split_of(pos, direction, rep),
                })
                n_subj += 1
            log.info(f"  {room.name}/{subj.name}: {n_subj:5d} mẫu")
    log.info(f"Tìm thấy {len(samples)} mẫu SSHAR ({SSHAR_DEVICE}, {len(SSHAR_RX)} receiver)")
    return samples


def subsample(samples, n, seed):
    """Lấy n mẫu chia đều theo lớp (dùng cho --max-samples)."""
    rng = random.Random(seed)
    by_cls = {}
    for s in samples:
        by_cls.setdefault(s['label'], []).append(s)
    per = max(1, n // max(len(by_cls), 1))
    out = []
    for c in sorted(by_cls):
        cls = by_cls[c][:]
        rng.shuffle(cls)
        out += cls[:per]
    return out


def split_samples(samples):
    """dict tên tập → list mẫu: SSHAR và Widar {'train','val','test'} (Widar --val-ratio 0: {'train','test'})."""
    if DATASET == 'sshar':
        if SPLIT_BY == 'protocol':      # theo lần lặp (SSHAR_ESP_protocol.md mục 3)
            return {k: [s for s in samples if s['split'] == k] for k in ('train', 'val', 'test')}
        if SPLIT_BY == 'user':          # cross-user
            groups = dict(zip(('train', 'val', 'test'), SSHAR_USER_CASES[USER_CASE]))
            return {k: [s for s in samples if int(s['user'].split('_')[1]) in g]
                    for k, g in groups.items()}
        raise ValueError("SSHAR: --split repeat (= protocol) hoặc --split user --user-case N")
    train_s, test_s = _split_widar(samples)
    if VAL_RATIO <= 0:
        return {'train': train_s, 'test': test_s}
    train_s, val_s = _stratified_take(train_s, VAL_RATIO, SEED + 1)
    return {'train': train_s, 'val': val_s, 'test': test_s}


def _stratified_take(samples, ratio, seed):
    """Tách ngẫu nhiên `ratio` mẫu của MỖI lớp → (phần còn lại, phần tách ra)."""
    rng = random.Random(seed)
    rest, taken = [], []
    for c in sorted({s['label'] for s in samples}):
        cls = [s for s in samples if s['label'] == c]
        rng.shuffle(cls)
        n = int(round(len(cls) * ratio))
        if n == 0 and len(cls) > 1 and ratio > 0:
            n = 1   # lớp quá ít mẫu (chạy thử --max-samples) vẫn có 1 mẫu
        taken += cls[:n]
        rest  += cls[n:]
    return rest, taken


def _split_widar(samples):
    if SPLIT_BY == 'random':
        # Chia đều theo từng lớp để test có đủ mọi cử chỉ
        rng = random.Random(SEED)
        train_s, test_s = [], []
        for c in range(len(GESTURES)):
            cls = [s for s in samples if s['label'] == c]
            rng.shuffle(cls)
            n_test = int(len(cls) * TEST_RATIO)
            if n_test == 0 and len(cls) > 1 and TEST_RATIO > 0:
                n_test = 1   # lớp quá ít mẫu (chạy thử --max-samples) vẫn có 1 mẫu test
            test_s  += cls[:n_test]
            train_s += cls[n_test:]
        return train_s, test_s

    if SPLIT_BY in ('user', 'location', 'orientation'):
        test_s  = [s for s in samples if s[SPLIT_BY] in TEST_VALUES]
        train_s = [s for s in samples if s[SPLIT_BY] not in TEST_VALUES]
        return train_s, test_s

    raise ValueError(f"SPLIT_BY không hợp lệ: {SPLIT_BY}")


def prep_params():
    """Tham số tiền xử lý (ghi vào config + làm khoá cache)."""
    if DATASET == 'sshar':
        # 200 Hz: cửa sổ 64 điểm = 0.32 s (Widar: 256 điểm = 0.256 s), hop 4 = 20 ms
        # → L = (1000 - 64) / 4 + 1 = 235 khung; nfft 200 → 1 Hz/bin, 0..60 Hz = 61 bin
        return {'signal': 'amp', 'fs': 200, 'nperseg': 64, 'noverlap': 60, 'nfft': 200,
                'f_max': 60, 'highpass': DFS_HIGHPASS}
    return {'fs': 1000, 'nperseg': 256, 'noverlap': 248, 'nfft': 1000,
            'f_max': 60, 'highpass': DFS_HIGHPASS}


def _worker_init(worker_id):
    # Mỗi worker một seed khác nhau nhưng tái lập được
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    # 1 luồng/worker: tránh 4 worker x 4 luồng tranh 4 CPU + bớt malloc arena (phân mảnh RAM)
    torch.set_num_threads(1)


def make_loader(samples, shuffle, device, cache_root=None):
    labels = [s['label'] for s in samples]
    if cache_root is not None:
        ds = CachedWiDarDataset(cache_root, [s['id'] for s in samples], labels)
    else:
        ds = WiDarDataset(LazyCSIList([s['paths'] for s in samples]), labels,
                          **prep_params())
    # Test: 2 worker, không pin_memory. Worker phải persistent: tạo lại DataLoader
    # mỗi epoch = 1 thread pin_memory mới gọi CUDA → leak ~0.2 GB RAM/epoch ở
    # tiến trình chính (đo trên Kaggle). Test chỉ 43 batch nên bỏ pin không đáng kể.
    nw = NUM_WORKERS if shuffle else min(2, NUM_WORKERS)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=shuffle and len(ds) > BATCH_SIZE,   # train: bỏ batch lẻ cuối (BN + contrastive)
        num_workers=nw,
        pin_memory=PIN_MEMORY and shuffle and device == 'cuda',
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
        worker_init_fn=_worker_init,
    )


# =============================================================================
# Ghi log
# =============================================================================

def setup_logging(run_dir):
    """Log ra console + <run_dir>/train.log, có timestamp."""
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):      # gọi main() nhiều lần trong notebook
        log.removeHandler(h)
        h.close()
    fmt = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', '%Y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / 'train.log', encoding='utf-8')
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    logging.captureWarnings(True)
    wlog = logging.getLogger('py.warnings')
    for h in list(wlog.handlers):
        wlog.removeHandler(h)
    wlog.addHandler(fh)


def _run(cmd):
    try:
        return subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ''


def env_info(device):
    info = {
        'python'    : platform.python_version(),
        'platform'  : platform.platform(),
        'torch'     : torch.__version__,
        'cuda'      : torch.version.cuda,
        'cudnn'     : torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'device'    : device,
        'gpus'      : [f"{torch.cuda.get_device_name(d)} "
                       f"({torch.cuda.get_device_properties(d).total_memory / 2**30:.1f} GB)"
                       for d in range(torch.cuda.device_count())] if device == 'cuda' else [],
        'cpu_count' : os.cpu_count(),
        'is_kaggle' : IS_KAGGLE,
        'git_commit': _run(['git', 'rev-parse', '--short', 'HEAD']) or None,
        'git_dirty' : bool(_run(['git', 'status', '--porcelain', '--untracked-files=no'])),
        'code_dir'  : str(CODE_DIR),
    }
    try:
        with open('/proc/meminfo') as f:
            info['ram_gb'] = round(int(f.readline().split()[1]) / 2**20, 1)
    except OSError:
        pass
    try:
        info['disk_free_output_gb'] = round(shutil.disk_usage(OUTPUT_DIR).free / 2**30, 1)
    except OSError:
        pass
    return info


def sample_meta(s):
    """Thông tin 1 mẫu để ghi vào CSV (base name, user, location, orientation)."""
    m = {
        'sample'     : s['id'],
        'user'       : s['user'],
        'location'   : s['location'],
        'orientation': s['orientation'],
    }
    if 'rep' in s:
        m['rep'] = s['rep']
    return m


def log_split_stats(sets, class_names):
    """In phân bố mẫu các tập theo lớp, user, location, orientation."""
    cnt  = {n: None for n in sets}
    tag  = '/'.join(sets)
    for key in ('label', 'user', 'location', 'orientation'):
        cnt  = {n: Counter(s[key] for s in ss) for n, ss in sets.items()}
        keys = sorted(set().union(*cnt.values()), key=lambda x: (len(str(x)), str(x)))
        name = (lambda k: class_names[k]) if key == 'label' else str
        log.info(f"  theo {key:<11}: "
                 + ' | '.join(f"{name(k)} " + '/'.join(str(c[k]) for c in cnt.values()) for k in keys)
                 + f"   ({tag})")


def config_dict():
    return {
        'data_root'   : str(DATA_ROOT),
        'output_dir'  : str(OUTPUT_DIR),
        'users'       : USERS,
        'gestures'    : GESTURES,
        'V'           : V,
        'dataset'     : DATASET,
        'split_by'    : SPLIT_BY,
        'user_case'   : USER_CASE if DATASET == 'sshar' and SPLIT_BY == 'user' else None,
        'test_values' : TEST_VALUES,
        'test_ratio'  : TEST_RATIO,
        'val_ratio'   : VAL_RATIO,
        'rx_pos'      : RX_POS,
        'use_mask'    : RX_DIRECTIONS is not None,
        'D'           : D,
        'batch_size'  : BATCH_SIZE,
        'num_epochs'  : NUM_EPOCHS,
        'lr'          : LR,
        'beta1'       : BETA1,
        'beta2'       : BETA2,
        'weight_decay': WEIGHT_DECAY,
        'grad_clip'   : GRAD_CLIP,
        'seed'        : SEED,
        'min_packets' : MIN_PACKETS,
        'dfs_highpass': DFS_HIGHPASS,
        'prep'        : prep_params(),
        'amp'         : AMP,
        'multi_gpu'   : MULTI_GPU,
        'use_cache'   : USE_CACHE,
        'cache_dir'   : str(CACHE_DIR),
        'cache_search': [str(d) for d in CACHE_SEARCH],
        'pin_memory'  : PIN_MEMORY,
        'num_workers' : NUM_WORKERS,
        'max_samples' : MAX_SAMPLES,
        'time_limit_h': TIME_LIMIT_H,
        'patience'    : PATIENCE,
    }


def save_run_info(run_dir, env, sets, class_names, mask):
    """config.json (siêu tham số + môi trường) và split.csv (mẫu nào thuộc tập nào)."""
    config = {'run_name': run_dir.name, **config_dict(), 'class_names': class_names,
              **{f'n_{n}': len(ss) for n, ss in sets.items()},
              'spatial_mask': None if mask is None else
              [[0 if x == 0 else '-inf' for x in row] for row in mask.tolist()],
              'env': env, 'cmdline': sys.argv}
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(run_dir / 'split.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['set', 'sample', 'user', 'location', 'orientation', 'rep', 'label', 'class_name'])
        for name, ss in sets.items():
            for s in ss:
                m = sample_meta(s)
                w.writerow([name, m['sample'], m['user'], m['location'], m['orientation'],
                            m.get('rep', ''), s['label'], class_names[s['label']]])


# =============================================================================
# Main
# =============================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True   # ảnh cố định 224×224 → chọn kernel nhanh nhất

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Mỗi lần chạy 1 thư mục riêng: <OUTPUT_DIR>/runs/<thời gian>_<split>
    split_tag = SPLIT_BY if not TEST_VALUES else f"{SPLIT_BY}-{'_'.join(map(str, TEST_VALUES))}"
    if DATASET == 'sshar':
        split_tag = f"sshar-{SSHAR_DEVICE}_" + (f"user{USER_CASE}" if SPLIT_BY == 'user' else 'repeat')
    run_dir   = Path(RUN_DIR) if RUN_DIR else \
        Path(OUTPUT_DIR) / 'runs' / (RUN_NAME or f"{datetime.now():%Y%m%d_%H%M%S}_{split_tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)
    t_main = time.time()

    try:
        env = env_info(device)
        log.info("=" * 78)
        log.info(f"UniFi | run {run_dir.name}")
        log.info("=" * 78)
        for k, v in env.items():
            log.info(f"  env.{k:<20} {v}")
        if device == 'cpu':
            log.warning("Không có GPU! Trên Kaggle: Settings → Accelerator → GPU")
        for k, v in config_dict().items():
            log.info(f"  cfg.{k:<20} {v}")

        class_names = [GESTURE_NAMES.get(g, f'g{g}') for g in GESTURES]

        # ── Dữ liệu ───────────────────────────────────────────────────────
        if not Path(DATA_ROOT).is_dir():
            raise FileNotFoundError(f"Không thấy thư mục dữ liệu {DATA_ROOT} — truyền --data-root")
        log.info(f"Quét dữ liệu trong {DATA_ROOT} ...")
        samples = scan_sshar() if DATASET == 'sshar' else scan_samples()
        if not samples:
            raise RuntimeError(f"Không có mẫu nào trong {DATA_ROOT} — kiểm tra --data-root / --users")
        if MAX_SAMPLES:
            samples = subsample(samples, MAX_SAMPLES, SEED)
            log.warning(f"--max-samples: chỉ dùng {len(samples)} mẫu (chạy thử)")

        sets = split_samples(samples)
        empty = [n for n, ss in sets.items() if not ss]
        if empty:
            raise RuntimeError(f"Tập rỗng: {empty} — kiểm tra SPLIT_BY/TEST_VALUES")
        log.info(' | '.join(f"{n.capitalize()}: {len(ss)} mẫu" for n, ss in sets.items()))
        log_split_stats(sets, class_names)
        if DATASET == 'sshar' and not MAX_SAMPLES and USERS == SSHAR_SUBJECTS \
                and GESTURES == SSHAR_ACTS:
            got = {n: len(ss) for n, ss in sets.items()}
            if got != SSHAR_EXPECTED:
                raise RuntimeError(f"Số mẫu {got} != protocol {SSHAR_EXPECTED}")
            log.info(f"Số mẫu khớp protocol: {got}")
        if DATASET == 'sshar':
            if SPLIT_BY == 'user':
                us = {n: ', '.join(sorted({s['user'].split('_')[1] for s in ss})) for n, ss in sets.items()}
                log.info(f"👥 Split user case {USER_CASE} | test: {us['test']} | train: {us['train']} "
                         f"| val: {us['val']}")
            else:
                log.info("👥 Split repeat (theo lần lặp, SSHAR_ESP_protocol.md mục 3)")
            sizes = {n: len(ss) for n, ss in sets.items()}
            log.info(f"📦 sizes: {sizes} | seed {SEED} | patience {PATIENCE}")

        Mask = build_spatial_mask(V, RX_DIRECTIONS) if RX_DIRECTIONS is not None else None
        if Mask is not None:
            log.info("Spatial mask (1 = được attend):\n" +
                     '\n'.join('    r%d: ' % (i + 1) + ' '.join('1' if x == 0 else '.' for x in row)
                               for i, row in enumerate(Mask.tolist())))
        else:
            log.info("Không dùng spatial mask (mọi view attend lẫn nhau)")

        save_run_info(run_dir, env, sets, class_names, Mask)

        # ── Cache tiền xử lý ──────────────────────────────────────────────
        cache_root = None
        if USE_CACHE:
            # sắp theo id → khoá cache không phụ thuộc cách chia (dùng chung giữa các run)
            used = sorted((s for ss in sets.values() for s in ss), key=lambda s: s['id'])
            cache_root = build_or_load_cache(
                used, [s['id'] for s in used], load_view, prep_params(), CACHE_DIR,
                num_workers=NUM_WORKERS, search_dirs=CACHE_SEARCH)
            if CACHE_ONLY:
                log.info(f"--cache-only: xong, cache ở {cache_root} — tạo Kaggle Dataset từ "
                         f"thư mục này rồi Add Input cho notebook train")
                return None
        else:
            log.info("Không dùng cache: đọc file + STFT mỗi epoch (chậm)")

        # Tập chọn checkpoint: val (SSHAR) hoặc test (Widar, như UniFi)
        eval_name = 'val' if 'val' in sets else 'test'
        train_loader = make_loader(sets['train'], shuffle=True, device=device, cache_root=cache_root)
        eval_loader  = make_loader(sets[eval_name], shuffle=False, device=device, cache_root=cache_root)
        final_eval = None
        if eval_name == 'val':
            test_loader = make_loader(sets['test'], shuffle=False, device=device, cache_root=cache_root)
            final_eval  = [('val',  eval_loader, [sample_meta(s) for s in sets['val']]),
                           ('test', test_loader, [sample_meta(s) for s in sets['test']])]
        if eval_name == 'val':
            log.info("Chọn checkpoint theo val acc; test chỉ đánh giá 1 lần bằng checkpoint đó sau khi train")
        else:
            log.warning("--val-ratio 0: chọn checkpoint theo TEST acc (kết quả lạc quan)")

        # Kiểm tra 1 batch trước khi train
        t0 = time.time()
        X_f, X_t, y = next(iter(eval_loader))
        log.info(f"Batch mẫu: X_f {tuple(X_f.shape)} [{X_f.min():.3f}, {X_f.max():.3f}] | "
                 f"X_t {tuple(X_t.shape)} [{X_t.min():.3f}, {X_t.max():.3f}] | "
                 f"y {y.tolist()[:8]}... | {time.time() - t0:.1f}s")

        # ── Mô hình + train ──────────────────────────────────────────────
        model = UniFi(V=V, D=D, num_classes=len(GESTURES), Mask=Mask)
        for name, m in model.named_children():
            log.info(f"  module {name:<11} {sum(p.numel() for p in m.parameters()):>12,} tham số")

        result = train(model, train_loader, eval_loader,
                       num_epochs=NUM_EPOCHS, lr=LR,
                       beta1=BETA1, beta2=BETA2, device=device,
                       save_path=run_dir / 'best_model.pth',
                       log_dir=run_dir,
                       eval_meta=[sample_meta(s) for s in sets[eval_name]],   # cùng thứ tự loader
                       eval_name=eval_name, final_eval=final_eval,
                       class_names=class_names,
                       amp=AMP, multi_gpu=MULTI_GPU,
                       weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP,
                       time_limit_s=TIME_LIMIT_H * 3600 if TIME_LIMIT_H else None,
                       log_interval=LOG_INTERVAL, patience=PATIENCE,
                       extra_summary={'run_dir': str(run_dir), 'dataset': DATASET,
                                      'split_by': SPLIT_BY, 'test_values': TEST_VALUES,
                                      'val_ratio': VAL_RATIO,
                                      **{f'n_{n}': len(ss) for n, ss in sets.items()},
                                      'git_commit': env.get('git_commit')})

        # ── Vẽ hình + nén ─────────────────────────────────────────────────
        if MAKE_PLOTS:
            try:
                import plot_logs
                plot_logs.main(run_dir)
                log.info(f"Hình: {run_dir / 'plots'}")
            except Exception:
                log.exception("Vẽ hình lỗi (bỏ qua)")
        return result

    except Exception:
        log.exception("LỖI — dừng chương trình")
        raise
    finally:
        log.info(f"Tổng thời gian: {(time.time() - t_main) / 60:.1f} phút")
        if ZIP_RUN:
            try:
                zip_path = shutil.make_archive(str(run_dir), 'zip', run_dir.parent, run_dir.name)
                log.info(f"Đã nén: {zip_path}")
            except Exception:
                log.exception("Nén thư mục run lỗi (bỏ qua)")
        for h in log.handlers:
            h.flush()


# =============================================================================
# Dòng lệnh
# =============================================================================

def _int_or_str(x):
    try:
        return int(x)
    except ValueError:
        return x


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Train UniFi trên Widar 3.0 hoặc SSHAR ESP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group('dữ liệu / đường dẫn')
    g.add_argument('--dataset', choices=['widar', 'sshar'], default=DATASET)
    g.add_argument('--data-root', '--data_path', dest='data_root', default=None,
                   help='widar/ chứa userX/*.dat, hoặc sshar_amp_interp/ chứa room_XX (mặc định: tự tìm)')
    g.add_argument('--output-dir', default=str(OUTPUT_DIR), help='nơi tạo runs/<tên run>/')
    g.add_argument('--run-name', '--run_name', dest='run_name', default=RUN_NAME,
                   help='tên thư mục run (mặc định: thời gian + kiểu chia)')
    g.add_argument('--results_dir', '--results-dir', dest='results_dir', default=None,
                   help='SSHAR, theo kaggle_run_convention.md: ghi results_dir/<run_name>/UniFi/seed_<s>/ '
                        'cho từng seed trong --seeds, kèm summary.txt / summary.csv / per_seed.csv')
    g.add_argument('--users', nargs='+', default=None,
                   help='Widar: user5 ... | SSHAR: subject_01 ... (mặc định: tất cả)')
    g.add_argument('--gestures', nargs='+', type=int, default=None,
                   help='gesture id (Widar) / act id (SSHAR) (mặc định: tất cả)')
    g.add_argument('--max-samples', type=int, default=MAX_SAMPLES, help='dùng ít mẫu để chạy thử')

    g = p.add_argument_group('chia train/test')
    g.add_argument('--split', choices=['random', 'user', 'location', 'orientation', 'protocol', 'repeat'],
                   default=None, help='mặc định: random (Widar), user (SSHAR). '
                                      'SSHAR: user (kèm --user_case) | repeat (= protocol)')
    g.add_argument('--user-case', '--user_case', dest='user_case', type=int, default=USER_CASE, choices=range(len(SSHAR_USER_CASES)),
                   help='SSHAR --split user: case 0..4 (train/val/test theo người, xem SSHAR_USER_CASES)')
    g.add_argument('--test-values', nargs='*', type=_int_or_str, default=TEST_VALUES,
                   help="ví dụ: --split location --test-values 3 ; --split user --test-values user5")
    g.add_argument('--test-ratio', type=float, default=TEST_RATIO)
    g.add_argument('--val-ratio', type=float, default=VAL_RATIO,
                   help='Widar: tỉ lệ val tách từ phần train (mỗi lớp); 0 = không val, chọn theo test. '
                        'SSHAR: bỏ qua (val theo protocol)')
    g.add_argument('--seed', type=int, default=SEED)
    g.add_argument('--seeds', nargs='+', type=int, default=None,
                   help='nhiều seed, chạy lần lượt (dùng với --results_dir)')

    g = p.add_argument_group('huấn luyện')
    g.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    g.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=BATCH_SIZE)
    g.add_argument('--lr', type=float, default=LR)
    g.add_argument('--beta1', type=float, default=BETA1)
    g.add_argument('--beta2', type=float, default=BETA2)
    g.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY)
    g.add_argument('--grad-clip', type=float, default=GRAD_CLIP)
    g.add_argument('--no-mask', action='store_true', help='tắt spatial mask')
    g.add_argument('--no-highpass', action='store_true', help='tắt lọc thông cao 2 Hz trước DFS')

    g = p.add_argument_group('tốc độ / môi trường')
    g.add_argument('--workers', type=int, default=NUM_WORKERS)
    g.add_argument('--no-amp', action='store_true', help='tắt mixed precision')
    g.add_argument('--multi-gpu', action='store_true',
                   help='DataParallel trên mọi GPU (leak RAM ~0.2 GB/epoch, chậm hơn 1 GPU)')
    g.add_argument('--single-gpu', action='store_true', help='(mặc định, giữ để tương thích)')
    g.add_argument('--no-cache', action='store_true', help='không cache tiền xử lý')
    g.add_argument('--cache-dir', default=str(CACHE_DIR), help='nơi ghi cache (build mới)')
    g.add_argument('--cache-search', nargs='*', default=[str(d) for d in CACHE_SEARCH],
                   help='thư mục tìm cache build sẵn (chỉ đọc), vd. /kaggle/input')
    g.add_argument('--pin-memory', action='store_true', help='bật pin_memory cho train loader')
    g.add_argument('--trace-mem', action='store_true',
                   help='tracemalloc: in dòng code Python giữ thêm RAM mỗi epoch (chậm hơn chút)')
    g.add_argument('--cache-only', action='store_true',
                   help='chỉ build cache rồi thoát; dùng với --cache-dir /kaggle/working/unifi_cache')
    g.add_argument('--patience', type=int, default=PATIENCE,
                   help='early stopping: số epoch không cải thiện val acc trước khi dừng (0 = tắt)')
    g.add_argument('--time-limit-hours', type=float, default=TIME_LIMIT_H,
                   help='dừng sớm (vẫn ghi kết quả) trước giới hạn này; 0 = không giới hạn')
    g.add_argument('--log-interval', type=int, default=LOG_INTERVAL)
    g.add_argument('--no-plots', action='store_true')
    g.add_argument('--zip', dest='zip_run', action='store_true', default=ZIP_RUN,
                   help='nén thư mục run thành .zip (mặc định bật trên Kaggle)')
    g.add_argument('--no-zip', dest='zip_run', action='store_false')
    return p.parse_args(argv)


def apply_args(a):
    """Ghi đè biến cấu hình của module theo tham số dòng lệnh."""
    global DATA_ROOT, OUTPUT_DIR, RUN_NAME, USERS, GESTURES, MAX_SAMPLES, SPLIT_BY, TEST_VALUES
    global TEST_RATIO, SEED, NUM_EPOCHS, BATCH_SIZE, LR, BETA1, BETA2, WEIGHT_DECAY, GRAD_CLIP
    global RX_DIRECTIONS, DFS_HIGHPASS, NUM_WORKERS, AMP, MULTI_GPU, USE_CACHE, CACHE_DIR
    global TIME_LIMIT_H, LOG_INTERVAL, MAKE_PLOTS, ZIP_RUN, CACHE_SEARCH, CACHE_ONLY, PIN_MEMORY, PATIENCE
    global DATASET, GESTURE_NAMES, V, RX_POS, VAL_RATIO, USER_CASE
    DATASET      = a.dataset
    if DATASET == 'sshar':
        DATA_ROOT     = Path(a.data_root) if a.data_root else find_sshar_root()
        USERS         = a.users or list(SSHAR_SUBJECTS)
        GESTURES      = a.gestures or list(SSHAR_ACTS)
        GESTURE_NAMES = {k: f'act{k:02d}' for k in SSHAR_ACTS}
        V             = len(SSHAR_RX)
        RX_POS        = None
        RX_DIRECTIONS = None
        SPLIT_BY      = {'repeat': 'protocol', None: 'user'}.get(a.split, a.split)
        USER_CASE     = a.user_case
        if SPLIT_BY not in ('protocol', 'user'):
            raise SystemExit("SSHAR: --split repeat hoặc --split user")
    else:
        DATA_ROOT = Path(a.data_root) if a.data_root else find_data_root()
        USERS     = a.users or USERS
        GESTURES  = a.gestures or GESTURES
        SPLIT_BY  = a.split or 'random'
    OUTPUT_DIR   = Path(a.output_dir)
    RUN_NAME     = a.run_name
    MAX_SAMPLES  = a.max_samples
    TEST_VALUES  = a.test_values
    TEST_RATIO   = a.test_ratio
    VAL_RATIO    = a.val_ratio
    SEED         = a.seed
    NUM_EPOCHS   = a.epochs
    BATCH_SIZE   = a.batch_size
    LR           = a.lr
    BETA1        = a.beta1
    BETA2        = a.beta2
    WEIGHT_DECAY = a.weight_decay
    GRAD_CLIP    = a.grad_clip
    if a.no_mask:
        RX_DIRECTIONS = None   # (SSHAR luôn không có mask)
    DFS_HIGHPASS = not a.no_highpass
    NUM_WORKERS  = a.workers
    AMP          = not a.no_amp
    MULTI_GPU    = a.multi_gpu and not a.single_gpu
    USE_CACHE    = not a.no_cache
    CACHE_DIR    = Path(a.cache_dir)
    CACHE_SEARCH = [Path(d) for d in a.cache_search]
    CACHE_ONLY   = a.cache_only
    PIN_MEMORY   = a.pin_memory
    if a.trace_mem:
        os.environ['UNIFI_TRACEMALLOC'] = '1'
    if CACHE_ONLY and not USE_CACHE:
        raise SystemExit('--cache-only không đi cùng --no-cache')
    TIME_LIMIT_H = a.time_limit_hours or None
    PATIENCE     = a.patience
    LOG_INTERVAL = a.log_interval
    MAKE_PLOTS   = not a.no_plots
    ZIP_RUN      = a.zip_run


# Bắt buộc trên Windows khi num_workers > 0:
# worker process import lại main.py, không được chạy train lần nữa
def run_convention(a):
    """kaggle_run_convention.md: mỗi seed 1 lần main() vào results/<RUN>/<MODEL>/seed_<s>/, rồi tổng hợp."""
    global SEED, RUN_DIR, ZIP_RUN
    import gc
    import sshar_results
    seeds    = a.seeds or [SEED]
    run_name = a.run_name or (f"SSHAR_ESP_{MODEL_NAME}_"
                              + (f"user{USER_CASE}" if SPLIT_BY == 'user' else 'repeat')
                              + f"_p{PATIENCE}_seed{'-'.join(map(str, seeds))}")
    run_root = Path(a.results_dir) / run_name
    ZIP_RUN  = False                 # notebook tự nén cả results/<RUN>
    if CACHE_ONLY:                   # chỉ build cache, không train / không ghi results
        main()
        return
    rows = []
    for seed in seeds:
        SEED    = seed
        RUN_DIR = run_root / MODEL_NAME / f'seed_{seed}'
        main()
        rows.append(sshar_results.export_seed(RUN_DIR, seed))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    groups = dict(zip(('train', 'val', 'test'), SSHAR_USER_CASES[USER_CASE])) if SPLIT_BY == 'user' else None
    cfg = {'split': 'user' if SPLIT_BY == 'user' else 'repeat',
           'user_case': USER_CASE if SPLIT_BY == 'user' else None,
           'users': ({k: [f'{u:02d}' for u in v] for k, v in groups.items()} if groups
                     else 'tất cả (chia theo lần lặp)'),
           'patience': PATIENCE, 'seeds': [r['seed'] for r in rows],
           'batch_size': BATCH_SIZE, 'epochs': NUM_EPOCHS, 'lr': LR, 'beta1': BETA1, 'beta2': BETA2,
           'checkpoint': 'val accuracy cao nhất', 'report': 'test (+ val cùng checkpoint)',
           'data_path': str(DATA_ROOT)}
    txt = sshar_results.write_run_summary(run_root, MODEL_NAME, rows, cfg)
    print('\n' + txt)
    print(f"Kết quả: {run_root}")


if __name__ == '__main__':
    _a = parse_args()
    apply_args(_a)
    if _a.results_dir:
        if DATASET != 'sshar':
            raise SystemExit('--results_dir chỉ dùng với --dataset sshar')
        run_convention(_a)
    else:
        if _a.seeds:
            raise SystemExit('--seeds cần --results_dir')
        main()
