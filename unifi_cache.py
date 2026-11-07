"""
Cache tiền xử lý cho UniFi.

Vì sao cần:
    Đọc 6 file .dat + HPF + PCA + STFT tốn ~0.2–0.3 s/mẫu (CPU). Với ~6700 mẫu
    và 4 CPU của Kaggle → ~8 phút/epoch chỉ để đọc dữ liệu, 100 epoch > 12 giờ.
    Phần này không đổi giữa các epoch (không có augmentation) → tính 1 lần.

Lưu gì:
    Bản đồ DFS [F, L] và pha [K, T] đã min-max về [0, 1], lượng tử hoá uint16
    (sai số ≤ 7.6e-6, nhỏ hơn nhiều so với 1/255 của ảnh). Colormap + resize
    (~3 ms/view) vẫn chạy online → kết quả gần như giống hệt pipeline không cache
    (chỉ vài pixel nằm sát ranh giới 2 màu trong bảng 256 màu bị lệch 1 nấc,
    sai khác tối đa ~0.01).
    Kích thước ~1 MB/mẫu → ~6.5 GB cho 6700 mẫu.

Cấu trúc thư mục cache (<cache_dir>/<key>/):
    data.bin     uint16 nối liền tất cả bản đồ
    index.npy    int64 [N, V, 2, 3] = (offset, h, w) cho (mẫu, view, dfs|pha)
    meta.json    tên mẫu theo thứ tự + tham số tiền xử lý
    DONE         chỉ ghi khi build xong → cache dở dang sẽ bị build lại
"""

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from unifi_preprocessing import compute_maps, maps_to_tensors

log = logging.getLogger('unifi')

CACHE_VERSION = 1
Q = 65535.0   # thang lượng tử uint16


def cache_key(sample_ids, prep):
    """Hash của danh sách mẫu + tham số tiền xử lý → đổi gì là build cache mới."""
    blob = json.dumps({'v': CACHE_VERSION, 'ids': list(sample_ids), 'prep': prep},
                      sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


class _MapsJob(Dataset):
    """Dataset phụ để build cache song song bằng DataLoader workers."""

    def __init__(self, path_lists, load_view, prep):
        self.path_lists = path_lists
        self.load_view  = load_view
        self.prep       = prep

    def __len__(self):
        return len(self.path_lists)

    def __getitem__(self, i):
        out = []
        for p in self.path_lists[i]:
            dfs01, pha01 = compute_maps(self.load_view(p), **self.prep)
            out.append((np.round(dfs01 * Q).astype(np.uint16),
                        np.round(pha01 * Q).astype(np.uint16)))
        return i, out


def _identity(x):
    # batch_size=None mặc định gọi default_convert (numpy → tensor); giữ nguyên numpy
    return x


def _same_prep(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _iter_meta(root, max_depth):
    """Mọi meta.json dưới root (≤ max_depth cấp), không đi vào thư mục chứa .dat."""
    root = Path(root)
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        if 'meta.json' in filenames:
            yield Path(dirpath) / 'meta.json'
        if len(Path(dirpath).parts) - base >= max_depth or any(f.endswith('.dat') for f in filenames):
            dirnames[:] = []


def find_cache(sample_ids, prep, search_dirs, max_depth=8):
    """
    Tìm cache đã build xong trong `search_dirs` (vd. Kaggle Dataset gắn ở
    /kaggle/input/datasets/<owner>/<slug>/...) có cùng tham số tiền xử lý và
    chứa ĐỦ các mẫu cần dùng (được phép nhiều hơn). Trả về thư mục hoặc None.
    """
    need = set(sample_ids)
    for d in search_dirs:
        if not Path(d).is_dir():
            continue
        for meta_p in _iter_meta(d, max_depth):
            root = meta_p.parent
            try:
                meta = json.loads(meta_p.read_text(encoding='utf-8'))
            except (OSError, ValueError) as e:
                log.info(f"[cache] Bỏ qua {root}: không đọc được meta.json ({e})")
                continue
            if 'ids' not in meta or 'prep' not in meta:
                continue   # meta.json của thứ khác
            why = None
            if not ((root / 'data.bin').exists() and (root / 'index.npy').exists()):
                why = 'thiếu data.bin/index.npy'
            elif meta.get('bytes') is not None and (root / 'data.bin').stat().st_size != meta['bytes']:
                why = 'data.bin không đủ kích thước (build dở / upload lỗi)'
            elif meta.get('version') != CACHE_VERSION:
                why = f"version {meta.get('version')} != {CACHE_VERSION}"
            elif not _same_prep(meta.get('prep'), prep):
                why = f"khác tham số tiền xử lý: {meta.get('prep')} vs {prep}"
            else:
                miss = need.difference(meta['ids'])
                if miss:
                    why = f"thiếu {len(miss)}/{len(need)} mẫu, vd. {sorted(miss)[:3]}"
            if why:
                log.info(f"[cache] Bỏ qua {root}: {why}")
                continue
            return root
    return None


def build_or_load_cache(samples, sample_ids, load_view, prep, cache_dir,
                        num_workers=4, log_every=200, search_dirs=()):
    """
    Trả về đường dẫn thư mục cache (đã build xong) cho danh sách `samples`.

    samples    : list dict có key 'paths' (V đường dẫn .dat)
    sample_ids : list tên mẫu (cùng thứ tự), dùng làm khoá
    load_view  : hàm path → H_r [K, T] complex
    prep       : dict tham số cho compute_maps
    """
    key  = cache_key(sample_ids, prep)
    root = Path(cache_dir) / key
    if (root / 'DONE').exists():
        size = (root / 'data.bin').stat().st_size / 2**30
        log.info(f"[cache] Dùng lại cache có sẵn: {root} ({size:.2f} GB, {len(samples)} mẫu)")
        return root

    found = find_cache(sample_ids, prep, [cache_dir, *search_dirs])
    if found is not None:
        size = (found / 'data.bin').stat().st_size / 2**30
        log.info(f"[cache] Dùng cache có sẵn (chứa đủ {len(samples)} mẫu cần): {found} ({size:.2f} GB)")
        return found
    log.info(f"[cache] Không thấy cache phù hợp trong {[str(cache_dir), *map(str, search_dirs)]}")

    if root.exists():
        log.warning(f"[cache] Cache dở dang ở {root} → xoá và build lại")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(root).free / 2**30
    need = len(samples) * 1.1 / 1024   # ~1.1 MB/mẫu (ước lượng)
    log.info(f"[cache] Build cache: {len(samples)} mẫu → {root}")
    log.info(f"[cache] Ước lượng cần ~{need:.1f} GB, ổ đĩa còn trống {free:.1f} GB")
    if free < need * 1.1:
        log.warning("[cache] Có thể không đủ dung lượng! Đổi --cache-dir hoặc dùng --no-cache")

    V     = len(samples[0]['paths'])
    index = np.zeros((len(samples), V, 2, 3), dtype=np.int64)
    loader = DataLoader(_MapsJob([s['paths'] for s in samples], load_view, prep),
                        batch_size=None, shuffle=False, num_workers=num_workers,
                        collate_fn=_identity)

    t0, offset, done = time.time(), 0, 0
    with open(root / 'data.bin', 'wb') as f:
        for i, views in loader:
            for v, maps in enumerate(views):
                for k, m in enumerate(maps):
                    m = np.ascontiguousarray(m)
                    f.write(m.tobytes())
                    index[i, v, k] = (offset, m.shape[0], m.shape[1])
                    offset += m.size
            done += 1
            if done % log_every == 0 or done == len(samples):
                dt   = time.time() - t0
                rate = done / max(dt, 1e-9)
                eta  = (len(samples) - done) / max(rate, 1e-9)
                log.info(f"[cache] {done:5d}/{len(samples)} mẫu | {rate:5.1f} mẫu/s | "
                         f"{offset * 2 / 2**30:.2f} GB | đã chạy {dt / 60:.1f} phút | "
                         f"còn ~{eta / 60:.1f} phút")

    np.save(root / 'index.npy', index)
    with open(root / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump({'version': CACHE_VERSION, 'ids': list(sample_ids), 'prep': prep,
                   'n': len(samples), 'V': V, 'bytes': offset * 2}, f, ensure_ascii=False)
    (root / 'DONE').write_text('ok')
    log.info(f"[cache] Xong trong {(time.time() - t0) / 60:.1f} phút, "
             f"{offset * 2 / 2**30:.2f} GB")
    return root


class CachedWiDarDataset(Dataset):
    """
    Thay WiDarDataset khi đã có cache. Trả về cùng định dạng:
        X_f [V, 3, H, W], X_t [V, 3, H, W], y (long)
    """

    def __init__(self, cache_root, sample_ids, labels, colormap='parula', img_size=(224, 224)):
        cache_root = Path(cache_root)
        meta = json.loads((cache_root / 'meta.json').read_text(encoding='utf-8'))
        pos  = {sid: i for i, sid in enumerate(meta['ids'])}
        missing = [s for s in sample_ids if s not in pos]
        if missing:
            raise KeyError(f"{len(missing)} mẫu không có trong cache, ví dụ {missing[:3]}")

        self.rows     = [pos[s] for s in sample_ids]
        self.labels   = list(labels)
        self.index    = np.load(cache_root / 'index.npy')
        self.bin_path = str(cache_root / 'data.bin')
        self.colormap = colormap
        self.img_size = img_size
        self._f       = None   # mở lười trong từng worker (file handle không pickle được)

    def __len__(self):
        return len(self.rows)

    def _map(self, off, h, w):
        # Đọc bằng pread thay vì memmap: memmap giữ trang đã đọc trong RSS của
        # từng worker (persistent) → sau nhiều epoch mỗi worker "giữ" gần cả
        # file cache (~6 GB) → container Kaggle hết RAM ở cuối run.
        n   = int(h) * int(w)
        buf = os.pread(self._f.fileno(), n * 2, int(off) * 2) if hasattr(os, 'pread')             else (self._f.seek(int(off) * 2), self._f.read(n * 2))[1]
        return np.frombuffer(buf, dtype=np.uint16).reshape(h, w).astype(np.float32) / Q

    def __getitem__(self, idx):
        if self._f is None:
            self._f = open(self.bin_path, 'rb', buffering=0)
        X_f_list, X_t_list = [], []
        for v_idx in self.index[self.rows[idx]]:       # [2, 3]
            dfs01 = self._map(*v_idx[0])
            pha01 = self._map(*v_idx[1])
            X_f, X_t = maps_to_tensors(dfs01, pha01, self.colormap, self.img_size)
            X_f_list.append(X_f)
            X_t_list.append(X_t)
        # fp16: giảm 1/2 RAM cho shared memory + pinned memory của DataLoader
        # (sai số ≤ 2.4e-4 trên [0, 1]; train/evaluate đổi lại .float() trên GPU)
        return (torch.stack(X_f_list).half(), torch.stack(X_t_list).half(),
                torch.tensor(self.labels[idx], dtype=torch.long))
