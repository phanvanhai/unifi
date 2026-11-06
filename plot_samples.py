"""
Vẽ ảnh DFS + pha của một số mẫu Widar ra PNG để kiểm tra trực quan.
So sánh DFS khi tắt / bật lọc thông cao 2 Hz.

    python plot_samples.py

Kết quả ở figures/:
    raw_g<gesture>.png     bộ thu r1 trước resize, có trục vật lý:
                           DFS (không lọc) | DFS (lọc 2 Hz) | pha
    input_g<gesture>.png   ảnh đầu vào mô hình (224×224), 6 bộ thu:
                           hàng DFS (không lọc) / DFS (lọc 2 Hz) / pha
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from main import OUTPUT_DIR, V, load_view, find_user_dir
from unifi_preprocessing import WiDarDataset, extract_dfs, extract_phase, PARULA

OUT_DIR  = Path(OUTPUT_DIR) / 'figures'
USER     = 'user3'
GESTURES = [1, 2, 3, 4, 5, 6]
LOC, ORI, REP = 1, 1, 1
FS, NPERSEG, NOVERLAP, F_MAX = 1000, 256, 248, 60


def sample_paths(gesture):
    base = f'{USER}-{gesture}-{LOC}-{ORI}-{REP}'
    user_dir = find_user_dir(USER)
    return base, [user_dir / f'{base}-r{r}.dat' for r in range(1, V + 1)]


def plot_raw(base, H_r, out_path):
    """DFS [F, L] (tắt / bật lọc) và pha [K, T] của 1 bộ thu, chưa resize."""
    dfs = {hp: extract_dfs(H_r, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                           f_max=F_MAX, highpass=hp)
           for hp in (False, True)}
    T_pha = extract_phase(H_r)
    hop   = NPERSEG - NOVERLAP
    L     = dfs[False].shape[1]
    t_dfs = (NPERSEG / 2 + hop * np.array([0, L - 1])) / FS

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, hp in zip(axes[:2], (False, True)):
        ax.imshow(dfs[hp], aspect='auto', origin='lower', cmap=PARULA,
                  extent=[t_dfs[0], t_dfs[1], -F_MAX, F_MAX])
        ax.set(title=f'DFS {"lọc 2 Hz" if hp else "không lọc"}  {dfs[hp].shape}',
               xlabel='Thời gian (s)', ylabel='Doppler (Hz)')
    axes[2].imshow(T_pha, aspect='auto', origin='lower', cmap=PARULA,
                   extent=[0, T_pha.shape[1] / FS, 1, T_pha.shape[0]])
    axes[2].set(title=f'Pha CSI ratio  {T_pha.shape}', xlabel='Thời gian (s)', ylabel='Sub-carrier')
    fig.suptitle(f'{base}-r1  (trước resize)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_input(base, H_views, out_path):
    """Ảnh đầu vào mô hình: đi qua đúng WiDarDataset (min-max → parula → resize)."""
    X_f_off, X_t, _ = WiDarDataset([H_views], [0], highpass=False)[0]  # [V, 3, 224, 224]
    X_f_on,  _,   _ = WiDarDataset([H_views], [0], highpass=True)[0]
    rows = [(X_f_off, 'DFS không lọc'), (X_f_on, 'DFS lọc 2 Hz'), (X_t, 'Pha')]

    fig, axes = plt.subplots(len(rows), V, figsize=(2.2 * V, 2.4 * len(rows)))
    for row, (X, name) in enumerate(rows):
        for v in range(V):
            ax = axes[row, v]
            ax.imshow(X[v].permute(1, 2, 0).numpy(), origin='lower')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f'r{v + 1}')
            if v == 0:
                ax.set_ylabel(name)
    fig.suptitle(f'{base}  (đầu vào mô hình 3×224×224)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for g in GESTURES:
        base, paths = sample_paths(g)
        H_views = [load_view(p) for p in paths]
        plot_raw(base, H_views[0], OUT_DIR / f'raw_g{g}.png')
        plot_input(base, H_views, OUT_DIR / f'input_g{g}.png')
        print(f'{base}: T = {[h.shape[1] for h in H_views]}')
    print(f'Đã lưu vào {OUT_DIR}')
