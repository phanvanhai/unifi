"""
Vẽ kết quả training từ thư mục log do main.py tạo ra.

    python plot_logs.py                      # run mới nhất trong <OUTPUT_DIR>/runs
    python plot_logs.py runs/20261104_...    # một run cụ thể

Đọc : history.csv, test_predictions_best.csv, config.json
Ghi : <run_dir>/plots/
    curves_loss.png        loss tổng train/test + từng thành phần (G, V, P)
    curves_acc.png         accuracy train/test, đánh dấu epoch tốt nhất
    confusion_matrix.png   số lượng + chuẩn hoá theo hàng (epoch tốt nhất)
    acc_per_view.png       accuracy nhánh global so với từng view
    acc_by_<meta>.png      accuracy theo user / location / orientation
    report.txt             precision / recall / F1 từng lớp
"""
import sys
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def col(rows, key):
    return np.array([float(r[key]) if r[key] not in ('', 'None') else np.nan for r in rows])


def plot_curves(hist, out):
    ep = col(hist, 'epoch')

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, (key, title) in zip(axes, [('loss', 'Loss tổng'), ('L_global', 'L_global'),
                                      ('L_view', 'L_view (tổng V view)'), ('L_pcon', 'L_pcon')]):
        ax.plot(ep, col(hist, f'train_{key}'), label='train')
        ax.plot(ep, col(hist, f'test_{key}'),   label='test')
        ax.set(title=title, xlabel='Epoch')
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'curves_loss.png', dpi=120)
    plt.close(fig)

    best = int(np.nanargmax(col(hist, 'test_acc')))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, col(hist, 'train_acc'), label='train')
    ax.plot(ep, col(hist, 'test_acc'),   label='test')
    ax.scatter(ep[best], col(hist, 'test_acc')[best], color='red', zorder=3,
               label=f'best {col(hist, "test_acc")[best]:.4f} @ {int(ep[best])}')
    ax.set(title='Accuracy', xlabel='Epoch', ylim=(0, 1.02))
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'curves_acc.png', dpi=120)
    plt.close(fig)


def plot_confusion(y_true, y_pred, names, out):
    C  = len(names)
    cm = np.zeros((C, C), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(6 + 1.2 * C, 0.6 * C + 3))
    for ax, M, title, fmt in [(axes[0], cm, 'Số lượng', '{:d}'),
                              (axes[1], cmn, 'Chuẩn hoá theo hàng', '{:.2f}')]:
        ax.imshow(M, cmap='Blues', vmin=0)
        thr = M.max() / 2
        for i in range(C):
            for j in range(C):
                ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center',
                        color='white' if M[i, j] > thr else 'black', fontsize=9)
        ax.set(xticks=range(C), yticks=range(C), xticklabels=names, yticklabels=names,
               xlabel='Dự đoán', ylabel='Thật', title=title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.suptitle(f'Confusion matrix — acc = {(y_true == y_pred).mean():.4f}')
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    return cm


def write_report(cm, names, out):
    lines = [f"{'class':>12} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}"]
    f1s = []
    for c, n in enumerate(names):
        tp   = cm[c, c]
        prec = tp / max(cm[:, c].sum(), 1)
        rec  = tp / max(cm[c].sum(), 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-12)
        f1s.append(f1)
        lines.append(f"{n:>12} {prec:10.4f} {rec:10.4f} {f1:10.4f} {cm[c].sum():8d}")
    lines.append(f"\naccuracy = {np.trace(cm) / cm.sum():.4f} | macro F1 = {np.mean(f1s):.4f}")
    (out / 'report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


def plot_per_view(preds, y_true, out):
    vcols = sorted([k for k in preds[0] if k.startswith('view') and k.endswith('_pred')],
                   key=lambda k: int(k[4:-5]))
    accs  = [(col(preds, 'y_pred') == y_true).mean()]
    accs += [(col(preds, k) == y_true).mean() for k in vcols]
    labels = ['global'] + [k[:-5] for k in vcols]

    fig, ax = plt.subplots(figsize=(1 + 0.9 * len(labels), 4))
    bars = ax.bar(labels, accs, color=['C3'] + ['C0'] * len(vcols))
    ax.bar_label(bars, fmt='%.3f', fontsize=8)
    ax.set(title='Accuracy: global vs từng view', ylim=(0, 1.08))
    fig.tight_layout()
    fig.savefig(out / 'acc_per_view.png', dpi=120)
    plt.close(fig)


def plot_by_meta(preds, out):
    for key in ('user', 'location', 'orientation'):
        if key not in preds[0]:
            continue
        groups = sorted({r[key] for r in preds}, key=lambda x: (len(x), x))
        if len(groups) < 2:
            continue
        accs = [np.mean([int(r['correct']) for r in preds if r[key] == g]) for g in groups]
        ns   = [sum(r[key] == g for r in preds) for g in groups]

        fig, ax = plt.subplots(figsize=(1 + 0.8 * len(groups), 4))
        bars = ax.bar(groups, accs)
        ax.bar_label(bars, labels=[f'{a:.3f}\n(n={n})' for a, n in zip(accs, ns)], fontsize=8)
        ax.set(title=f'Accuracy theo {key}', xlabel=key, ylim=(0, 1.15))
        fig.tight_layout()
        fig.savefig(out / f'acc_by_{key}.png', dpi=120)
        plt.close(fig)


def main(run_dir):
    run_dir = Path(run_dir)
    out = run_dir / 'plots'
    out.mkdir(exist_ok=True)

    cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    names = cfg.get('class_names') or [f'g{g}' for g in cfg['gestures']]

    hist = read_csv(run_dir / 'history.csv')
    if hist:
        plot_curves(hist, out)

    pred_file = run_dir / 'test_predictions_best.csv'
    if pred_file.exists():
        preds  = read_csv(pred_file)
        y_true = col(preds, 'y_true').astype(int)
        y_pred = col(preds, 'y_pred').astype(int)
        cm = plot_confusion(y_true, y_pred, names, out)
        write_report(cm, names, out)
        plot_per_view(preds, y_true, out)
        plot_by_meta(preds, out)

    print(f'Đã lưu hình vào {out}')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    else:
        from main import OUTPUT_DIR
        runs = sorted((Path(OUTPUT_DIR) / 'runs').glob('*'))
        if not runs:
            sys.exit(f'Không có run nào trong {Path(OUTPUT_DIR) / "runs"}')
        run = runs[-1]
    main(run)
