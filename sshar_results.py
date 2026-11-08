"""
Xuất kết quả theo kaggle_run_convention.md mục 5:

results/<RUN>/
  summary.txt, summary.csv, config.json
  <MODEL>/
    per_seed.csv
    seed_<s>/
      metrics.json, history.csv, curve.png,
      cm.png, cm_normalized.png, confusion_matrix.csv, per_class_metrics.csv, predictions.csv
      val/          # cùng các file trên, đo trên val (cùng checkpoint)
      (+ file gốc của main.py: train.log, summary.json, best_model.pth, plots/, ...)

File ở seed_<s>/ là kết quả trên TEST, seed_<s>/val/ là trên VAL (cùng checkpoint chọn theo val).
"""
import csv
import json
import shutil
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = ('accuracy', 'precision', 'recall', 'f1')


def _read_cm(path):
    with open(path, encoding='utf-8') as f:
        rows = list(csv.reader(f))
    names = rows[0][1:]
    cm = np.array([[int(x) for x in r[1:]] for r in rows[1:]], dtype=np.int64)
    return names, cm


def _per_class(cm, names, path):
    rows = []
    for c, n in enumerate(names):
        tp = cm[c, c]
        p = tp / max(cm[:, c].sum(), 1)
        r = tp / max(cm[c].sum(), 1)
        f = 2 * p * r / max(p + r, 1e-12)
        rows.append([n, f'{p:.4f}', f'{r:.4f}', f'{f:.4f}', int(cm[c].sum())])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['class', 'precision', 'recall', 'f1', 'support'])
        w.writerows(rows)


def _plot_cm(cm, names, path, normalize, title):
    m = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(m, cmap='Blues', vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(names)), names, rotation=45, ha='right')
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel('Dự đoán')
    ax.set_ylabel('Thực tế')
    ax.set_title(title)
    thr = (m.max() if m.size else 0) / 2
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f'{m[i, j]:.2f}' if normalize else int(m[i, j]),
                    ha='center', va='center', fontsize=8, color='white' if m[i, j] > thr else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_curve(history_csv, path, best_epoch):
    with open(history_csv, encoding='utf-8') as f:
        h = list(csv.DictReader(f))
    if not h:
        return
    ep = [int(r['epoch']) for r in h]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, name in ((axs[0], 'loss', 'Loss'), (axs[1], 'acc', 'Accuracy')):
        ax.plot(ep, [float(r[f'train_{key}']) for r in h], label='train')
        ax.plot(ep, [float(r[f'val_{key}']) for r in h], label='val')
        if best_epoch:
            ax.axvline(best_epoch, color='gray', ls='--', lw=1, label=f'best (epoch {best_epoch})')
        ax.set_xlabel('epoch')
        ax.set_title(name)
        ax.grid(alpha=.3)
        ax.legend()
    axs[0].set_yscale('log')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _export_set(seed_dir, out_dir, name, res, summary, seed):
    """confusion_matrix / predictions / per_class / cm png / metrics.json của 1 tập (val hoặc test)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seed_dir / f'confusion_matrix_{name}_final.csv', out_dir / 'confusion_matrix.csv')
    shutil.copyfile(seed_dir / f'{name}_predictions_final.csv', out_dir / 'predictions.csv')
    names, cm = _read_cm(out_dir / 'confusion_matrix.csv')
    _per_class(cm, names, out_dir / 'per_class_metrics.csv')
    _plot_cm(cm, names, out_dir / 'cm.png', False, f'{name.upper()} (số mẫu)')
    _plot_cm(cm, names, out_dir / 'cm_normalized.png', True, f'{name.upper()} (chuẩn hoá theo hàng)')
    metrics = {'eval_set': name, 'seed': seed,
               'accuracy': res['acc'], 'precision': res['macro_precision'],
               'recall': res['macro_recall'], 'f1': res['macro_f1'], 'n': res['n'],
               'best_epoch': summary['best_epoch'], 'epochs_trained': summary['epochs_done'],
               'stop_reason': summary['stop_reason'], 'patience': summary.get('patience'),
               'view_acc': res.get('view_acc'), 'class_acc': res.get('class_acc'),
               'total_time_s': summary.get('total_time_s')}
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def export_seed(seed_dir, seed):
    """Tạo các file theo quy ước trong seed_<s>/ từ output gốc của main.py. Trả về metrics chính."""
    seed_dir = Path(seed_dir)
    summary = json.load(open(seed_dir / 'summary.json', encoding='utf-8'))
    final = summary['final']
    _plot_curve(seed_dir / 'history.csv', seed_dir / 'curve.png', summary.get('best_epoch'))
    main_m = _export_set(seed_dir, seed_dir, 'test', final['test'], summary, seed)
    _export_set(seed_dir, seed_dir / 'val', 'val', final['val'], summary, seed)
    if (seed_dir / 'curve.png').exists():
        shutil.copyfile(seed_dir / 'curve.png', seed_dir / 'val' / 'curve.png')
    main_m['val_accuracy'] = final['val']['acc']
    main_m['val_f1'] = final['val']['macro_f1']
    return main_m


def _fmt(vals):
    vals = np.asarray(vals, dtype=float)
    return f'{vals.mean():.4f} ± {vals.std():.4f}' if len(vals) > 1 else f'{vals.mean():.4f}'


def write_run_summary(run_root, model, rows, cfg):
    """per_seed.csv, summary.csv, summary.txt, config.json ở results/<RUN>/."""
    run_root = Path(run_root)
    cols = ['seed', 'eval_set', *METRICS, 'best_epoch', 'epochs_trained', 'stop_reason']
    cols += ['val_accuracy', 'val_f1']
    with open(run_root / model / 'per_seed.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    agg = {m: [r[m] for r in rows] for m in METRICS}
    with open(run_root / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'split', 'user_case', 'eval_set', 'n_seeds', 'seeds',
                    *[f'{m}_mean' for m in METRICS], *[f'{m}_std' for m in METRICS]])
        w.writerow([model, cfg['split'], cfg.get('user_case'),
                    rows[0]['eval_set'] if rows else '', len(rows), ' '.join(str(r['seed']) for r in rows),
                    *[f'{np.mean(agg[m]):.4f}' for m in METRICS],
                    *[f'{np.std(agg[m]):.4f}' for m in METRICS]])

    with open(run_root / 'config.json', 'w', encoding='utf-8') as f:
        json.dump({'model': model, **cfg}, f, indent=2, ensure_ascii=False)

    eval_set = (rows[0]['eval_set'] if rows else '?').upper()
    lines = [f"RUN        : {run_root.name}",
             f"MODEL      : {model}",
             *[f"{k:<11}: {v}" for k, v in cfg.items()],
             "",
             f"Kết quả trên {eval_set} (macro, checkpoint tốt nhất theo val):",
             f"  {'seed':>6} | {'acc':>7} | {'prec':>7} | {'rec':>7} | {'f1':>7} | best_ep | epochs | dừng"]
    for r in rows:
        lines.append(f"  {r['seed']:>6} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                     f"{r['f1']:.4f} | {r['best_epoch']:>7} | {r['epochs_trained']:>6} | {r['stop_reason']}")
    if rows:
        lines.append("  " + " | ".join(f"{m} {_fmt(agg[m])}" for m in METRICS) + f"  (n_seeds = {len(rows)})")
        lines.append(f"  VAL (cùng checkpoint): acc {_fmt([r['val_accuracy'] for r in rows])} | "
                     f"f1 {_fmt([r['val_f1'] for r in rows])}")
    txt = '\n'.join(lines) + '\n'
    (run_root / 'summary.txt').write_text(txt, encoding='utf-8')
    return txt
