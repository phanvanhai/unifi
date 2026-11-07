"""
UniFi — Module 3: Loss + Training Loop
=======================================
Liu et al., Proc. ACM IMWUT, vol. 7, no. 4, 2023.
DOI: 10.1145/3631429

Hàm mất mát tổng:
    L = L_global + β₁·Σ Lⁱ_view + β₂·L_pcon
    β₁ = 1,  β₂ = 0.05

    L_global  : CE trên z_global → classifier
    Lⁱ_view   : CE trên zⁱ per view → cùng classifier (shared)
    L_pcon    : proxy contrastive trên z_global
"""

import contextlib
import csv
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger('unifi')


# =============================================================================
# Proxy Contrastive Loss
# =============================================================================

class ProxyContrastiveLoss(nn.Module):
    """
    Proxy-based Contrastive Loss — công thức (25) trong bài.

    Ý tưởng:
        Class proxy wc = trọng số hàng c của FC classifier.
        Proxy tự nhiên hội tụ về "trung tâm" lớp c trong quá trình train.

        Với mỗi mẫu zᵢ thuộc lớp c:
            - Kéo zᵢ gần ŵc  (proxy đúng lớp)
            - Đẩy zᵢ xa ŵk   (proxy lớp khác, k≠c)
            - Đẩy zᵢ xa žⱼ   (sample khác trong batch, j≠i)

        Lᵢ = -log( e^(ŵcᵀžᵢ/τ) / [ Σₖ e^(ŵkᵀžᵢ/τ) + Σ_{j≠i} e^(žᵢᵀžⱼ/τ) ] )
    """

    def __init__(self, D=64, proj_dim=128, tau=0.07):
        """
        Tham số:
            D        : chiều đặc trưng input (= 64)
            proj_dim : chiều projection space (không báo cáo, thử 128)
            tau      : temperature (không báo cáo, thử 0.07)
        """
        super().__init__()
        self.tau = tau

        # Projection head cho sample z: D → proj_dim → proj_dim
        self.proj_sample = nn.Sequential(
            nn.Linear(D, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

        # Projection head cho proxy w — riêng biệt với proj_sample
        self.proj_proxy = nn.Sequential(
            nn.Linear(D, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, z, labels, proxy_weights):
        """
        Tham số:
            z             : Tensor [B, D]    embedding mẫu
            labels        : Tensor [B]       nhãn lớp (bắt đầu từ 0)
            proxy_weights : Tensor [C, D]    = classifier.weight

        Trả về:
            loss : scalar Tensor
        """
        B = z.shape[0]

        # L2 normalize → dot product = cosine similarity ∈ [-1, 1]
        z_norm = F.normalize(self.proj_sample(z), dim=-1)              # [B, P]
        w_norm = F.normalize(self.proj_proxy(proxy_weights), dim=-1)   # [C, P]

        sim_zw = z_norm @ w_norm.T / self.tau    # [B, C]  sample-to-proxy
        sim_zz = z_norm @ z_norm.T / self.tau    # [B, B]  sample-to-sample

        # Bỏ j = i khỏi mẫu số: đặt đường chéo = -inf → e^(-inf) = 0
        eye    = torch.eye(B, dtype=torch.bool, device=z.device)
        sim_zz = sim_zz.masked_fill(eye, float('-inf'))

        # Mẫu số: C proxy (gồm cả proxy đúng lớp) + B-1 mẫu khác
        all_terms = torch.cat([sim_zw, sim_zz], dim=1)                 # [B, C+B]
        numerator = sim_zw.gather(1, labels[:, None]).squeeze(1)       # [B]

        # -log(e^num / Σ e^all) = logsumexp(all) - num  (an toàn overflow)
        loss = torch.logsumexp(all_terms, dim=1) - numerator           # [B]
        return loss.mean()


# =============================================================================
# Hàm tính toàn bộ loss
# =============================================================================

def compute_loss(logit_global, logit_views, z_global,
                 labels, pcon_loss_fn, classifier,
                 beta1=1.0, beta2=0.05):
    """
    Tính hàm mất mát tổng:
        L = L_global + β₁·Σᵢ Lⁱ_view + β₂·L_pcon

    Trả về:
        loss, L_global, L_view (tổng V view), L_pcon  — scalar Tensor
    """
    # Tính loss ở fp32 (khi AMP bật, logit là fp16)
    logit_global = logit_global.float()
    logit_views  = logit_views.float()
    z_global     = z_global.float()

    B, V, C = logit_views.shape

    L_global = F.cross_entropy(logit_global, labels)

    # Σᵢ CE(view i) = V · mean CE trên cả B·V cặp (mỗi view có đúng B mẫu)
    L_view = F.cross_entropy(logit_views.reshape(B * V, C),
                             labels.repeat_interleave(V)) * V
    # (reshape [B, V, C] → [B*V, C]: hàng b*V+i ↔ nhãn labels[b] → repeat_interleave)

    # proxy_weights = classifier.weight [C, D], hàng c = proxy của lớp c
    L_pcon = pcon_loss_fn(z_global, labels, classifier.weight.float())

    loss = L_global + beta1 * L_view + beta2 * L_pcon
    return loss, L_global, L_view, L_pcon


# =============================================================================
# Tiện ích
# =============================================================================

def unwrap(model):
    """Bỏ lớp DataParallel (nếu có) để truy cập .classifier, .D, state_dict gốc."""
    return model.module if isinstance(model, nn.DataParallel) else model


def autocast(amp):
    """fp16 autocast trên CUDA khi amp=True, ngược lại không làm gì."""
    return torch.autocast('cuda', dtype=torch.float16) if amp else contextlib.nullcontext()


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):          # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _read_first_int(paths):
    for p in paths:
        try:
            with open(p) as f:
                v = f.read().split()[0]
            if v.isdigit():
                return int(v)
        except (OSError, IndexError):
            pass
    return None


def ram_str():
    """RAM của container (cgroup, cùng số Kaggle dùng để kill) + /dev/shm (batch đang chờ)."""
    used  = _read_first_int(['/sys/fs/cgroup/memory.current',
                             '/sys/fs/cgroup/memory/memory.usage_in_bytes'])
    limit = _read_first_int(['/sys/fs/cgroup/memory.max',
                             '/sys/fs/cgroup/memory/memory.limit_in_bytes'])
    parts = []
    if used is not None:
        s = f"ram {used / 2**30:.1f}"
        if limit and limit < 2**50:
            s += f"/{limit / 2**30:.0f}G"
        else:
            s += "G"
        parts.append(s)
    try:
        st = os.statvfs('/dev/shm')
        parts.append(f"shm {(st.f_blocks - st.f_bfree) * st.f_frsize / 2**30:.1f}G")
    except (OSError, AttributeError):
        pass
    return (' | ' + ' '.join(parts)) if parts else ''


def gpu_mem_str(device):
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        return ram_str()
    parts = []
    for d in range(torch.cuda.device_count()):
        alloc = torch.cuda.max_memory_allocated(d) / 2**30
        total = torch.cuda.get_device_properties(d).total_memory / 2**30
        parts.append(f"gpu{d} {alloc:.1f}/{total:.0f}G")
    return ' | ' + ' '.join(parts) + ram_str()


def fmt_time(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


# =============================================================================
# Training loop
# =============================================================================

def train_one_epoch(model, dataloader, optimizer,
                    pcon_loss_fn, device,
                    beta1=1.0, beta2=0.05,
                    scaler=None, amp=False, grad_clip=None,
                    epoch=0, log_interval=20):
    """
    Chạy một epoch training.

    Trả về:
        dict: loss trung bình (loss, L_global, L_view, L_pcon), acc, thời gian
              chờ dữ liệu / tính toán, số mẫu/giây
    """
    model.train()
    pcon_loss_fn.train()
    core = unwrap(model)

    sums      = {'loss': 0.0, 'L_global': 0.0, 'L_view': 0.0, 'L_pcon': 0.0}
    n_batches = 0
    correct   = 0
    total     = 0
    n_nonfinite = 0
    t_data = t_comp = 0.0
    n_total_batches = len(dataloader)

    t_epoch = t_last = time.time()
    for X_f, X_t, labels in dataloader:
        t_ready = time.time()
        t_data += t_ready - t_last

        # X_f, X_t: [B, V, 3, H, W] | labels: [B]
        # dataset trả fp16 (đỡ RAM/shm) → về fp32 trên GPU
        X_f    = X_f.to(device, non_blocking=True).float()
        X_t    = X_t.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)

        # ── Forward (AMP: fp16 cho ResNet, loss tính fp32) ────────────
        with autocast(amp):
            logit_global, logit_views, z_global = model(X_t, X_f)

        loss, L_g, L_v, L_p = compute_loss(
            logit_global, logit_views, z_global, labels,
            pcon_loss_fn, core.classifier, beta1, beta2)

        if not torch.isfinite(loss):
            n_nonfinite += 1
            log.warning(f"  [epoch {epoch}] batch {n_batches + 1}: loss = {loss.item()} → bỏ qua batch")
            optimizer.zero_grad(set_to_none=True)
            t_last = time.time()
            continue

        # ── Backward ──────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g['params']], grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # ── Ghi nhận ──────────────────────────────────────────────────
        sums['loss']     += loss.item()      # .item() đồng bộ GPU → t_comp chính xác
        sums['L_global'] += L_g.item()
        sums['L_view']   += L_v.item()
        sums['L_pcon']   += L_p.item()
        n_batches        += 1
        correct += (logit_global.argmax(-1) == labels).sum().item()
        total   += labels.shape[0]

        t_last  = time.time()
        t_comp += t_last - t_ready

        if log_interval and (n_batches % log_interval == 0 or n_batches == n_total_batches):
            el = t_last - t_epoch
            log.info(
                f"  [epoch {epoch}] batch {n_batches:4d}/{n_total_batches} | "
                f"loss {sums['loss'] / n_batches:.4f} "
                f"(G {sums['L_global'] / n_batches:.3f} V {sums['L_view'] / n_batches:.3f} "
                f"P {sums['L_pcon'] / n_batches:.3f}) | acc {correct / max(total, 1):.4f} | "
                f"{total / max(el, 1e-9):.1f} mẫu/s | data {t_data / el * 100:.0f}% | "
                f"còn ~{fmt_time(el / n_batches * (n_total_batches - n_batches))}"
                f"{gpu_mem_str(device)}")

    out = {k: v / max(n_batches, 1) for k, v in sums.items()}
    out['acc']          = correct / max(total, 1)
    out['time_data_s']  = t_data
    out['time_comp_s']  = t_comp
    out['samples_per_s'] = total / max(time.time() - t_epoch, 1e-9)
    out['nonfinite']    = n_nonfinite
    return out


def evaluate(model, dataloader, device,
             pcon_loss_fn=None, beta1=1.0, beta2=0.05, amp=False):
    """
    Đánh giá model trên test set.

    Trả về dict:
        acc                           : accuracy của nhánh global
        loss, L_global, L_view, L_pcon: loss trung bình, cùng công thức với train
                                        (chỉ có khi truyền pcon_loss_fn)
        y_true     : ndarray [N]      nhãn thật
        y_pred     : ndarray [N]      dự đoán global
        probs      : ndarray [N, C]   softmax của logit_global
        view_preds : ndarray [N, V]   dự đoán của từng view (logit_views)
    Thứ tự mẫu = thứ tự dataloader (test loader không shuffle).
    """
    model.eval()
    if pcon_loss_fn is not None:
        pcon_loss_fn.eval()
    core = unwrap(model)

    sums      = {'loss': 0.0, 'L_global': 0.0, 'L_view': 0.0, 'L_pcon': 0.0}
    n_batches = 0
    y_true, probs, view_preds = [], [], []
    t0 = time.time()

    with torch.no_grad():
        for X_f, X_t, labels in dataloader:
            X_f    = X_f.to(device, non_blocking=True).float()
            X_t    = X_t.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)

            with autocast(amp):
                logit_global, logit_views, z_global = model(X_t, X_f)
            logit_global = logit_global.float()

            if pcon_loss_fn is not None:
                loss, L_g, L_v, L_p = compute_loss(
                    logit_global, logit_views, z_global, labels,
                    pcon_loss_fn, core.classifier, beta1, beta2)
                sums['loss']     += loss.item()
                sums['L_global'] += L_g.item()
                sums['L_view']   += L_v.item()
                sums['L_pcon']   += L_p.item()
                n_batches        += 1

            y_true.append(labels.cpu())
            probs.append(F.softmax(logit_global, dim=-1).cpu())
            view_preds.append(logit_views.argmax(-1).cpu())   # [B, V]

    y_true     = torch.cat(y_true).numpy()
    probs      = torch.cat(probs).numpy()
    view_preds = torch.cat(view_preds).numpy()
    y_pred     = probs.argmax(-1)

    out = {
        'acc'       : float((y_pred == y_true).mean()) if len(y_true) else 0.0,
        'y_true'    : y_true,
        'y_pred'    : y_pred,
        'probs'     : probs,
        'view_preds': view_preds,
        'time_s'    : time.time() - t0,
    }
    if n_batches:
        out.update({k: v / n_batches for k, v in sums.items()})
    return out


# =============================================================================
# Chỉ số phụ
# =============================================================================

def per_view_acc(res):
    return [float((res['view_preds'][:, v] == res['y_true']).mean())
            for v in range(res['view_preds'].shape[1])]


def per_class_acc(res, num_classes):
    out = []
    for c in range(num_classes):
        m = res['y_true'] == c
        out.append(float((res['y_pred'][m] == c).mean()) if m.any() else float('nan'))
    return out


def macro_f1(cm):
    f1s = []
    for c in range(cm.shape[0]):
        tp   = cm[c, c]
        prec = tp / max(cm[:, c].sum(), 1)
        rec  = tp / max(cm[c].sum(), 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(f1s))


def confusion(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def format_confusion(cm, names):
    w = max(8, max(len(n) for n in names) + 1)
    lines = [' ' * w + ''.join(f'{n[:w - 1]:>{w}}' for n in names)]
    for c, n in enumerate(names):
        lines.append(f'{n[:w - 1]:>{w}}' + ''.join(f'{x:>{w}d}' for x in cm[c]))
    return '\n'.join(lines)


# =============================================================================
# Ghi log ra file
# =============================================================================

HISTORY_COLS = ['epoch', 'lr', 'time_s', 'train_time_s', 'test_time_s',
                'data_wait_s', 'samples_per_s',
                'train_loss', 'train_L_global', 'train_L_view', 'train_L_pcon', 'train_acc',
                'test_loss', 'test_L_global', 'test_L_view', 'test_L_pcon', 'test_acc',
                'test_macro_f1', 'test_view_acc_mean', 'gpu_max_mem_gb', 'is_best']


def save_predictions(path, res, meta=None, class_names=None):
    """
    Ghi dự đoán từng mẫu ra CSV — dùng để vẽ confusion matrix, phân tích lỗi
    theo user / location / orientation, so sánh độ chính xác từng view.

    Cột: idx, [meta...], y_true, y_pred, correct, [true_name, pred_name],
         view1_pred..viewV_pred, prob_0..prob_{C-1}
    """
    N, C = res['probs'].shape
    V    = res['view_preds'].shape[1]
    meta_keys = list(meta[0].keys()) if meta else []

    header = (['idx'] + meta_keys + ['y_true', 'y_pred', 'correct']
              + (['true_name', 'pred_name'] if class_names else [])
              + [f'view{v + 1}_pred' for v in range(V)]
              + [f'prob_{c}' for c in range(C)])

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(N):
            yt, yp = int(res['y_true'][i]), int(res['y_pred'][i])
            row = [i] + [meta[i][k] for k in meta_keys]
            row += [yt, yp, int(yt == yp)]
            if class_names:
                row += [class_names[yt], class_names[yp]]
            row += [int(x) for x in res['view_preds'][i]]
            row += [f'{p:.6f}' for p in res['probs'][i]]
            w.writerow(row)


def save_confusion_matrix(path, y_true, y_pred, num_classes, class_names=None):
    """Ma trận nhầm lẫn [C, C]: hàng = nhãn thật, cột = dự đoán."""
    cm = confusion(y_true, y_pred, num_classes)
    names = class_names or [str(c) for c in range(num_classes)]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['true/pred'] + list(names))
        for c in range(num_classes):
            w.writerow([names[c]] + cm[c].tolist())
    return cm


def train(model, train_loader, test_loader,
          num_epochs=100, lr=0.005,
          beta1=1.0, beta2=0.05, device='cpu',
          save_path='best_model.pth',
          log_dir=None, test_meta=None, class_names=None,
          amp=False, multi_gpu=False, weight_decay=0.0, grad_clip=None,
          time_limit_s=None, log_interval=20, extra_summary=None):
    """
    Training loop đầy đủ.

    Tham số:
        model        : UniFi model
        train_loader : DataLoader cho training set
        test_loader  : DataLoader cho test set (KHÔNG shuffle) — chọn checkpoint theo test acc cao nhất
        num_epochs   : số epoch
        lr           : learning rate (= 0.005, báo cáo)
        beta1, beta2 : hệ số loss (= 1.0, 0.05, báo cáo)
        device       : 'cuda' hoặc 'cpu'
        save_path    : nơi lưu checkpoint tốt nhất
        log_dir      : thư mục ghi log (None → không ghi file). Tạo ra:
            history.csv               loss/acc train + test mỗi epoch (flush mỗi
                                      epoch → Kaggle bị ngắt vẫn còn log)
            test_predictions_best.csv  dự đoán từng mẫu test tại epoch tốt nhất
            test_predictions_last.csv  dự đoán từng mẫu test tại epoch cuối
            confusion_matrix_best.csv / confusion_matrix_last.csv
            last_model.pth            checkpoint epoch cuối
            summary.json              best acc, best epoch, tổng thời gian
        test_meta    : list dict thông tin từng mẫu test (cùng thứ tự test_loader)
        class_names  : list tên lớp (ghi kèm vào CSV)
        amp          : mixed precision fp16 (chỉ có tác dụng trên CUDA)
        multi_gpu    : bọc DataParallel khi có > 1 GPU (Kaggle T4 x2)
        time_limit_s : dừng sớm (vẫn ghi đủ kết quả) nếu epoch kế tiếp có thể vượt
                       giới hạn này — tránh bị Kaggle cắt ngang ở mốc 12 giờ
    """
    model = model.to(device)
    core  = model
    amp   = bool(amp and str(device).startswith('cuda'))

    if multi_gpu and str(device).startswith('cuda') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log.info(f"DataParallel trên {torch.cuda.device_count()} GPU")

    num_classes = core.classifier.out_features
    names = class_names or [str(c) for c in range(num_classes)]

    # Proxy contrastive loss
    pcon_loss_fn = ProxyContrastiveLoss(D=core.D, proj_dim=128, tau=0.07).to(device)

    # Adam optimizer — học tất cả parameters của model VÀ pcon
    params = list(core.parameters()) + list(pcon_loss_fn.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scaler = make_grad_scaler(amp)

    n_model = sum(p.numel() for p in core.parameters())
    n_pcon  = sum(p.numel() for p in pcon_loss_fn.parameters())
    log.info(f"Tham số: model {n_model:,} | pcon head {n_pcon:,} | "
             f"AMP {amp} | lr {lr} | wd {weight_decay} | grad_clip {grad_clip}")
    log.info(f"Batch/epoch: train {len(train_loader)} | test {len(test_loader)}")

    hist_f = writer = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        hist_f = open(log_dir / 'history.csv', 'w', newline='', encoding='utf-8')
        writer = csv.DictWriter(hist_f, fieldnames=HISTORY_COLS)
        writer.writeheader()
        hist_f.flush()

    best_test_acc = -1.0
    best_epoch    = 0
    best_f1       = float('nan')
    epochs_done   = 0
    t_start       = time.time()
    res           = None
    epoch_times   = []
    stop_reason   = 'completed'

    try:
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            if str(device).startswith('cuda'):
                for d in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(d)

            tr = train_one_epoch(model, train_loader, optimizer, pcon_loss_fn, device,
                                 beta1, beta2, scaler=scaler, amp=amp, grad_clip=grad_clip,
                                 epoch=epoch, log_interval=log_interval)
            t_train = time.time() - t0

            # Test (cùng công thức loss để so sánh với train)
            res = evaluate(model, test_loader, device, pcon_loss_fn, beta1, beta2, amp=amp)
            cm  = confusion(res['y_true'], res['y_pred'], num_classes)
            f1  = macro_f1(cm)
            v_acc = per_view_acc(res)
            c_acc = per_class_acc(res, num_classes)

            is_best = res['acc'] > best_test_acc
            dt = time.time() - t0
            epoch_times.append(dt)
            epochs_done = epoch
            gpu_gb = (max(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 2**30
                      if str(device).startswith('cuda') else 0.0)

            elapsed = time.time() - t_start
            eta     = np.mean(epoch_times[-5:]) * (num_epochs - epoch)
            log.info(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train loss {tr['loss']:.4f} (G {tr['L_global']:.3f} V {tr['L_view']:.3f} "
                f"P {tr['L_pcon']:.3f}) acc {tr['acc']:.4f} | "
                f"Test loss {res.get('loss', float('nan')):.4f} acc {res['acc']:.4f} "
                f"F1 {f1:.4f} | {fmt_time(dt)} (train {fmt_time(t_train)}, "
                f"chờ data {fmt_time(tr['time_data_s'])}) | đã chạy {fmt_time(elapsed)}, "
                f"còn ~{fmt_time(eta)}{ram_str()}" + ("  *best" if is_best else ""))
            log.info("          test acc theo view: "
                     + ' '.join(f"r{v + 1}={a:.3f}" for v, a in enumerate(v_acc))
                     + f" | theo lớp: "
                     + ' '.join(f"{n}={a:.3f}" for n, a in zip(names, c_acc)))
            if tr['nonfinite']:
                log.warning(f"          {tr['nonfinite']} batch có loss NaN/Inf đã bị bỏ qua")

            if writer is not None:
                writer.writerow({
                    'epoch'         : epoch,
                    'lr'            : optimizer.param_groups[0]['lr'],
                    'time_s'        : round(dt, 2),
                    'train_time_s'  : round(t_train, 2),
                    'test_time_s'   : round(res['time_s'], 2),
                    'data_wait_s'   : round(tr['time_data_s'], 2),
                    'samples_per_s' : round(tr['samples_per_s'], 2),
                    'train_loss'    : tr['loss'],
                    'train_L_global': tr['L_global'],
                    'train_L_view'  : tr['L_view'],
                    'train_L_pcon'  : tr['L_pcon'],
                    'train_acc'     : tr['acc'],
                    'test_loss'     : res.get('loss'),
                    'test_L_global' : res.get('L_global'),
                    'test_L_view'   : res.get('L_view'),
                    'test_L_pcon'   : res.get('L_pcon'),
                    'test_acc'      : res['acc'],
                    'test_macro_f1' : f1,
                    'test_view_acc_mean': float(np.mean(v_acc)),
                    'gpu_max_mem_gb': round(gpu_gb, 3),
                    'is_best'       : int(is_best),
                })
                hist_f.flush()

            # Lưu model + dự đoán tại epoch có test acc cao nhất
            if is_best:
                best_test_acc = res['acc']
                best_epoch    = epoch
                best_f1       = f1
                torch.save(core.state_dict(), save_path)
                if log_dir is not None:
                    save_predictions(log_dir / 'test_predictions_best.csv',
                                     res, test_meta, class_names)
                    save_confusion_matrix(log_dir / 'confusion_matrix_best.csv',
                                          res['y_true'], res['y_pred'],
                                          num_classes, class_names)

            # Dừng sớm nếu epoch kế tiếp có thể vượt giới hạn thời gian
            if time_limit_s and epoch < num_epochs:
                need = max(epoch_times[-3:]) * 1.15
                if time.time() - t_start + need > time_limit_s:
                    stop_reason = 'time_limit'
                    log.warning(f"Dừng sớm sau epoch {epoch}: đã chạy {fmt_time(time.time() - t_start)}, "
                                f"epoch kế tiếp cần ~{fmt_time(need)} > giới hạn {fmt_time(time_limit_s)}")
                    break
    except KeyboardInterrupt:
        stop_reason = 'interrupted'
        log.warning("Bị ngắt (KeyboardInterrupt) — vẫn ghi kết quả đã có")
    except Exception:
        stop_reason = 'error'
        raise
    finally:
        # Ghi kết quả cả khi bị ngắt giữa chừng (Ctrl+C, hết giờ Kaggle, lỗi...)
        if hist_f is not None:
            hist_f.close()
        if log_dir is not None and res is not None:
            torch.save(core.state_dict(), log_dir / 'last_model.pth')
            save_predictions(log_dir / 'test_predictions_last.csv',
                             res, test_meta, class_names)
            cm_last = save_confusion_matrix(log_dir / 'confusion_matrix_last.csv',
                                            res['y_true'], res['y_pred'],
                                            num_classes, class_names)
            summary = {
                'stop_reason'    : stop_reason,
                'best_test_acc'  : best_test_acc,
                'best_test_macro_f1': best_f1,
                'best_epoch'     : best_epoch,
                'last_test_acc'  : res['acc'],
                'last_test_macro_f1': macro_f1(cm_last),
                'last_test_view_acc': per_view_acc(res),
                'last_test_class_acc': dict(zip(names, per_class_acc(res, num_classes))),
                'epochs_done'    : epochs_done,
                'num_epochs'     : num_epochs,
                'mean_epoch_time_s': round(float(np.mean(epoch_times)), 1) if epoch_times else None,
                'total_time_s'   : round(time.time() - t_start, 1),
                'best_model_path': str(save_path),
            }
            if extra_summary:
                summary.update(extra_summary)
            with open(log_dir / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"Best test acc: {best_test_acc:.4f} (macro F1 {best_f1:.4f}) tại epoch {best_epoch}"
             f" | dừng: {stop_reason} | tổng {fmt_time(time.time() - t_start)}")
    if res is not None:
        log.info("Confusion matrix (epoch cuối, hàng = thật, cột = dự đoán):\n"
                 + format_confusion(confusion(res['y_true'], res['y_pred'], num_classes), names))
    log.info(f"Model tốt nhất: {save_path}")
    if log_dir is not None:
        log.info(f"Log + kết quả: {log_dir}")
    return {'best_test_acc': best_test_acc, 'best_epoch': best_epoch, 'stop_reason': stop_reason}


# =============================================================================
# Test loss
# =============================================================================

def _pcon_reference(pcon, z, labels, w):
    """Bản vòng lặp gốc (theo từng mẫu) — dùng để kiểm tra bản vector hoá."""
    z_n = F.normalize(pcon.proj_sample(z), dim=-1)
    w_n = F.normalize(pcon.proj_proxy(w), dim=-1)
    s_zw, s_zz = z_n @ w_n.T / pcon.tau, z_n @ z_n.T / pcon.tau
    loss = 0.0
    for i in range(z.shape[0]):
        c = labels[i].item()
        terms = torch.cat([s_zw[i], s_zz[i, :i], s_zz[i, i + 1:]])
        loss += -(s_zw[i, c] - torch.logsumexp(terms, dim=0))
    return loss / z.shape[0]


def test_loss():
    print("=== Test Proxy Contrastive Loss ===\n")

    B, D, C, V = 8, 64, 6, 6
    z_global     = torch.randn(B, D)
    labels       = torch.randint(0, C, (B,))
    classifier   = nn.Linear(D, C, bias=False)
    logit_global = torch.randn(B, C)
    logit_views  = torch.randn(B, V, C)
    pcon_loss_fn = ProxyContrastiveLoss(D=D, proj_dim=128, tau=0.07)

    L_pcon = pcon_loss_fn(z_global, labels, classifier.weight)
    L_ref  = _pcon_reference(pcon_loss_fn, z_global, labels, classifier.weight)
    print(f"L_pcon (vector hoá) {L_pcon.item():.6f} | (vòng lặp) {L_ref.item():.6f}")

    loss, L_g, L_v, L_p = compute_loss(logit_global, logit_views, z_global, labels,
                                       pcon_loss_fn, classifier)
    L_v_ref = sum(F.cross_entropy(logit_views[:, i], labels) for i in range(V))
    print(f"L_view (vector hoá) {L_v.item():.6f} | (vòng lặp) {L_v_ref.item():.6f}")
    print(f"L_total: {loss.item():.4f} = {L_g.item() + L_v.item() + 0.05 * L_p.item():.4f}")


if __name__ == '__main__':
    test_loss()
