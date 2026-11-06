"""
UniFi — Module 2: Feature Fusion
=================================
Liu et al., Proc. ACM IMWUT, vol. 7, no. 4, 2023.
DOI: 10.1145/3631429

Pipeline khối này:
    F_inter [B, V, D, W']
        ↓ tách per view → Fⁱ_inter [B, D, W']
        ↓ Multi-scale Conv1D (kernel 3, 5, 7)
        ↓ GlobalAvgPool → pⁱₖ [B, D]  per scale k
        ↓ Segment-level attention → zⁱ [B, D]  per view
        ↓ View-level attention
    z_global [B, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """
    Tổng hợp đặc trưng từ nhiều scale và nhiều views.

    Segment-level: học xem scale (kernel size) nào
                   phù hợp với cử chỉ đang xảy ra.
                   Kernel nhỏ → nắm nét ngắn/nhanh.
                   Kernel lớn → nắm xu hướng dài/chậm.

    View-level:    học xem view nào đang cung cấp
                   thông tin tốt hơn (ít bị che khuất,
                   ít multipath hơn).
    """

    def __init__(self, D=64, kernel_sizes=(3, 5, 7)):
        """
        Tham số:
            D            : chiều đặc trưng (= 64)
            kernel_sizes : tập kernel size cho multi-scale conv
                           không được báo cáo, thử (3, 5, 7)
        """
        super().__init__()

        self.kernel_sizes = kernel_sizes
        K_scale = len(kernel_sizes)   # số scales = 3

        # ── Multi-scale Conv1D ────────────────────────────────────────
        # Một Conv1D riêng cho mỗi kernel size
        # ModuleList: list các nn.Module, tự đăng ký vào parameters()
        # Nếu dùng list Python thông thường thì PyTorch không biết
        # các module trong đó → optimizer không cập nhật
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=D,
                out_channels=D,
                kernel_size=k,
                padding=k // 2,   # same padding: giữ nguyên W'
                                  # padding = k//2:
                                  #   k=3 → pad=1, W'_out = W'
                                  #   k=5 → pad=2, W'_out = W'
                                  #   k=7 → pad=3, W'_out = W'
                bias=False
            )
            for k in kernel_sizes
        ])
        # Mỗi conv: [B, D, W'] → [B, D, W']

        # ── Segment-level Attention ───────────────────────────────────
        # Nhận concat K_scale vectors [B, K_scale*D]
        # Trả về K_scale trọng số [B, K_scale]
        # Dùng để weighted sum các scale
        self.seg_fc = nn.Linear(K_scale * D, K_scale)
        # Lưu ý: đây là FC layer đơn giản, không có hidden layer
        # Input:  [B, K_scale*D] = [B, 3*64] = [B, 192]
        # Output: [B, K_scale]   = [B, 3]

        # ── View-level Attention ──────────────────────────────────────
        # Nhận đặc trưng mỗi view [B, D]
        # Trả về 1 scalar trọng số per view
        self.view_fc = nn.Linear(D, 1)
        # Input:  [B, D]  = [B, 64]
        # Output: [B, 1]  → squeeze → scalar per view

    def forward(self, F_inter):
        """
        Tham số:
            F_inter : Tensor [B, V, D, W']

        Trả về:
            z_views  : Tensor [B, V, D]  đặc trưng per view
            z_global : Tensor [B, D]     đặc trưng toàn cục
        """
        B, V, D, W = F_inter.shape

        # ── Xử lý từng view độc lập ───────────────────────────────────
        z_list = []   # sẽ chứa V tensor [B, D]

        for i in range(V):

            # Lấy view i của toàn batch
            # F_inter[:, i, :, :] → [B, D, W']
            Fi = F_inter[:, i, :, :]   # [B, D, W']
            # Conv1d expect [B, C, L] = [B, D, W'] → đúng format

            # ── Multi-scale Conv1D + GlobalAvgPool ────────────────────
            # Chạy từng conv, pool về 1 vector D chiều
            p_scales = []   # sẽ chứa K_scale tensor [B, D]

            for conv in self.convs:
                # Conv1D trượt dọc W', padding giữ nguyên W'
                p_k = conv(Fi)              # [B, D, W']

                # GlobalAvgPool: trung bình trên chiều W' → T biến mất
                # mean(dim=-1): mean trên chiều cuối = W'
                p_k = p_k.mean(dim=-1)      # [B, D]

                p_scales.append(p_k)

            # p_scales: list K_scale=3 tensor [B, D]

            # ── Segment-level Attention ───────────────────────────────
            # Stack: [K_scale, B, D] → transpose → [B, K_scale, D]
            p_stack = torch.stack(p_scales, dim=1)   # [B, K_scale, D]

            # Flatten để đưa vào FC
            # reshape(B, -1): gộp K_scale và D thành một chiều
            # -1 để PyTorch tự tính: K_scale * D = 3 * 64 = 192
            p_flat = p_stack.reshape(B, -1)           # [B, K_scale*D]

            # FC → ReLU → Softmax → trọng số per scale
            alpha_seg = self.seg_fc(p_flat)            # [B, K_scale]
            alpha_seg = F.relu(alpha_seg)
            alpha_seg = torch.softmax(alpha_seg, dim=-1)
            # alpha_seg[b, k] = mức độ quan trọng của scale k
            # Σₖ alpha_seg[b, k] = 1

            # Weighted sum các scale
            # alpha_seg [B, K_scale] → unsqueeze(-1) → [B, K_scale, 1]
            # broadcast với p_stack [B, K_scale, D]
            # → [B, K_scale, D]  → sum(dim=1) → [B, D]
            alpha_seg = alpha_seg.unsqueeze(-1)        # [B, K_scale, 1]
            zi = (alpha_seg * p_stack).sum(dim=1)      # [B, D]
            # zi: đặc trưng view i đã tổng hợp các scale

            z_list.append(zi)

        # z_list: list V tensor [B, D]
        # Stack: [B, V, D]
        z_views = torch.stack(z_list, dim=1)   # [B, V, D]

        # ── View-level Attention ──────────────────────────────────────
        # Học xem view nào quan trọng hơn

        # FC trên từng view: [B, V, D] → [B, V, 1]
        # nn.Linear áp lên chiều cuối (-1 = D)
        # → mỗi view cho ra 1 scalar
        alpha_view = self.view_fc(z_views)      # [B, V, 1]
        alpha_view = F.relu(alpha_view)
        alpha_view = torch.softmax(alpha_view, dim=1)
        # softmax trên dim=1 (chiều V)
        # → Σᵢ alpha_view[b, i, 0] = 1
        # alpha_view[b, i, 0] = mức độ quan trọng của view i

        # Weighted sum các views
        # alpha_view [B, V, 1] * z_views [B, V, D]
        # → [B, V, D] → sum(dim=1) → [B, D]
        z_global = (alpha_view * z_views).sum(dim=1)   # [B, D]

        return z_views, z_global


# =============================================================================
# Test
# =============================================================================

def test_fusion():
    print("=== Test Feature Fusion ===\n")

    B, V, D, W = 2, 6, 64, 7

    # Dữ liệu giả từ inter-view attention
    F_inter = torch.randn(B, V, D, W)
    print(f"F_inter shape: {F_inter.shape}")   # [2, 6, 64, 7]

    model = FeatureFusion(D=D, kernel_sizes=(3, 5, 7))

    # In số tham số
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Số tham số FeatureFusion: {n_params:,}")

    with torch.no_grad():
        z_views, z_global = model(F_inter)

    print(f"z_views shape:  {z_views.shape}")    # [2, 6, 64]
    print(f"z_global shape: {z_global.shape}")   # [2, 64]

    print(f"z_views có NaN:  {torch.isnan(z_views).any()}")
    print(f"z_global có NaN: {torch.isnan(z_global).any()}")


if __name__ == '__main__':
    test_fusion()