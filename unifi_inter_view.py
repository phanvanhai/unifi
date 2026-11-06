"""
UniFi — Module 2: Inter-view Attention
=======================================
Liu et al., Proc. ACM IMWUT, vol. 7, no. 4, 2023.
DOI: 10.1145/3631429

Pipeline khối này:
    F_intra [B, V, D, W']
        ↓ permute
    F_intra [B, W', V, D]
        ↓ temporal_aware_avg
    F_tilde [B, W', V, D]   (dùng làm K, V)
        ↓ build_spatial_mask
    Mask [V, V]
        ↓ MaskedCrossViewAttention
    F_inter [B, W', V, D]
        ↓ permute
    F_inter [B, V, D, W']
"""

import math
import random

import torch
import torch.nn as nn


# =============================================================================
# Bước 1 — Temporal-aware Weighted Average
# =============================================================================

def temporal_aware_avg(F_intra, S=1):
    """
    Làm mượt F_intra theo trục thời gian W' để bù lệch đồng bộ
    nhỏ giữa các thiết bị.

    Công thức:
        F_tilde[t] = Σ_{j=−S}^{S} wⱼ · F[t+j]
        wⱼ = 1/(1+|j|)²  rồi normalize tổng = 1

    Ví dụ S=1, W'=7:
        j=-1: F_tilde[t] += w₋₁ · F[t-1]   (lùi 1 bước)
        j= 0: F_tilde[t] += w₀  · F[t]      (đúng vị trí)
        j= 1: F_tilde[t] += w₁  · F[t+1]   (tiến 1 bước)
        Biên (t=0 thiếu F[-1], t=W'-1 thiếu F[W']) → đóng góp 0

    Tham số:
        F_intra : Tensor [B, W', V, D]
        S       : bán kính cửa sổ (không báo cáo trong bài, thử S=1)

    Trả về:
        F_tilde : Tensor [B, W', V, D]
    """
    # TODO: cong thuc trong so tu suy, vi trong bai bao
    # wⱼ = 1/(1+S)² thì đều nhau tại mọi j nen co ve khong hop ly
    # Tính trọng số wⱼ = 1/(1+|j|)²
    js      = list(range(-S, S + 1))   # ví dụ S=1: [-1, 0, 1]
    weights = [1.0 / (1 + abs(j))**2 for j in js]
    w_sum   = sum(weights)
    weights = [w / w_sum for w in weights]
    # S=1: weights = [1/6, 4/6, 1/6] ≈ [0.167, 0.667, 0.167]

    F_tilde = torch.zeros_like(F_intra)  # [B, W', V, D]

    for j, w in zip(js, weights):

        if j == 0:
            # Đóng góp F[t] vào đúng vị trí t
            F_tilde += w * F_intra

        elif j < 0:
            # Đóng góp F[t-|j|] vào vị trí t
            # Nguồn: F_intra[:, :j,  :, :] = F[0 .. W'-|j|-1]
            #   dùng :j với j âm, Python hiểu là bỏ |j| phần tử cuối
            # Đích:  F_tilde[:, -j:, :, :] = F_tilde[|j| .. W'-1]
            #   bỏ |j| vị trí đầu vì chúng không có F[t-|j|]
            #
            # Ví dụ j=-1, W'=7:
            #   nguồn: F[0,1,2,3,4,5]  (bỏ F[6])
            #   đích:  tilde[1,2,3,4,5,6]  (bỏ tilde[0])
            #   → tilde[1]+=w*F[0], tilde[2]+=w*F[1], ...
            F_tilde[:, -j:, :, :] += w * F_intra[:, :j, :, :]

        else:  # j > 0
            # Đóng góp F[t+j] vào vị trí t
            # Nguồn: F_intra[:, j:,  :, :] = F[j .. W'-1]
            #   bỏ j phần tử đầu vì F[0..j-1] không có đích t=-j..-1
            # Đích:  F_tilde[:, :-j, :, :] = F_tilde[0 .. W'-j-1]
            #   bỏ j vị trí cuối vì chúng không có F[t+j]
            #
            # Ví dụ j=1, W'=7:
            #   nguồn: F[1,2,3,4,5,6]  (bỏ F[0])
            #   đích:  tilde[0,1,2,3,4,5]  (bỏ tilde[6])
            #   → tilde[0]+=w*F[1], tilde[1]+=w*F[2], ...
            F_tilde[:, :-j, :, :] += w * F_intra[:, j:, :, :]

    return F_tilde  # [B, W', V, D]


# =============================================================================
# Bước 2 — Spatial-aware View Mask
# =============================================================================

def build_spatial_mask(V, directions, m=4, r=2):
    """
    Xây dựng mask che các views không bổ sung nhau theo hướng Fresnel.

    Ý tưởng: hai thiết bị đặt cùng hướng nhìn thấy cùng một mặt
    của cử chỉ → thông tin trùng lặp. Hai thiết bị ở hướng khác
    nhau mới bổ sung cho nhau.

    Với mỗi view i:
        1. Tìm m views có góc chênh lệch lớn nhất (bổ sung nhất)
        2. Random chọn r views trong m candidates đó
        3. Mask: cho attend r views được chọn, che phần còn lại

    Tham số:
        V          : số views (= 6)
        directions : list V góc đặt thiết bị (radian)
        m          : số candidate views (không báo cáo, thử 4)
        r          : số target views được chọn (không báo cáo, thử 2)

    Trả về:
        Mask : Tensor [V, V] float
               0.0       = cho attend (views bổ sung nhau)
               -inf      = che (views trùng hướng)
    """
    # Khởi tạo toàn -inf
    # -inf trong attention score → exp(-inf) = 0 sau softmax
    # → view đó không đóng góp vào aggregate
    Mask = torch.full((V, V), float('-inf'))

    for i in range(V):
        # View i luôn attend với chính nó
        Mask[i, i] = 0.0

        # Tính góc chênh lệch với tất cả view khác
        diffs = []
        for j in range(V):
            if j == i:
                continue
            diff = abs(directions[i] - directions[j])
            # Wrap về [0, π]: góc tối đa giữa 2 hướng bất kỳ là π
            diff = min(diff, 2 * math.pi - diff)
            diffs.append((diff, j))

        # Sắp xếp giảm dần: views bổ sung nhất (góc lớn nhất) lên đầu
        diffs.sort(key=lambda x: x[0], reverse=True)

        # Chọn m candidates có góc chênh lớn nhất
        candidates = [j for _, j in diffs[:m]]

        # Random chọn r trong m candidates
        # min đề phòng m < r
        targets = random.sample(candidates, min(r, len(candidates)))

        # Mở mask cho các views được chọn
        for j in targets:
            Mask[i, j] = 0.0

    return Mask  # [V, V]


# =============================================================================
# Bước 3 — Masked Cross-view Attention
# =============================================================================

class MaskedCrossViewAttention(nn.Module):
    """
    Multi-head cross-view attention tính tại từng timestep t độc lập.

    Công thức:
        Q = F_intra · Wq    [dùng F gốc — sắc nét tại đúng t]
        K = F_tilde · Wk    [dùng F đã làm mượt]
        V = F_tilde · Wv

        Attn = softmax( Mask(Q·Kᵀ / √dₖ) )   [V×V per timestep]
        Out  = Attn · V
        F_inter = Concat[heads] · Wo

    Trick multi-head: thay vì h phép nhân nhỏ [D→dₖ],
    dùng 1 phép nhân lớn [D→D] rồi reshape tách heads.
        W_q.weight [D, D] = h ma trận [D, dₖ] xếp cạnh nhau
        Q [B*W', V, D] → reshape → [B*W', V, h, dₖ]
    """

    def __init__(self, D=64, num_heads=4):
        """
        Tham số:
            D         : chiều đặc trưng (= 64)
            num_heads : số attention heads
                        D phải chia hết cho num_heads
                        dₖ = D / num_heads = 64 / 4 = 16
        """
        super().__init__()

        assert D % num_heads == 0, \
            f"D={D} phải chia hết cho num_heads={num_heads}"

        self.D     = D
        self.h     = num_heads
        self.d_k   = D // num_heads   # chiều mỗi head
        self.scale = self.d_k ** -0.5
        # scale = 1/√dₖ: tránh dot product quá lớn
        # nếu dₖ lớn, Q·Kᵀ có giá trị lớn
        # → softmax saturate về 0/1 → gradient gần 0 → khó học

        # 4 ma trận chiếu — bias=False theo convention Transformer
        # nn.Linear(D, D) lưu weight [D, D]
        # khi gọi linear(x): output = x @ weight.T
        self.W_q = nn.Linear(D, D, bias=False)   # chiếu Q
        self.W_k = nn.Linear(D, D, bias=False)   # chiếu K
        self.W_v = nn.Linear(D, D, bias=False)   # chiếu V
        self.W_o = nn.Linear(D, D, bias=False)   # chiếu output

    def forward(self, F_intra, F_tilde, Mask=None):
        """
        Tham số:
            F_intra : Tensor [B, W', V, D]  → dùng làm Q
            F_tilde : Tensor [B, W', V, D]  → dùng làm K, V
            Mask    : Tensor [V, V] hoặc None

        Trả về:
            F_inter : Tensor [B, W', V, D]
        """
        B, W, V, D = F_intra.shape

        # ── Gộp B và W' thành một chiều ──────────────────────────────
        # Attention tính độc lập tại từng t
        # → treat B*W' như batch dimension lớn hơn
        # → xử lý song song tất cả (batch, timestep) cùng lúc
        # [B, W', V, D] → [B*W', V, D]
        Q_in = F_intra.reshape(B * W, V, D)
        K_in = F_tilde.reshape(B * W, V, D)
        V_in = F_tilde.reshape(B * W, V, D)

        # ── Chiếu Q, K, V ────────────────────────────────────────────
        # nn.Linear áp lên chiều cuối (-1 = D)
        # Qh = F_intra · Wqh (công thức bài báo)
        # Thực tế: output = input @ W_q.weight.T
        # [B*W', V, D] @ [D, D] → [B*W', V, D]
        Q  = self.W_q(Q_in)   # [B*W', V, D]
        K  = self.W_k(K_in)   # [B*W', V, D]
        V_ = self.W_v(V_in)   # [B*W', V, D]

        # ── Tách thành h heads ────────────────────────────────────────
        # D = h * d_k → tách chiều D thành (h, d_k)
        # [B*W', V, D] → [B*W', V, h, d_k]
        # phần [:,:,0,:] = head 1, phần [:,:,1,:] = head 2, ...
        Q  = Q.reshape(B*W, V, self.h, self.d_k)
        K  = K.reshape(B*W, V, self.h, self.d_k)
        V_ = V_.reshape(B*W, V, self.h, self.d_k)

        # Đưa h lên trước V để matmul tính song song h heads
        # [B*W', V, h, d_k] → [B*W', h, V, d_k]
        # transpose(1, 2): hoán đổi chiều 1 (V) và chiều 2 (h)
        Q  = Q.transpose(1, 2)    # [B*W', h, V, d_k]
        K  = K.transpose(1, 2)    # [B*W', h, V, d_k]
        V_ = V_.transpose(1, 2)   # [B*W', h, V, d_k]

        # ── Tính attention score ──────────────────────────────────────
        # scores[b, head, i, j] = tương quan view i với view j tại t
        # Q [B*W', h, V, d_k] × Kᵀ [B*W', h, d_k, V]
        # → [B*W', h, V, V]
        # K.transpose(-2, -1): hoán đổi 2 chiều cuối
        #   [B*W', h, V, d_k] → [B*W', h, d_k, V]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        # scale: chia √dₖ để tránh giá trị quá lớn
        scores = scores * self.scale   # [B*W', h, V, V]

        # ── Áp spatial mask ───────────────────────────────────────────
        if Mask is not None:
            # Mask [V, V] cần broadcast lên [B*W', h, V, V]
            # unsqueeze(0): thêm chiều → [1, V, V]
            # unsqueeze(0): thêm chiều → [1, 1, V, V]
            # PyTorch tự broadcast lên [B*W', h, V, V]
            scores = scores + Mask.unsqueeze(0).unsqueeze(0)
            # Vị trí Mask=-inf → scores=-inf
            # → sau softmax: exp(-inf)=0 → view đó bị loại

        # ── Normalize ─────────────────────────────────────────────────
        # softmax trên dim=-1 (chiều V cuối = các view hàng xóm j)
        # tại mỗi view i: Σⱼ attn[i,j] = 1
        attn = torch.softmax(scores, dim=-1)   # [B*W', h, V, V]

        # ── Aggregate ─────────────────────────────────────────────────
        # Hh = Attn · V
        # attn [B*W', h, V, V] × V_ [B*W', h, V, d_k]
        # → [B*W', h, V, d_k]
        out = torch.matmul(attn, V_)   # [B*W', h, V, d_k]

        # ── Concat h heads ────────────────────────────────────────────
        # [B*W', h, V, d_k] → [B*W', V, h, d_k] → [B*W', V, D]
        # transpose(1, 2): đưa V lên trước h
        out = out.transpose(1, 2)              # [B*W', V, h, d_k]
        # reshape: gộp h và d_k lại thành D = h*d_k
        out = out.reshape(B * W, V, self.D)   # [B*W', V, D]

        # ── Output projection ─────────────────────────────────────────
        # Wo: trộn thông tin từ h heads
        out = self.W_o(out)   # [B*W', V, D]

        # ── Khôi phục chiều B và W' ───────────────────────────────────
        F_inter = out.reshape(B, W, V, D)   # [B, W', V, D]

        return F_inter


# =============================================================================
# Ghép toàn bộ Inter-view pipeline
# =============================================================================

class InterViewAttention(nn.Module):
    """
    Pipeline đầy đủ:
        F_intra [B, V, D, W']
            ↓ permute → [B, W', V, D]
            ↓ temporal_aware_avg → F_tilde [B, W', V, D]
            ↓ MaskedCrossViewAttention
            ↓ permute → [B, V, D, W']
        F_inter [B, V, D, W']
    """

    def __init__(self, D=64, num_heads=4, S=1):
        """
        Tham số:
            D         : chiều đặc trưng (= 64)
            num_heads : số attention heads (không báo cáo, thử 4)
            S         : bán kính temporal avg (không báo cáo, thử 1)
        """
        super().__init__()
        self.S    = S
        self.attn = MaskedCrossViewAttention(D=D, num_heads=num_heads)

    def forward(self, F_intra, Mask=None):
        """
        Tham số:
            F_intra : Tensor [B, V, D, W']
            Mask    : Tensor [V, V] hoặc None

        Trả về:
            F_inter : Tensor [B, V, D, W']
        """
        # ── Permute về [B, W', V, D] ──────────────────────────────────
        # Đưa W' lên trước V để xử lý attention theo t
        # permute(0, 3, 1, 2):
        #   chiều mới 0 = chiều cũ 0 = B
        #   chiều mới 1 = chiều cũ 3 = W'
        #   chiều mới 2 = chiều cũ 1 = V
        #   chiều mới 3 = chiều cũ 2 = D
        F = F_intra.permute(0, 3, 1, 2)   # [B, W', V, D]

        # ── Temporal-aware avg → F_tilde ──────────────────────────────
        # F_tilde dùng làm K và V trong attention
        # F gốc dùng làm Q — giữ sắc nét tại đúng t
        F_tilde = temporal_aware_avg(F, S=self.S)   # [B, W', V, D]

        # ── Masked cross-view attention ───────────────────────────────
        F_inter = self.attn(F, F_tilde, Mask)   # [B, W', V, D]

        # ── Permute về [B, V, D, W'] ──────────────────────────────────
        # Nhất quán với format của intra-view output
        F_inter = F_inter.permute(0, 2, 3, 1)   # [B, V, D, W']

        return F_inter


# =============================================================================
# Test
# =============================================================================

def test_inter_view():
    print("=== Test Inter-view Attention ===\n")

    B, V, D, W = 2, 6, 64, 7

    # Dữ liệu giả từ intra-view
    F_intra = torch.randn(B, V, D, W)
    print(f"F_intra shape: {F_intra.shape}")   # [2, 6, 64, 7]

    # Tạo mask: 6 thiết bị đặt đều nhau quanh vòng tròn
    directions = [i * math.pi / 3 for i in range(V)]
    # [0, π/3, 2π/3, π, 4π/3, 5π/3]
    Mask = build_spatial_mask(V, directions, m=4, r=2)
    print(f"Mask shape: {Mask.shape}")         # [6, 6]
    print(f"Mask:\n{Mask}\n")
    # Mỗi hàng i: vị trí 0.0 = views được attend
    #             vị trí -inf = views bị che

    # Chạy inter-view attention
    # TODO: Khong ro num_heads, S
    model = InterViewAttention(D=D, num_heads=4, S=1)

    # torch.no_grad(): tắt tính gradient — chỉ dùng khi test
    # tiết kiệm bộ nhớ, không ảnh hưởng kết quả forward
    with torch.no_grad():
        F_inter = model(F_intra, Mask)

    print(f"F_inter shape: {F_inter.shape}")
    # Kỳ vọng: [2, 6, 64, 7] — giống F_intra

    # Kiểm tra giá trị hợp lệ
    print(f"F_inter có NaN: {torch.isnan(F_inter).any()}")  # False
    print(f"F_inter min: {F_inter.min():.4f}")
    print(f"F_inter max: {F_inter.max():.4f}")


if __name__ == '__main__':
    test_inter_view()