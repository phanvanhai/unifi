import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNet18Backbone(nn.Module):
    """
    ResNet18 tùy chỉnh:
    - Bỏ lớp FC cuối (không cần phân loại)
    - Thêm AdaptiveAvgPool để gộp chiều H về 1
    - Output: [B, 512, W'] với W' = W_input / 32
    """

    def __init__(self, pretrained=False):
        super().__init__()

        # Load ResNet18 — pretrained=False vì ảnh CSI
        # khác hoàn toàn ảnh ImageNet
        # Dùng API weights= (torchvision >= 0.13, bản trên Kaggle);
        # pretrained=True cần bật Internet trong notebook Kaggle để tải trọng số
        base = resnet18(weights='IMAGENET1K_V1' if pretrained else None)

        # Giữ tất cả layer trừ avgpool và fc cuối
        # nn.Sequential đóng gói nhiều lớp thành một khối
        self.features = nn.Sequential(
            base.conv1,    # [B, 64,  H/2,  W/2 ]
            base.bn1,      # Batch Norm
            base.relu,
            base.maxpool,  # [B, 64,  H/4,  W/4 ]
            base.layer1,   # [B, 64,  H/4,  W/4 ]
            base.layer2,   # [B, 128, H/8,  W/8 ]
            base.layer3,   # [B, 256, H/16, W/16]
            base.layer4,   # [B, 512, H/32, W/32]
        )

        # Pool chiều H về 1, giữ nguyên chiều W
        # AdaptiveAvgPool2d((out_H, out_W))
        # None = giữ nguyên chiều đó
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        # Output: [B, 512, 1, W/32]

    def forward(self, x):
        """
        x   : [B, 3, H, W]
        out : [B, 512, W']   W' = W/32
        """
        x = self.features(x)  # [B, 512, H/32, W/32]
        x = self.h_pool(x)    # [B, 512, 1,    W/32]
        x = x.squeeze(2)      # [B, 512, W/32]
        # squeeze(2): bỏ chiều index 2 vì kích thước = 1
        return x


class IntraViewExtractor(nn.Module):
    """
    Trích xuất đặc trưng trong từng view.

    Kiến trúc:
        ResNet18_t(X_t) → F_t [B, 512, W_t]
        ResNet18_f(X_f) → F_f [B, 512, W_f]
        align W_f về W_t
        concat → [B, 1024, W_t]
        Channel Attention → [B, 1024, W_t]
        Conv1D → [B, D, W_t]
    """

    def __init__(self, D=64):
        """
        Tham số:
            D          : chiều đặc trưng output (= 64)

        Resize ảnh được làm ở WiDarDataset (img_size), không làm ở đây.
        """
        super().__init__()

        # Hai backbone riêng, KHÔNG share với nhau
        # Nhưng mỗi backbone dùng chung cho tất cả V views
        self.resnet_t = ResNet18Backbone(pretrained=False)
        self.resnet_f = ResNet18Backbone(pretrained=False)
        # Output mỗi backbone: [B, 512, W']

        # Số kênh sau concat: 512 + 512 = 1024
        C_concat = 1024

        # Channel Attention
        # Nhận vector trung bình [B, 1024]
        # Trả về trọng số [B, 1024]
        self.channel_attn = nn.Sequential(
            nn.Linear(C_concat, C_concat),
            nn.Softmax(dim=-1)   # dim=-1 = chiều cuối = kênh
        )

        # Conv1D hợp nhất 1024 kênh xuống D=64
        # kernel_size=1: pointwise conv — không thay đổi W
        self.conv1d = nn.Conv1d(
            in_channels=C_concat,
            out_channels=D,
            kernel_size=1,
            bias=False
        )

        self.D = D

    def forward(self, X_t, X_f):
        """
        X_t : [B, 3, H, W]   ảnh pha  (đã resize ở WiDarDataset, mặc định 224×224)
        X_f : [B, 3, H, W]   ảnh DFS  (đã resize ở WiDarDataset, mặc định 224×224)
        out : [B, D, W']      D=64, W' = W/32
        """

        # --- ResNet18 backbone ---
        F_t = self.resnet_t(X_t)  # [B, 512, 7]  (224/32 = 7)
        F_f = self.resnet_f(X_f)  # [B, 512, 7]

        # --- Align chiều W nếu khác nhau ---
        # AdaptiveAvgPool1d(size): pool 1D về đúng size
        W_t = F_t.shape[-1]  # lấy W của F_t làm chuẩn
        if F_f.shape[-1] != W_t:
            F_f = F.adaptive_avg_pool1d(F_f, W_t)
        # F_t, F_f: [B, 512, W_t]

        # --- Concat theo chiều kênh ---
        # torch.cat([a, b], dim=1): nối theo chiều kênh
        F_tf = torch.cat([F_t, F_f], dim=1)  # [B, 1024, W_t]

        # --- Channel Attention ---
        # Tính trung bình trên chiều W để ra vector kênh
        # TODO: tu suy luan dung mean
        attn_input = F_tf.mean(dim=-1)        # [B, 1024]
        alpha = self.channel_attn(attn_input)  # [B, 1024]

        # Nhân trọng số vào từng vị trí thời gian
        # alpha [B, 1024] → unsqueeze(-1) → [B, 1024, 1]
        # broadcast với F_tf [B, 1024, W_t]
        alpha = alpha.unsqueeze(-1)            # [B, 1024, 1]
        F_weighted = F_tf * alpha              # [B, 1024, W_t]

        # --- Conv1D → output ---
        F_intra = self.conv1d(F_weighted)      # [B, D, W_t]

        return F_intra  # [B, 64, W']


class IntraViewWrapper(nn.Module):
    """
    Chạy IntraViewExtractor trên tất cả V views.
    ResNet18 shared — cùng một model chạy V lần.
    """

    def __init__(self, D=64):
        super().__init__()
        self.extractor = IntraViewExtractor(D=D)

    def forward(self, X_t_views, X_f_views):
        """
        X_t_views : [B, V, 3, H, W]
        X_f_views : [B, V, 3, H, W]
        out       : [B, V, D, W']
        """
        B, V = X_t_views.shape[:2]

        # Gộp B và V thành một batch lớn [B*V, 3, H, W] rồi chạy extractor 1 lần.
        # Tương đương chạy V lần (extractor dùng chung cho mọi view) nhưng:
        #   - GPU chạy 1 kernel lớn thay vì V kernel nhỏ → nhanh hơn
        #   - BatchNorm tính thống kê trên cả B*V ảnh, running stats cập nhật
        #     1 lần/batch thay vì V lần với thống kê của từng view riêng lẻ
        X_t = X_t_views.flatten(0, 1)            # [B*V, 3, H, W]
        X_f = X_f_views.flatten(0, 1)            # [B*V, 3, H, W]

        F_i = self.extractor(X_t, X_f)           # [B*V, D, W']
        out = F_i.unflatten(0, (B, V))           # [B, V, D, W']

        return out


def test_intra_view():
    print("=== Test Intra-view Extraction ===")

    B, V = 2, 6
    H, W = 224, 224   # ảnh đã resize ở WiDarDataset

    # Tạo tensor giả — torch.randn sinh số ngẫu nhiên
    X_t = torch.randn(B, V, 3, H, W)   # [2, 6, 3, 224, 224]
    X_f = torch.randn(B, V, 3, H, W)   # [2, 6, 3, 224, 224]

    model = IntraViewWrapper(D=64)

    # Tắt gradient khi chỉ test (tiết kiệm bộ nhớ)
    with torch.no_grad():
        F_intra = model(X_t, X_f)

    print(f"F_intra shape: {F_intra.shape}")
    # Kỳ vọng: [2, 6, 64, 7]
    # W' = 224/32 = 7

if __name__ == '__main__':
    test_intra_view()