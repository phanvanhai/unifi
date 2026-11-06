import torch.nn as nn

from unifi_intra_view    import IntraViewWrapper
from unifi_inter_view    import InterViewAttention
from unifi_fusion        import FeatureFusion


class UniFi(nn.Module):
    def __init__(self, V=6, D=64, num_classes=6, Mask=None):
        """
        V           : số views (receivers)
        D           : chiều đặc trưng
        num_classes : số lớp gesture
        Mask        : Tensor [V, V] (từ build_spatial_mask) hoặc None
        """
        super().__init__()
        self.V          = V
        self.D          = D   # train() dùng model.D cho ProxyContrastiveLoss
        self.intra      = IntraViewWrapper(D)
        self.inter      = InterViewAttention(D)
        self.fusion     = FeatureFusion(D)
        self.classifier = nn.Linear(D, num_classes)

        # Buffer: tự chuyển device theo model.to(device), không phải parameter
        self.register_buffer('Mask', Mask)

    def forward(self, X_t, X_f, Mask=None):
        """
        X_t : [B, V, 3, K, T]
        X_f : [B, V, 3, F, L]
        """
        if Mask is None:
            Mask = self.Mask

        F_intra           = self.intra(X_t, X_f)        # [B, V, D, W']
        F_inter           = self.inter(F_intra, Mask)   # [B, V, D, W']
        z_views, z_global = self.fusion(F_inter)        # [B, V, D], [B, D]
        logit_global      = self.classifier(z_global)   # [B, C]
        # nn.Linear áp lên chiều cuối → không cần reshape
        logit_views       = self.classifier(z_views)    # [B, V, C]
        return logit_global, logit_views, z_global
