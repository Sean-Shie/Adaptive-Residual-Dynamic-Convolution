import torch
import torch.nn as nn

# =========================
# Baseline CNN
# =========================
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# =========================
# NEW MODELS
# =========================
def pair(x):
    return x if isinstance(x, tuple) else (x, x)

class ARDConv(nn.Module):
    """
    Adaptive Residual Dynamic Convolution

    Goal:
    Stable CIFAR-10 performance toward 90%+
    while preserving the original 3-layer backbone.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        K: int = 4,
        tau: float = 1.5,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.groups = groups
        self.K = K
        self.tau = tau

        kH, kW = self.kernel_size
        hidden_dim = max(in_channels // 4, 16)

        # -------------------------------------------------
        # Base kernel + residual dynamic kernels
        # -------------------------------------------------
        self.base_weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kH, kW)
        )

        self.delta_weight = nn.Parameter(
            torch.empty(K, out_channels, in_channels // groups, kH, kW)
        )

        self.kernel_scale = nn.Parameter(
            torch.ones(K, out_channels, 1, 1, 1)
        )

        # -------------------------------------------------
        # Global context branch
        # -------------------------------------------------
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # -------------------------------------------------
        # Local routing branch
        # -------------------------------------------------
        self.router_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )

        # global descriptor injected into routing
        self.router_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.SiLU(inplace=True),
        )

        self.router_gate = nn.Conv2d(
            hidden_dim,
            K * out_channels,
            kernel_size=1,
            bias=True,
        )

        # -------------------------------------------------
        # Collaborative attention
        # -------------------------------------------------
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // 8, 8), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(in_channels // 8, 8), in_channels, 1),
            nn.Sigmoid(),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # -------------------------------------------------
        # Output normalization + recalibration
        # -------------------------------------------------
        gn_groups = 8 if out_channels % 8 == 0 else 1
        self.out_norm = nn.GroupNorm(gn_groups, out_channels)

        self.out_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels // 8, 8), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(out_channels // 8, 8), out_channels, 1),
            nn.Sigmoid(),
        )

        # stronger internal residual scaling
        if in_channels == out_channels and self.stride == (1, 1):
            self.identity_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.identity_scale = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        for k in range(self.K):
            nn.init.kaiming_uniform_(self.delta_weight[k], a=math.sqrt(5))

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor):
        identity = x
        B, C, H, W = x.shape

        # -------------------------------------------------
        # A. Input enhancement
        # -------------------------------------------------
        g_context = self.global_context(x)
        x = x * g_context

        c_attn = self.channel_gate(x)
        s_attn = self.spatial_gate(
            torch.cat(
                [
                    torch.mean(x, dim=1, keepdim=True),
                    torch.max(x, dim=1, keepdim=True)[0],
                ],
                dim=1,
            )
        )
        x = x * c_attn * s_attn

        # -------------------------------------------------
        # B. Multi-scale routing
        # -------------------------------------------------
        local_feat = self.router_conv(x)
        global_feat = self.router_global(x)
        local_feat = local_feat + global_feat

        alpha = self.router_gate(local_feat)
        alpha = alpha.view(B, self.K, self.out_channels, H, W)

        # temperature-controlled routing
        alpha = torch.softmax(alpha / self.tau, dim=1)

        # -------------------------------------------------
        # C. unfold
        # -------------------------------------------------
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
        )

        H_out = (
            H
            + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1

        W_out = (
            W
            + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1

        L = H_out * W_out
        x_unfold = x_unfold.transpose(1, 2)

        alpha_map = F.interpolate(
            alpha.view(B, self.K * self.out_channels, H, W),
            size=(H_out, W_out),
            mode="bilinear",
            align_corners=False,
        ).view(B, self.K, self.out_channels, L)

        # -------------------------------------------------
        # D. Dynamic residual kernel
        # W_eff = W_base + sum(alpha_k * ΔW_k)
        # -------------------------------------------------
        base = self.base_weight.view(self.out_channels, -1)
        delta = (self.delta_weight * self.kernel_scale).view(
            self.K,
            self.out_channels,
            -1,
        )

        final_out = torch.zeros(
            (B, self.out_channels, L),
            device=x.device,
            dtype=x.dtype,
        )

        # static path
        static_out = torch.matmul(x_unfold, base.t())
        final_out += static_out.transpose(1, 2)

        # dynamic residual path
        for k in range(self.K):
            res_k = torch.matmul(x_unfold, delta[k].t())
            final_out += (
                res_k * alpha_map[:, k].transpose(1, 2)
            ).transpose(1, 2)

        out = final_out.view(B, self.out_channels, H_out, W_out)

        # -------------------------------------------------
        # E. normalization + output SE
        # -------------------------------------------------
        out = self.out_norm(out)
        out = out * self.out_se(out)

        # stronger internal residual
        if self.identity_scale is not None:
            out = out + torch.tanh(self.identity_scale) * identity

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out

class ARD_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ARDConv(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            ARDConv(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            ARDConv(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

