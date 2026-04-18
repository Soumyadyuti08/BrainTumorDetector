import torch
import torch.nn as nn
import torch.nn.functional as F

NC = 4

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        mid = max(ch // 16, 8)
        self.mlp = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, ch, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        ca = torch.sigmoid(
            self.mlp(F.adaptive_avg_pool2d(x, 1).view(b, c)) +
            self.mlp(F.adaptive_max_pool2d(x, 1).view(b, c))
        ).view(b, c, 1, 1)
        x = x * ca
        sa = torch.sigmoid(self.spatial(
            torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)
        ))
        return x * sa


class ConvBN(nn.Sequential):
    def __init__(self, ic, oc, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(ic, oc, k, s, p, bias=False),
            nn.BatchNorm2d(oc),
            nn.GELU()
        )


class ResBlock(nn.Module):
    def __init__(self, ic, oc, stride=1, drop=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBN(ic, oc),
            nn.Conv2d(oc, oc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(oc)
        )
        self.cbam = CBAM(oc)
        self.drop = nn.Dropout2d(drop)
        self.pool = nn.MaxPool2d(2) if stride == 2 else nn.Identity()
        self.skip = (
            nn.Sequential(nn.Conv2d(ic, oc, 1, bias=False), nn.BatchNorm2d(oc))
            if ic != oc else nn.Identity()
        )

    def forward(self, x):
        return self.pool(F.gelu(self.drop(self.cbam(self.conv(x))) + self.skip(x)))


class BrainTumorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBN(3, 48, 7, 2, 3), nn.MaxPool2d(3, 2, 1))
        self.s1   = nn.Sequential(ResBlock(48,  64,  2, 0.05), ResBlock(64,  64,  1, 0.05))
        self.s2   = nn.Sequential(ResBlock(64,  128, 2, 0.08), ResBlock(128, 128, 1, 0.08))
        self.s3   = nn.Sequential(ResBlock(128, 256, 2, 0.10), ResBlock(256, 256, 1, 0.10), ResBlock(256, 256, 1, 0.10))
        self.s4   = nn.Sequential(ResBlock(256, 512, 2, 0.15), ResBlock(512, 512, 1, 0.15))

        self.gap3 = nn.AdaptiveAvgPool2d(1)
        self.gmp3 = nn.AdaptiveMaxPool2d(1)
        self.gap4 = nn.AdaptiveAvgPool2d(1)
        self.gmp4 = nn.AdaptiveMaxPool2d(1)

        self.fbn  = nn.BatchNorm1d(1536)  # 256*2 + 512*2

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NC)
        )

    def forward(self, x):
        x  = self.stem(x)
        x  = self.s1(x)
        x  = self.s2(x)
        s3 = self.s3(x)
        s4 = self.s4(s3)
        f3 = torch.cat([self.gap3(s3).flatten(1), self.gmp3(s3).flatten(1)], 1)
        f4 = torch.cat([self.gap4(s4).flatten(1), self.gmp4(s4).flatten(1)], 1)
        return self.head(self.fbn(torch.cat([f3, f4], 1)))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BrainTumorNet().to(device)
    model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
    model.eval()
    print("✓ Weights loaded successfully")

    dummy = torch.randn(1, 3, 256, 256).to(device)

    torch.onnx.export(
        model,
        dummy,
        "brain_tumor_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )
    print("✓ Saved: brain_tumor_model.onnx")