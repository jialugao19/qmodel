import torch
import torch.nn as nn
import torch.nn.functional as F

from qmodel.components.amp_scaler import CudaGraphAMPScaler

# ---------------- Scaler ----------------

# ---------------- Dummy Model ----------------
class TinyMLP(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.l1 = nn.Linear(d, d)
        self.l2 = nn.Linear(d, d)
        self.l3 = nn.Linear(d, 10)
    def forward(self, x):
        x = F.gelu(self.l1(x))
        x = F.gelu(self.l2(x))
        return self.l3(x)

# ---------------- Minimal Graph Training Demo ----------------
device = "cuda"
torch.manual_seed(0)
torch.cuda.manual_seed(0)
model = TinyMLP().to(device)
# 关键：capturable=True；PyTorch 2.x 的 AdamW/Adam 支持该选项
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, capturable=True, foreach=True)
scaler = CudaGraphAMPScaler(init_scale=2.**14, growth=2.0, backoff=0.5, growth_interval=1024, device=device)

B, D = 1024, 512
static_x = torch.empty(B, D, device=device)
static_y = torch.empty(B,   device=device, dtype=torch.long)

loss_fn = nn.CrossEntropyLoss()

# 预热，完成参数/状态张量分配，避免 Graph 内新分配

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        x = torch.randn_like(static_x)
        y = torch.randint(0, 10, (B,), device=device)
        optimizer.zero_grad(set_to_none=True)
        out  = model(x)
        loss = loss_fn(out, y)
        (loss * 1.0).backward()
        optimizer.step()

torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)

# ---- 捕获：无 Python 分支，所有控制流以张量算子表达 ----
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
# if 1:
    out  = model(static_x)
    loss = loss_fn(out, static_y)
    loss_scaled = scaler.scale(loss)
    loss_scaled.backward()
    scaler.step(optimizer)
    scaler.update()

# ---- 重放回路 ----
def train_step_numpy():
    x = torch.randn(B, D, device=device)
    y = torch.randint(0, 10, (B,), device=device)
    static_x.copy_(x)
    static_y.copy_(y)
    g.replay()
    return loss.detach()

for it in range(2000):
    loss_val = train_step_numpy()
    if (it+1) % 200 == 0:
        # 仅打印以便观察；不影响图捕获/重放
        print(f"[{it+1}] loss={loss_val.item():.4f} scale={scaler.scaler.item():.1f}")
