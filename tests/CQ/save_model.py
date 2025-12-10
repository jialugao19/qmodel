from train_config_1m import Config, ModelConfig1, EvalConfig1
from qmodel.models import build_model
from qmodel.components.checkpoint import CheckpointSaver

import numpy as np
import torch

model = build_model(Config.model_class, Config.model)
# root_dir = "train_logs/1122/CQ2_1min_convgru1_clip_1122_233437"
root_dir = "train_logs/1122/CQ2_1min_convgru1_clip_1122_221344"
saver = CheckpointSaver("", Config(), "cpu")

iteration = 40000
path = f"{root_dir}/ckpt/iter_{iteration}.pt"

saver.load(path, model, None, None, None)

# print the dtype of each val in model
for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")

# convert to float32
model = model.to(dtype=torch.float32)


# save to onnx

dummy_input = torch.randn(1, Config.window_size, ModelConfig1.input_dim, dtype=torch.float32)
print(dummy_input.shape)
print(model(dummy_input).shape)
model_name = f"{root_dir}/cq2_1min_convgru1_{iteration}.onnx"
# save allowing dynamic batch size
torch.onnx.export(model, (dummy_input,), model_name,
    # input_names=["x"],
    # output_names=["y"],
    # dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}
)
print(f"ONNX model saved to {model_name}")

import onnx

model2 = onnx.load(model_name)
onnx.checker.check_model(model2)
print("ONNX model check passed.")
