import torch
import fvcore.nn
import fvcore.common
from fvcore.nn import FlopCountAnalysis
from models.pvcnn_cls import get_model

model = get_model(k=40, extra_channel=3).cuda()
model.eval()
# model = deit_tiny_patch16_224()

inputs = torch.randn((1,6,1024)).cuda()
k = 1024.0
flops = FlopCountAnalysis(model, inputs).total()
print(f"Flops : {flops}")
flops = flops/(k**3)
print(f"Flops : {flops:.1f}G")
params = fvcore.nn.parameter_count(model)[""]
print(f"Params : {params}")
params = params/(k**2)
print(f"Params : {params:.1f}M")
