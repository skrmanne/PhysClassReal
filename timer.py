import os, sys
import torch
import numpy as np
from neural_methods.model.VIRENet import VIRENet

model = VIRENet(frame_depth=10, img_size=96)
device = torch.device("cuda")
model.to(device)
# (N*D, C, H, W); N=1; D=10; C=3; HxW=96x96;
dummy_input = torch.randn(40, 3, 96, 96, dtype=torch.float).to(device)

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(dummy_input)

# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
            
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)
