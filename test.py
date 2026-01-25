import ctypes, numpy as np, cv2, torch
from fsrcnn.fsrcnn import FSRCNN_BIG_PRETRAIN
from time import sleep, monotonic_ns
from torchsr.models import carn, carn_m, edsr_baseline, edsr_r16f64, edsr_r32f256, ninasr_b0, rcan
from fsrcnn.fsrcnn import FSRCNN_SMALL_PRETRAIN, FSRCNN_BIG_PRETRAIN
from functools import partial
import torch.nn as nn
from ninasr_small import NinaSR 
import sys
import cudaGLStream

def fsrcnn_small(scale, pretrained):
    model = FSRCNN_SMALL_PRETRAIN(scale=scale)
    return model

def ninasr_small(scale, pretrained):
    return NinaSR(8, 16, scale)

MODELS = [
    ("fsrcnn", fsrcnn_small),
    ("carn", carn),
    ("ninasr_b0", ninasr_b0),
    ("small", ninasr_small),
    ("edsr_r16f64", edsr_r16f64),
]

sleep(5)
libtest = ctypes.CDLL("./test.so")
libtest.get_frame.restype = ctypes.POINTER(ctypes.c_uint8)
libtest.get_frame_width.restype = ctypes.c_int
libtest.get_frame_height.restype = ctypes.c_int
libtest.get_frame_pitch.restype = ctypes.c_int
libtest.get_gpu_to_cpu_time.restype = ctypes.c_int
libtest.init()
streamer = cudaGLStream.CudaGLStreamer()
print(streamer)
print("Starting testing", sys.stderr)

target_width = 2560
target_height = 1440
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
height, width, pitch = libtest.get_frame_height(), libtest.get_frame_width(), libtest.get_frame_pitch()
print(height, width, pitch, file=sys.stderr)
print("model,no upscaling,no uspcaling std,upscaling,upscaling std")
def to_tensor_cuda_half(img):
    return (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device=device, dtype=torch.half, non_blocking=True)
        .div_(255.0)
    )

for name, constructor in MODELS:
    model = torch.compile(constructor(scale=4, pretrained = True).to(device).eval())


    buffer_size = 200
    frame_times_no_upscaling = np.zeros(buffer_size, dtype=np.float32)
    frame_times_upscaling = np.zeros(buffer_size, dtype=np.float32)
    frame_index = 0  # circular index
    for _ in range(buffer_size * 2):
        start = monotonic_ns()
        libtest.frame_wait()
        frame_ptr = libtest.get_frame()
        frame_np = np.ctypeslib.as_array(frame_ptr, shape=(height, pitch // 4, 4))[:, :width, :3]
        input_tensor = to_tensor_cuda_half(frame_np)
        libtest.frame_post()

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(input_tensor)
        img_chw = output.squeeze(0)
        streamer.im_show(img_chw)

        frame_times_upscaling[frame_index] = (monotonic_ns() - start) / 1e6
        frame_index = (frame_index + 1) % buffer_size

    print(name, 
          frame_times_no_upscaling.mean(), frame_times_no_upscaling.std(),
          frame_times_upscaling.mean(), frame_times_upscaling.std(), sep=",")

