import ctypes, numpy as np, cv2, torch
from fsrcnn.fsrcnn import FSRCNN_BIG_PRETRAIN
from time import sleep, monotonic_ns

sleep(15)
libtest = ctypes.CDLL("./test.so")
libtest.get_frame.restype = ctypes.POINTER(ctypes.c_uint8)
libtest.get_frame_width.restype = ctypes.c_int
libtest.get_frame_height.restype = ctypes.c_int
libtest.get_frame_pitch.restype = ctypes.c_int
libtest.get_gpu_to_cpu_time.restype = ctypes.c_int
libtest.init()
print("going")
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
model = FSRCNN_BIG_PRETRAIN(scale=2).to(device).half().eval()
model.load_state_dict(torch.load("./fsrcnn/fsrcnn_finetuned_0.0036803.pth", map_location="cpu"))
torch.backends.cudnn.benchmark = True


height, width, pitch = libtest.get_frame_height(), libtest.get_frame_width(), libtest.get_frame_pitch()

out_h, out_w = height * 2, width * 2
output_pinned = torch.empty((out_h, out_w, 3), dtype=torch.uint8).pin_memory()
out_np_view = output_pinned.numpy()

def to_tensor_optimized(frame_ptr, height, width, pitch):
    address = ctypes.cast(frame_ptr, ctypes.c_void_p).value
    
    buffer_size = height * pitch
    frame_buffer = (ctypes.c_ubyte * buffer_size).from_address(address)
    return (
        torch.frombuffer(frame_buffer, dtype=torch.uint8)
        .view(height, pitch // 4, 4)[:, :width, :3] # Remove padding/Alpha on CPU view
        .to(device, non_blocking=True)               # Send 1-byte pixels to GPU
        .permute(2, 0, 1)                            # HWC -> CHW (Instant on GPU)
        .unsqueeze(0)                                # Add batch dim
        .to(torch.half)                              # Convert to FP16 on GPU
        .div_(255.0)                                 # Normalize in-place
    )

buffer_size = 200
frame_times_no_upscaling = np.zeros(buffer_size, dtype=np.float32)
frame_times_upscaling = np.zeros(buffer_size, dtype=np.float32)

frame_index = 0  # circular index

for _ in range(buffer_size * 2):
    start = monotonic_ns()
    libtest.frame_wait()
    frame_ptr = libtest.get_frame()
    input_tensor = to_tensor_optimized(frame_ptr, height, width, pitch)
    libtest.frame_post()
    frame_times_no_upscaling[frame_index] = (monotonic_ns() - start) / 1e6
    frame_index = (frame_index + 1) % buffer_size

frame_index = 0  # circular index

for _ in range(buffer_size * 2):
    start = monotonic_ns()
    libtest.frame_wait()
    frame_ptr = libtest.get_frame()
    input_tensor = to_tensor_optimized(frame_ptr, height, width, pitch)
    libtest.frame_post()

    with torch.no_grad():
        output = model(input_tensor)

    processed = output.mul_(255).clamp_(0, 255).to(torch.uint8)
    torch.cuda.synchronize()

    # # processed_gpu = (
    # #     output.squeeze(0)
    # #     .permute(1, 2, 0)
    # #     .mul_(255)
    # #     .clamp_(0, 255)
    # #     .to(torch.uint8)
    # # )

    # Non-blocking copy from GPU to Pinned RAM
    # output_pinned.copy_(processed_gpu, non_blocking=True)
    # cv2.imshow("Zero-Copy-ish Display", out_np_view)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break
    frame_times_upscaling[frame_index] = (monotonic_ns() - start) / 1e6
    frame_index = (frame_index + 1) % buffer_size

print("No Upscaling:", frame_times_no_upscaling.mean(), frame_times_no_upscaling.std())
print("Upscaling:", frame_times_upscaling.mean(), frame_times_upscaling.std())

# cv2.destroyAllWindows()

