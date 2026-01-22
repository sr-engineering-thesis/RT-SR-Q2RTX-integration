import ctypes, numpy as np, cv2, torch
from fsrcnn.fsrcnn import FSRCNN_BIG_PRETRAIN
from time import sleep

sleep(15)
libtest = ctypes.CDLL("./test.so")
libtest.get_frame.restype = ctypes.POINTER(ctypes.c_uint8)
libtest.get_frame_width.restype = ctypes.c_int
libtest.get_frame_height.restype = ctypes.c_int
libtest.get_frame_pitch.restype = ctypes.c_int
libtest.init()
print("going")
device = torch.device("cuda")
model = FSRCNN_BIG_PRETRAIN(scale=2).to(device).half().eval()
model.load_state_dict(torch.load("./fsrcnn/fsrcnn_finetuned_0.0036803.pth", map_location="cpu"))
torch.backends.cudnn.benchmark = True

height, width, pitch = libtest.get_frame_height(), libtest.get_frame_width(), libtest.get_frame_pitch()

def to_tensor_cuda_half(img):
    return (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device=device, dtype=torch.half, non_blocking=True)
        .div_(255.0)
    )

with torch.inference_mode():
    while True:
        libtest.frame_wait()
        frame_ptr = libtest.get_frame()
        frame_np = np.ctypeslib.as_array(frame_ptr, shape=(height, pitch // 4, 4))[:, :width, :3]
        input_tensor = to_tensor_cuda_half(frame_np)

        output = model(input_tensor)
        out_np = (
            output[0]
            .mul(255.0)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .byte()
            .cpu()
            .numpy()
        )

        cv2.imshow("Window Name", out_np)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        libtest.frame_post()

cv2.destroyAllWindows()

