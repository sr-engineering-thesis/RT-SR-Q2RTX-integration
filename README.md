This repository contains inference part of RT-SR integration with Q2RTX.

Before running the program, install neccessary dependencies:
```
pip install torch numpy torchsr
```
You will also need to build addtional package `cudaGLStream` from source contained in this repo.
First make sure that both libtorch, CUDA toolkit and glfw3.0 are installed on your system.
You can do this on Debian 13 by running:
```
sudo apt install libtorch libglfw3-dev nvidia-cuda-toolkit
```
Then run:
```
cd cudacanvas
build.sh
```
This should install cudaGLStream (updated [cudacanvas](https://github.com/OutofAi/cudacanvas)).

Next, build Q2RTX following standard Q2RTX building procedure.
Start the game, make sure it is running in `1280x720` resolution and enter a game level.
Start `app.py`, this should open an addtional window with SR image and start the gameplay.

