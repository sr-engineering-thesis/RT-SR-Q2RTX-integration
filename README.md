# RT-SR Inference Integration with Q2RTX

This repository contains the inference component of a real-time super-resolution (RT-SR) integration with **Q2RTX**. The project enables GPU-accelerated SR inference on frames rendered by Q2RTX, with results displayed in a separate output window.

## Requirements

### Python Dependencies

Install the required Python packages:

```bash
pip install torch numpy torchsr
```

Ensure that PyTorch is installed with CUDA support compatible with your GPU and driver.

### System Dependencies

You must build an additional native package, **`cudaGLStream`**, from source included in this repository. Before building, ensure the following system dependencies are installed:

* **libtorch**
* **CUDA Toolkit**
* **GLFW 3.0**

On **Debian 13**, these can be installed with:

```bash
sudo apt install libtorch libglfw3-dev nvidia-cuda-toolkit
```

## Building `cudaGLStream`

The `cudaGLStream` package is an updated version of [`cudacanvas`](https://github.com/OutofAi/cudacanvas) adapted for this project.

To build and install it:

```bash
cd cudacanvas
./build.sh
```

Upon successful completion, `cudaGLStream` will be built and installed.

## Building and Running Q2RTX

Next, build **Q2RTX** following the standard Q2RTX build procedure as described in its official documentation.

After building:

1. Launch Q2RTX.
2. Set the game resolution to **1280×720**.
3. Load any in-game level and ensure gameplay is running.

## Running the RT-SR Application

With Q2RTX running, start the inference application:

```bash
python app.py
```

This will:

* Open an additional window displaying the super-resolved output.
* Begin real-time SR inference synchronized with the running Q2RTX gameplay.
