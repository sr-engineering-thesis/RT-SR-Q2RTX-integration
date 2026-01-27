from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cudaGLStream',
    ext_modules=[
        CUDAExtension(
            name='cudaGLStream',
            sources=['cudacanvas.cpp'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--expt-relaxed-constexpr']
            },
            libraries=['glfw', 'GL'],  # On Windows use: ['glfw3', 'opengl32']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

