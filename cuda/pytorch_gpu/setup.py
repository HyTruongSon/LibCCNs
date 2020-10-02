from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='RisiContraction_18',
    ext_modules=[
        CUDAExtension('RisiContraction_18', [
            'RisiContraction_18.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })