from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

LIBTORCH='libtorch'

setup(
	name='ccn1d_lib', 
	ext_modules = [CppExtension('ccn1d_lib', ['ccn1d_lib.cpp'])], 
	library_dirs = [
		LIBTORCH + '/include', 
		LIBTORCH + '/include/torch/csrc/api/include',
		LIBTORCH + '/include/TH',
		LIBTORCH + '/include/THC'],
	cmdclass = {'build_ext': BuildExtension})