from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='RPGN_OP',
    ext_modules=[
        CUDAExtension('RPGN_OP', [
            'src/rpgn_ops_api.cpp',

            'src/rpgn_ops.cpp',
            'src/cuda.cu'
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)