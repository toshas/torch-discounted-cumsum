#!/usr/bin/env python
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

with open(os.path.join('torch_discounted_cumsum', 'version.py')) as f:
    version_pycode = f.read()
exec(version_pycode)

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


long_description = """
This package implements an efficient parallel algorithm for the computation of discounted cumulative sums 
with differentiable bindings to PyTorch. The `cumsum` operation is frequently seen in data science 
domains concerned with time series, including Reinforcement Learning (RL). 

The traditional sequential algorithm performs the computation of the output elements in a loop. For an input of size 
`N`, it requires `O(N)` operations and takes `O(N)` time steps to complete. 

The proposed parallel algorithm requires a total of `O(N log N)` operations, but takes only `O(log N)` time, which is a 
considerable trade-off in many applications involving large inputs.  

Features of the parallel algorithm:
- Speed logarithmic in the input size
- Better numerical precision than sequential algorithms

Features of the package:
- CPU: sequential algorithm in C++
- GPU: parallel algorithm in CUDA
- Gradients computation for input and gamma
- Batch support for input and gamma
- Both left and right directions of summation supported
- PyTorch bindings

Find more details and the most up-to-date information on the project webpage:
https://www.github.com/toshas/torch-discounted-cumsum
"""


def configure_extensions():
    out = [
        CppExtension(
            'torch_discounted_cumsum_cpu',
            [
                os.path.join('torch_discounted_cumsum', 'discounted_cumsum_cpu.cpp'),
            ],
        )
    ]
    try:
        out.append(
            CUDAExtension(
                'torch_discounted_cumsum_cuda',
                [
                    os.path.join('torch_discounted_cumsum', 'discounted_cumsum_cuda.cpp'),
                    os.path.join('torch_discounted_cumsum', 'discounted_cumsum_cuda_kernel.cu'),
                ],
            )
        )
    except Exception as e:
        print(f'Failed to build CUDA extension, this part of the package will not work. Reason: {str(e)}')
    return out


setup(
    name='torch_discounted_cumsum',
    version=__version__,
    description='Fast discounted cumulative sums in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    python_requires='>=3.6',
    packages=find_packages(),
    author='Anton Obukhov',
    license='BSD',
    url='https://www.github.com/toshas/torch-discounted-cumsum',
    ext_modules=configure_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    },
    keywords=[
        'pytorch', 'discounted', 'cumsum', 'cumulative', 'sum', 'scan', 'differentiable',
        'reinforcement', 'learning', 'rewards', 'time', 'series'
    ],
)
