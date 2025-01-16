# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prodigyopt",
    version="1.1.2",
    author="Konstantin Mishchenko",
    author_email="konsta.mish@gmail.com",
    description="An Adam-like optimizer for neural networks with adaptive estimation of learning rate",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/konstmish/prodigy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
