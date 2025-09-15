from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="adaptive-k-retrieval",
    version="0.0.1",
    description="Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Chihiro Taguchi",
    author_email="ctaguchi@nd.edu",
    url="https://github.com/megagonlabs/adaptive-k-retrieval",
    packages=find_packages(),
    license="BSD",
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires=">=3.10",
)