# setup.py
from setuptools import setup, find_packages

setup(
    name="turboquant-serve",
    version="0.1.1",
    description="TurboQuant KV-cache compression for any HuggingFace model",
    packages=find_packages(),
    py_modules=["serve", "compare"],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.51.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.0.0",
        "huggingface_hub",
    ],
    extras_require={
        "bench": ["datasets", "nltk"],       # for needle-in-haystack benchmark
        "dev":   ["pytest", "black", "ruff", "twine", "build"],
    },
    entry_points={
        "console_scripts": [
            "tq-serve=serve:main",
            "tq-compare=compare:main",
            "tq-bench=bench.longcontext:main",
        ],
    },
)
