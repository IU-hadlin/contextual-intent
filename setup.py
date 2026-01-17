from setuptools import setup, find_packages

setup(
    name="came_bench",
    version="0.1.0",
    description="CAME-Bench: Benchmark for Contextual Memory and Intent",
    author="Ruozhen Yang",
    author_email="ruozhen2@illinois.edu",
    packages=find_packages(),
    install_requires=[
        "grpcio>=1.71.2",
        "grpcio-tools>=1.74.0",
        "litellm>=1.74.0",
        "python-dotenv>=1.0.1",
        "qdrant-client>=1.9.0",
        "protobuf>=6.31.0",
        "tiktoken==0.11.0",
        "dspy-ai",
        "numpy<2",
        "tqdm>=4.66.0",
        "openai>=1.90.0,<1.110.0",
        "pydantic>=2.7.3",
        "huggingface_hub",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
)
