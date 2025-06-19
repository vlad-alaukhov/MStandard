from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag_processor",
    version="0.1.0",
    author="Vlad Alaukhov",
    author_email="your.email@example.com",
    description="RAG Processing Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vlad-alaukhov/rag-processor",
    packages=find_packages(),
    py_modules=['rag_processor'],
    setup_requires=["wheel", "setuptools"],
    install_requires=[
        "torch==2.6.0+cpu",  # Версия для CPU (без CUDA)
        "pymupdf==1.25.5",
        "pdfminer.six==20221105",
        "camelot-py==0.10.1",
        "opencv-python-headless==4.11.0.86",
        "python-docx==1.1.2",
        "pandas==2.2.3",
        "langchain-core==0.3.51",
        "langchain-community==0.3.21",
        "langchain-openai==0.3.12",
        "langchain-huggingface==0.1.2",
        "tiktoken==0.9.0",
        "faiss-cpu==1.10.0",
        "python-dotenv==1.1.0",
        "requests==2.32.3",
        "sentence-transformers==4.0.2",
        "openpyxl==3.1.5",
    ],
    extras_require={
        "gpu": [
        "torch==2.6.0+cu118",  # Явная версия с CUDA
        "nvidia-cublas-cu12==12.4.5.8",
        "nvidia-cuda-runtime-cu12==12.4.127"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.8",
    keywords="rag ai nlp document-processing",
    project_urls={
        "Documentation": "https://github.com/vlad_alaukhov/rag-processor",
        "Source Code": "https://github.com/vlad_alaukhov/rag-processor"
    },
)