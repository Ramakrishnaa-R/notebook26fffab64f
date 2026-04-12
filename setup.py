from setuptools import setup

setup(
    name="audio-genre-classifier",
    version="1.0.0",
    python_requires=">=3.11,<3.12",
    install_requires=[
        "streamlit==1.32.0",
        "torch==2.0.1",
        "librosa==0.10.0",
        "transformers==4.34.0",
        "soundfile==0.12.1",
        "huggingface-hub==0.17.3",
        "numpy>=1.23.0,<2.0.0",
    ],
)
