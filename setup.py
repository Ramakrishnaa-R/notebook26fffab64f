from setuptools import setup

setup(
    name="audio-genre-classifier",
    version="1.0.0",
    python_requires=">=3.14",
    install_requires=[
        "streamlit==1.40.0",
        "torch==2.11.0",
        "librosa==0.10.2",
        "transformers==4.40.0",
        "soundfile==0.12.1",
        "huggingface-hub==0.21.4",
        "numpy==2.0.0",
    ],
)
