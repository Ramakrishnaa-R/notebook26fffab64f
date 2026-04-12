from setuptools import setup

setup(
    name="audio-genre-classifier",
    version="1.0.0",
    python_requires=">=3.9",
    install_requires=[
        "streamlit==1.40.0",
        "torch==2.5.1",
        "librosa==0.10.2",
        "transformers==4.46.3",
        "soundfile==0.12.1",
        "huggingface-hub==0.26.5",
        "numpy==1.26.4",
    ],
)
