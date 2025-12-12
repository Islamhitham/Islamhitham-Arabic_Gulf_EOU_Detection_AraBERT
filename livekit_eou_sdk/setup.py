from setuptools import setup, find_packages

setup(
    name="livekit-eou-sdk",
    version="1.0.0",
    description="Arabic End-of-Utterance (EOU) Detection SDK for LiveKit",
    author="islam hitham",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "livekit-agents"
    ],
)
