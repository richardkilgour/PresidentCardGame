from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="asshole",
    version="0.0.1",
    author="Richard Kilgour",
    author_email="richardkilgour@gmail.com",
    description="Asshole",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://",
    packages=find_packages(),
    install_requires=[
        'termcolor',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts' : ['play_asshole=asshole.PlayAsshole:main'],
    }
)