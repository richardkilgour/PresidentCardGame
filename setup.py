import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="asshole",
    version="0.0.1",
    author="Richard Kilgour",
    author_email="richardkilgour@gmail.com",
    description="Asshole",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)