from setuptools import setup, find_packages

setup(
    name="image_captioning",
    version="0.0.1",
    author="Mohammed Bilal Shihabudeen",
    author_email="mohammedbilal2396@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)