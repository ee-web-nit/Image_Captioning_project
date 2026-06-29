from setuptools import setup, find_packages

setup(
    name="image_captioning",
    version="1.0.0",
    author="Mohammed Bilal Shihabudeen",
    author_email="mohammedbilal2396@gmail.com",
    description="End-to-End Transformer-based Image Captioning System",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.10",
)