from setuptools import setup, find_packages

setup(
    name="my_package",
    version="0.1",
    packages=find_packages(),
    description="A package hosted on GitHub",
    long_description=open("README.md").read(),
    author="Your Name",
    url="https://github.com/your-v4vraj/my_package",
    install_requires=[],  # Add dependencies here (e.g., "numpy")
)