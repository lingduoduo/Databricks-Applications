from setuptools import setup, find_packages

setup(
    name='src',
    version="0.0.1",
    description=("Core ML MLflow jobs"),
    packages=find_packages(),
    package_data={'': ['*.json', '*.yaml', '*.jar', '*.dat',  '*.html']}
)