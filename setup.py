from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='freewire',
      version='0.0.0a',
      description='Freely wired feed-forward neural networks in PyTorch',
      install_requires=requirements,
      license='MIT',
      packages=find_packages())
