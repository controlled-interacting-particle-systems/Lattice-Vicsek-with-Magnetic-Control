from setuptools import setup, find_packages

setup(
    name='particle_lattice',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm',
    ],
    author='EL KHIYATI Zakarya',
    author_email='elkhiyati.zakarya@gmail.com',
    description='A package for simulating Vicsek-like particles on a 2D lattice with magnetic fields',
)
