# from setuptools import setup
from distutils.core import setup

setup(
    name='dynamic_stock_model',
    version="1.0",
    packages=["dynamic_stock_model", "dynamic_stock_model.tests"],
    author="Stefan Pauliuk",
    description="Python class for efficient handling of dynamic stock models",
    author_email="stefan.pauliuk@ntnu.no",
    license=open('LICENSE.txt').read(),
    install_requires=["numpy", "scipy"],
    long_description=open('README.md').read(),
    url = "https://github.com/stefanpauliuk/dynamic_stock_model",
    download_url = "https://github.com/stefanpauliuk/dynamic_stock_model/tarball/1.0",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
)
