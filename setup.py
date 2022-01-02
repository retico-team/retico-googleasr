#!/usr/bin/env python3

"""
Setup script.

Use this script to install the GoogleASR module of the retico simulation framework.
Usage:
    $ python3 setup.py install
The run the simulation:
    $ retico [-h]
"""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

config = {
    "description": "The GoogleASR incremental module for the retico framework",
    "author": "Thilo Michael",
    "url": "??",
    "download_url": "??",
    "author_email": "thilo.michael@tu-berlin.de",
    "version": "0.1",
    "install_requires": ["retico-core~=0.2.0", "google-cloud-speech~=2.2.1"],
    "packages": find_packages(),
    "name": "retico-googleasr",
}

setup(**config)
