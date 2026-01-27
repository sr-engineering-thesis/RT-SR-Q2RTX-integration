#!/bin/sh
python setup.py build_ext --inplace
pip install --no-build-isolation .
