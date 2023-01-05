#!/bin/bash

# untar the python installation
tar -xzf python310.tar.gz

# untar the python packages
tar -xzf packages.tar.gz

# making sure that the script will run using my python installation
export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD

# running the script 
python3 img_2_txt.py 'test.jpg'