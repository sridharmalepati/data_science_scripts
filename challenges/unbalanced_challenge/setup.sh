#!/bin/bash

pip install -r dependencies.txt
bash setup.sh
ipython notebook --pylab=inline
