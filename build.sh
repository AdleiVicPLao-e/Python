#!/bin/bash
# Install specific Python version
pyenv install 3.9.18 -s
pyenv global 3.9.18
pip install -r requirements.txt