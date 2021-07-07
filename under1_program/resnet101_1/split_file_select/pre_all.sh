#!/bin/bash

echo "shell script: split data start."
python split_data.py
echo "shell script: split data end."

echo "shell script: file_selection start."
python file_selection.py
echo "shell script: file_selection end."
