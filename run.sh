#!/bin/bash

for sigma in 10 25 25 40 50
do
    python main.py --sigma ${sigma}
done
