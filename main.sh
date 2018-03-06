#!/bin/bash

for i in $(seq 1 1); do
    python make_data.py ${i};
    python main.py logsumexp ${i};
done