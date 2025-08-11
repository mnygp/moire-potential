#!/bin/bash

find . -maxdepth 1 -type d ! -name '.' | while read dir; do
    if [[ -f "$dir/initial.json" ]]; then
        echo "Submitting in $dir"
        (cd "$dir" && mq submit "asr.moire.relax_matsim --d3 --fixcell --fmax 0.01" -R 8:1:8G:sm3090el8:3d)
        sleep 2
    else
        echo "Skipping $dir — no initial.json"
    fi
done
