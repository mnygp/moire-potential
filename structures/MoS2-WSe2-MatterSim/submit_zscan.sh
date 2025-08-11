#!/bin/bash

find . -maxdepth 1 -type d ! -name '.' | while read dir; do
    if [[ -f "$dir/initial.json" ]]; then
        echo "Submitting in $dir"
        (cd "$dir" && mq submit "asr.moire.zscan_matsim --start=1.5" -R 8:1:8G:sm3090el8:2h)
        sleep 2
    else
        echo "Skipping $dir — no initial.json"
    fi
done
