#!/bin/bash

args=(7 8 28 29 48 50 58 62 67 68 80 81 99)

for sol in "${args[@]}"; do
	asr run "asr.moire.makemoire --solution=$sol --make-subdirectory=True"
done
