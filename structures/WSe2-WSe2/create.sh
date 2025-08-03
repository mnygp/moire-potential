#!/bin/bash

args=(10 16 21 22 23 26 31 35 36 42 51 57 64 67 70)

for sol in "${args[@]}"; do
	asr run "asr.moire.makemoire --solution=$sol --make-subdirectory=True"
done
