#!/bin/bash

# Make sure you are in the folder containing your images
# and that all images are in the current directory or adjust the path

mkdir -p ../batches
i=1
batch=1
mkdir -p ../batches/batch_$batch

for img in *.{jpg,jpeg,png,bmp,tiff,tif}; do
  [ -e "$img" ] || continue  # skip if no matching files
  cp "$img" ../batches/batch_$batch/
  i=$((i+1))
  if [ $i -gt 50 ]; then
    i=1
    batch=$((batch+1))
    mkdir -p ../batches/batch_$batch
  fi
done

