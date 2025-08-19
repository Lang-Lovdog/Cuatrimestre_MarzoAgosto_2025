#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <video_file>"
  exit 1
fi
file="$1"

directory="${file%.mp4}_frames"

mkdir -p "$directory"

ffmpeg -i "$1" "${directory}/frame%04d.ppm"
