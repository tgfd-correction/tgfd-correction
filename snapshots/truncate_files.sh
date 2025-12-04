#!/bin/bash
for file in *.nt; do
  total_lines=$(wc -l < "$file")
  lines_to_keep=$(( total_lines *4 / 10 ))
  head -n "$lines_to_keep" "$file" > "${file%.nt}_truncated.nt"
done
