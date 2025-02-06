import sys

with open("4723k.tsv", "r", encoding="utf-8") as fp:
    lines = fp.readlines()

print(len(lines) - 1) # 4891172
