with open("../datasets/samanantar_4950k_filtered.tsv", "r", encoding="utf-8") as fp:
    lines = fp.readlines()

print(len(lines) - 1) # 4891172
