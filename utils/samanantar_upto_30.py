with open("../datasets/samanantar_corpus.tsv", "r", encoding="utf-8") as fp:
    lines = fp.readlines()

lines_to_write = []
for line in lines:
    en_line, _ = line.split("\t\t\t\t\t")
    en_line_count = len(en_line.split(" "))
    if en_line_count <= 30:
        lines_to_write.append(line)

with open("../datasets/samanantar_upto_30.tsv", "w", encoding="utf-8") as fp:
    fp.writelines(lines_to_write)
