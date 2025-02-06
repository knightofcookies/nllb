import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as fp:
        lines = fp.readlines()
    nl = []
    for line in lines:
        nl.append(line.replace("\t\t\t\t\t", "\x1f"))
    with open(sys.argv[1], "w", encoding="utf-8") as fp:
        fp.writelines(nl)

except Exception as e:
    print(e)
