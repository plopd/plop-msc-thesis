from pathlib import Path

from analysis.results import Result

datapath = Path("~/scratch/Chain").expanduser()
result = Result("Chain19Random.json", datapath, "Chain19Random.json")
with open("Chain19Random.txt") as f:
    content = f.readlines()
content = [int(x.strip().split("_")[0]) for x in content]
missing_values = []
for i in range(9600):
    if i not in content:
        print(f"{i}", file=open("missing_values", "a"), end="\n")
