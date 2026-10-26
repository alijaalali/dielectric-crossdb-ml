# Writes a deduplicated copy of the raw MP dielectric data, keeping the first
# occurrence of each material_id. The original raw file is left untouched.
import json
from pathlib import Path

DATA = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project/data")

raw = json.load(open(DATA / "mp_dielectric_raw.json"))
seen = set()
dedup = []
for d in raw:
    mid = d["material_id"]
    if mid in seen:
        continue
    seen.add(mid)
    dedup.append(d)

print("raw:", len(raw), "dedup:", len(dedup))
with open(DATA / "mp_dielectric_raw_dedup.json", "w") as fh:
    json.dump(dedup, fh)
print("wrote", DATA / "mp_dielectric_raw_dedup.json")
