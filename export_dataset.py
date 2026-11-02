
import sys
import numpy as np, pandas as pd
from yard_sim import *

days = int(sys.argv[1]) if len(sys.argv) > 1 else 21
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
log = []
occ, m = YardEnv(seed, 0, days * SPD).run(Nearest(), log=log)
t0 = pd.Timestamp("2026-01-05 00:00")                      
ts = lambda s: t0 + pd.to_timedelta(s * SLOT_MIN, unit="m")

slots = np.arange(len(occ))
o = pd.DataFrame(occ.astype(int), columns=[f"bay_{i:02d}" for i in range(N)])
o.insert(0, "split", np.where(slots < int(.7 * days) * SPD, "train", np.where(slots < int(.8 * days) * SPD, "val", "test")))
o.insert(0, "timestamp", ts(slots)); o.insert(0, "slot", slots)
o.to_csv("yard_occupancy.csv", index=False)

tr = pd.DataFrame(log).sort_values("arrival_slot").reset_index(drop=True)
tr.insert(0, "truck_id", [f"T{i:05d}" for i in range(len(tr))])
tr["arrival_time"] = ts(tr.arrival_slot)
tr["gate_in_time"] = ts(tr.gate_in_slot.fillna(0)).where(tr.gate_in_slot.notna())
tr["departure_time"] = ts(tr.departure_slot.fillna(0)).where(tr.departure_slot.notna())
tr["wait_min"] = (tr.gate_in_slot - tr.arrival_slot) * SLOT_MIN
tr["truck_type"] = np.where(tr.length_m < 14, "short", "long")
tr["cargo_class"] = tr.cargo.map(dict(enumerate(CARGO_NAMES)))
tr["zone_name"] = tr.zone_id.map(lambda z: ZONE_NAMES[int(z)] if pd.notna(z) else None)
tr = tr[["truck_id", "arrival_time", "gate_in_time", "departure_time", "wait_min", "truck_type", "length_m",
         "cargo_class", "dwell_slots", "bay_id", "zone_id", "zone_name", "status"]]
tr.to_csv("yard_trucks.csv", index=False)

pd.DataFrame(dict(bay_id=range(N), zone_id=BAY_ZONE, zone_name=[ZONE_NAMES[z] for z in BAY_ZONE],
                  bay_length_m=BAY_LEN, dist_to_gate_m=BAY_DIST)).to_csv("yard_layout.csv", index=False)
print(len(o), "slots,", len(tr), "trucks;", {k: round(v, 2) for k, v in m.items()})
