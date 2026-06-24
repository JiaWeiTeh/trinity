# Phase 2.3 four-arm statistics

### arms_mock4e3

| arm | segs | conv f | conv g | short% | med ev | max ev | aborts (kind) | out-of-box |
|---|---|---|---|---|---|---|---|---|
| A control | 27 | 0/27 (0%) | 0/27 (0%) | 0% | 24 | 24 | 0 | - |
| B metric | 27 | 0/27 (0%) | 0/27 (0%) | 0% | 25 | 26 | 0 | - |
| C cap+bounds | 27 | 0/27 (0%) | 0/27 (0%) | 0% | 121 | 121 | 0 | 0 |
| D hybr | 27 | 21/27 (78%) | 21/27 (78%) | 0% | 29 | 33 | 6 (structure:6) | 19 |

### arms_simple1e5

| arm | segs | conv f | conv g | short% | med ev | max ev | aborts (kind) | out-of-box |
|---|---|---|---|---|---|---|---|---|
| A control | 30 | 14/30 (47%) | 15/30 (50%) | 23% | 24 | 24 | 0 | - |
| B metric | 30 | 7/30 (23%) | 15/30 (50%) | 40% | 25 | 25 | 0 | - |
| C cap+bounds | 30 | 7/30 (23%) | 18/30 (60%) | 40% | 37 | 121 | 0 | 0 |
| D hybr | 30 | 18/30 (60%) | 24/30 (80%) | 40% | 10 | 33 | 6 (neg_dMdt:4, structure:1, timeout:1) | 5 |
