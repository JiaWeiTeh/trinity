#!/bin/bash
# F1 full-run equivalence on the edge cases: ORIGINAL 60k vs F1 coarse, via
# SEPARATE processes (run.py), batched by code version so each batch runs the 3
# configs in parallel on the idle cores (no file swap mid-run).
cd /home/user/trinity
set -u
FILE=trinity/bubble_structure/bubble_luminosity.py
PRE_P3=58cafde
restore() { git checkout HEAD -- "$FILE" 2>/dev/null; }
trap restore EXIT

printf 'mCloud    1e5\nsfe    0.3\nmodel_name    f1cmp_simple\n' > /tmp/f1cmp_simple.param

NAMES=(f1cmp_simple f1edge_lowdens f1edge_hidens)
PARAMS=(/tmp/f1cmp_simple.param \
        docs/dev/performance/f1edge_lowdens_himass_hisfe.param \
        docs/dev/performance/f1edge_hidens_himass_losfe.param)

run_batch() {  # $1 = label (orig|f1)
  local label=$1 i
  for i in "${!NAMES[@]}"; do
    timeout 3600 python run.py "${PARAMS[$i]}" > "/tmp/f1cmp_${label}_${NAMES[$i]}.log" 2>&1 &
  done
  wait
  for i in "${!NAMES[@]}"; do
    local f="outputs/${NAMES[$i]}/dictionary.jsonl"
    if [ -e "$f" ]; then cp "$f" "/tmp/f1cmp_${label}_${NAMES[$i]}_dict.jsonl"
    else echo "WARN: no dict for ${NAMES[$i]} ($label) -- check /tmp/f1cmp_${label}_${NAMES[$i]}.log"; fi
  done
}

echo "[$(date +%H:%M:%S)] === BATCH ORIGINAL 60k ($PRE_P3) ==="
git checkout "$PRE_P3" -- "$FILE"
grep -q "t_eval=np.linspace" "$FILE" && { echo "ABORT: still coarse after checkout"; exit 1; }
run_batch orig
echo "[$(date +%H:%M:%S)] original batch done"

echo "[$(date +%H:%M:%S)] === BATCH F1 coarse (HEAD) ==="
git checkout HEAD -- "$FILE"
grep -q "t_eval=np.linspace" "$FILE" || { echo "ABORT: not coarse after restore"; exit 1; }
run_batch f1
echo "[$(date +%H:%M:%S)] f1 batch done"

echo "[$(date +%H:%M:%S)] === COMPARE (original 60k vs F1 coarse) ==="
python - <<'PY'
import json
names=["f1cmp_simple","f1edge_lowdens","f1edge_hidens"]
keys=["t_now","R2","rShell","Eb","v2","current_phase"]
def final(p):
    last=None
    try:
        for ln in open(p):
            ln=ln.strip()
            if ln: last=ln
        d=json.loads(last); return {k:d.get(k) for k in keys}
    except Exception as e: return {"err":str(e)}
worst=0.0
for c in names:
    o=final(f"/tmp/f1cmp_orig_{c}_dict.jsonl"); f=final(f"/tmp/f1cmp_f1_{c}_dict.jsonl")
    print(f"\n--- {c} ---  orig_err={o.get('err','')} f1_err={f.get('err','')}")
    for k in keys:
        ov,fv=o.get(k),f.get(k)
        if isinstance(ov,(int,float)) and isinstance(fv,(int,float)) and ov:
            rd=abs(fv-ov)/abs(ov); worst=max(worst,rd)
            print(f"  {k:14} orig={ov:<18.8g} F1={fv:<18.8g} rel={rd:.2e}{'  <== DIVERGE' if rd>3e-3 else ''}")
        else:
            print(f"  {k:14} orig={ov} F1={fv}")
print(f"\nWORST rel-diff across edge configs: {worst:.2e}  (gate 3e-3)")
print("VERDICT:", "F1 SAFE on edge cases" if worst<=3e-3 else "F1 DIVERGES -> revert P3")
PY
echo "[$(date +%H:%M:%S)] === F1_EDGE_BATCH_DONE ==="
echo "file restored to coarse (expect 1): $(grep -c 't_eval=np.linspace' "$FILE")"
