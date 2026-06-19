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
# MATCHED-t comparison (NOT final-state): under the 1h cap the original-60k and
# F1-coarse runs truncate at DIFFERENT t (F1 is faster -> reaches further), so a
# raw final-state diff false-flags divergence. Interpolate F1 onto the original's
# t grid over the common [0, min(tmax)] range and compare R2/Eb/rShell there.
python - <<'PY'
import json
import numpy as np
names=["f1cmp_simple","f1edge_lowdens","f1edge_hidens"]
keys=["R2","Eb","rShell","v2"]
def traj(p):
    t=[]; d={k:[] for k in keys}
    try:
        for ln in open(p):
            ln=ln.strip()
            if not ln: continue
            j=json.loads(ln); t.append(j.get("t_now"))
            for k in keys: d[k].append(j.get(k))
        return np.array(t,float), {k:np.array(d[k],float) for k in keys}
    except Exception as e:
        return None, str(e)
worst=0.0
for c in names:
    ot,ov=traj(f"/tmp/f1cmp_orig_{c}_dict.jsonl")
    ft,fv=traj(f"/tmp/f1cmp_f1_{c}_dict.jsonl")
    if ot is None or ft is None:
        print(f"\n--- {c} --- ERROR orig={ov} f1={fv}"); continue
    order=np.argsort(ft); ft=ft[order]
    tmax=min(ot.max(), ft.max()); mask=ot<=tmax; tg=ot[mask]
    print(f"\n--- {c} --- matched-t over [0,{tmax:.4f}] ({mask.sum()} pts; orig_tmax={ot.max():.3f} f1_tmax={ft.max():.3f})")
    for k in keys:
        o=ov[k][mask]; f=np.interp(tg, ft, fv[k][order])
        denom=np.maximum(np.abs(o), np.abs(o).max()*1e-6+1e-300)  # robust vs v2 zero-crossing
        rel=np.abs(f-o)/denom
        if k in ("R2","Eb","rShell"): worst=max(worst, rel.max())
        flag="  <== DIVERGE" if (k in ("R2","Eb","rShell") and rel.max()>3e-3) else ""
        print(f"  {k:8} worst_rel={rel.max():.2e} @t={tg[rel.argmax()]:.4f}  median={np.median(rel):.2e}{flag}")
print(f"\nWORST matched-t rel-diff (R2/Eb/rShell): {worst:.2e}  (gate 3e-3)")
print("VERDICT:", "F1 EQUIVALENT on edge cases" if worst<=3e-3 else "F1 DIVERGES -> revert P3")
PY
echo "[$(date +%H:%M:%S)] === F1_EDGE_BATCH_DONE ==="
echo "file restored to coarse (expect 1): $(grep -c 't_eval=np.linspace' "$FILE")"
