# Trinity headless-plotting env for an interactive cluster node.
#
# SOURCE this (don't execute it) before running any scratch/paper_*.py script:
#
#     source tools/cluster/plot_env.sh
#     python scratch/paper_radiusEvolution.py -F outputs/my_sweep   # renders to PDF, no display
#
# Make it a real alias by adding to ~/.bashrc on the cluster:
#     alias trinity-plot='source /path/to/Trinity/tools/cluster/plot_env.sh'
#
# It points matplotlib at tools/cluster/matplotlibrc, which gives the Computer
# Modern LaTeX *look* with NO external LaTeX toolchain (Agg backend, mathtext).
# Under Agg, plt.show() is a harmless no-op, so the scripts need no edits.

# Resolve the repo root from this file's location (works sourced in bash or zsh).
_self="${BASH_SOURCE[0]:-$0}"
_repo="$(cd "$(dirname "$_self")/../.." && pwd)"

export MATPLOTLIBRC="$_repo/tools/cluster/matplotlibrc"
export MPLBACKEND=Agg
# Font cache needs a writable dir (cluster /home is often read-only at runtime).
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH:-${TMPDIR:-/tmp}}/mplcache-$USER}"
mkdir -p "$MPLCONFIGDIR"

echo "Trinity headless plotting env set: Agg + Computer Modern mathtext, no LaTeX."
echo "  MATPLOTLIBRC=$MATPLOTLIBRC"
echo "  MPLCONFIGDIR=$MPLCONFIGDIR"
unset _self _repo
