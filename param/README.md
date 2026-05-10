# `param/` — user-facing parameter files

This directory holds **user-configurable** `.param` files. Each file specifies
a single TRINITY run (or sweep) by overriding only the keys it cares about;
all other keys fall back to the schema defaults.

## Where things live

- **Schema + defaults**: [`src/_input/default.param`](../src/_input/default.param)
  is the canonical list of every valid parameter name and its default value.
  **Don't edit it unless you mean to change defaults for every run.**
- **Human-facing parameter docs** (what each parameter means, units, allowed
  values, physics): [`docs/source/parameters.rst`](../docs/source/parameters.rst).
- **Examples**: this folder. See e.g. `simple_cluster.param`, `rosette.param`,
  or any `*_sweep.param` for sweep syntax.

## How a run is configured

`read_param.py` loads the schema from `src/_input/default.param`, then loads
your `.param` from this directory, and merges. Any key in your file that is
**not** in the schema is rejected with an error. Keys you omit fall back to
the schema default.

To create a new run, copy an existing example and edit the keys you need.
