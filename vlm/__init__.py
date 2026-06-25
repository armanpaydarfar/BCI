"""
vlm/ — leaf modules extracted from vlm_service.py.

Our GPU-host VLM service (vlm_service.py) grew into a god file; this package
holds the pure, self-contained module-level helpers it used to define inline,
cut out verbatim so the service stays behaviour-identical (it re-imports the
names, so every call site resolves unchanged). These modules are a clean DAG —
they import only stdlib / numpy / the in-tree perception package, never
vlm_service — so they can be unit-tested in isolation.

Not to be confused with `perception/` (Vivian's vendored, attribution-headered
boundary) — this is OUR code. The package is named `vlm` (not `vlm_service`) to
avoid colliding with the vlm_service.py entry-point module.
"""
