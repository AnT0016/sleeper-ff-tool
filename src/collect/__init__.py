"""Data collectors for the point-in-time lake (Phase 8).

Each provider module returns raw, provider-native rows plus their provenance; persisting is
``store.lake``'s job. ``collect.registry`` is the authoritative table of *what* exists to collect.
"""
