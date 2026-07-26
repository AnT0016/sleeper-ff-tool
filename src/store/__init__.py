"""Storage layer for the point-in-time data lake (Phase 8).

``store.lake`` owns *where* and *how* captured data is persisted; collectors
(``collect.*``) own *what* is captured. Nothing in here knows about scoring.
"""
