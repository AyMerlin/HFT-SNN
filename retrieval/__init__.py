"""Retrieval layer: downloads and converts datasets into the files the framework reads.

Not part of the framework. `snn_hft` never imports it and performs no network access
itself; the retrieval layer imports from `snn_hft` only the modules that define the shared
data format (enforced by tests/lob/test_architecture.py).
"""
