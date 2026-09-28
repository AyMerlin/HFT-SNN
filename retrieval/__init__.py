"""Retrieval layer (plan §3.1, §4): download, order-book reconstruction, quality control and
writing of standardized LOB files.

Not part of the framework. The only framework module imported here is
`snn_hft.data.lob.schema` (enforced by tests/lob/test_architecture.py).
"""
