"""Limit-order-book data: standardized format (schema), read-only sources and containers.

Kept import-free on purpose: the retrieval layer imports `snn_hft.data.lob.schema`, and
importing it must not pull in the rest of the framework (plan §3.1).
"""
