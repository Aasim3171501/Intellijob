"""
IntelliJob — service layer for the api app.

Houses pure-Python helpers (no Django ORM, no spaCy, no pgvector) that
the views, management commands, and tests can import without spinning up
the framework. Each module should be importable in isolation.
"""

from __future__ import annotations
