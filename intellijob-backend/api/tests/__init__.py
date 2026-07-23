"""
IntelliJob — api app tests.

Django's test runner discovers tests in two ways:
  * functions/classes in api/tests.py (kept as a thin re-export shim)
  * any tests/ package under an app, whose modules match test_*.py

Real test modules live in this package so they can grow without
cluttering api/tests.py.
"""

from __future__ import annotations
