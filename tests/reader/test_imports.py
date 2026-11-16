"""Every module of the reader imports, including the session jobs no other test runs."""
import importlib
import pkgutil

import check2.reader


def test_every_reader_module_imports():
    names = [m.name for m in pkgutil.iter_modules(check2.reader.__path__)]
    assert "build" in names and "select" in names
    for name in names:
        importlib.import_module(f"check2.reader.{name}")
    importlib.import_module("check2.gates.reader")
