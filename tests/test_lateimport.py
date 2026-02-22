# SPDX-FileCopyrightText: 2026 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tests for lateimport."""

from __future__ import annotations

import threading

from types import MappingProxyType
from unittest.mock import patch

import pytest

from lateimport import LazyImport, create_lazy_getattr, lazy_import


# ---------------------------------------------------------------------------
# LazyImport / lazy_import
# ---------------------------------------------------------------------------


class TestLazyImportCreation:
    def test_not_resolved_on_creation(self):
        proxy = lazy_import("os")
        assert not proxy.is_resolved()

    def test_repr_pending(self):
        proxy = lazy_import("os")
        assert "pending" in repr(proxy)
        assert "os" in repr(proxy)

    def test_repr_resolved(self):
        proxy = lazy_import("os")
        _ = proxy._resolve()
        assert "resolved" in repr(proxy)

    def test_lazy_import_returns_lazy_import_instance(self):
        proxy = lazy_import("os")
        assert isinstance(proxy, LazyImport)


class TestLazyImportResolution:
    def test_resolves_module(self):
        import os

        proxy = lazy_import("os")
        assert proxy._resolve() is os

    def test_resolves_attribute(self):
        import os.path

        proxy = lazy_import("os.path", "join")
        assert proxy._resolve() is os.path.join

    def test_call_resolves_and_calls(self):
        proxy = lazy_import("os.path", "join")
        result = proxy("a", "b")
        assert result == "a/b"

    def test_is_resolved_after_resolution(self):
        proxy = lazy_import("os")
        proxy._resolve()
        assert proxy.is_resolved()

    def test_resolution_cached(self):
        proxy = lazy_import("os")
        first = proxy._resolve()
        second = proxy._resolve()
        assert first is second


class TestLazyImportAttributeChaining:
    def test_chained_attribute_creates_child_proxy(self):
        proxy = lazy_import("os")
        child = proxy.path
        assert isinstance(child, LazyImport)
        assert not child.is_resolved()

    def test_chained_resolution(self):
        import os.path

        proxy = lazy_import("os")
        assert proxy.path.join._resolve() is os.path.join

    def test_introspection_attr_resolves_immediately(self):
        proxy = lazy_import("os")
        # __name__ is in INTROSPECTION_ATTRIBUTES — should resolve the module
        assert proxy.__name__ == "os"
        assert proxy.is_resolved()


class TestLazyImportErrors:
    def test_bad_module_raises_import_error(self):
        proxy = lazy_import("no_such_module_xyzzy")
        with pytest.raises(ImportError, match="no_such_module_xyzzy"):
            proxy._resolve()

    def test_bad_attribute_raises_attribute_error(self):
        proxy = lazy_import("os", "no_such_attr_xyzzy")
        with pytest.raises(AttributeError, match="no_such_attr_xyzzy"):
            proxy._resolve()

    def test_bad_introspection_attr_raises_attribute_error(self):
        # Python resolves dunder attrs on the type, not the instance, so we
        # call __getattr__ directly to exercise the introspection branch.
        proxy = lazy_import("os", "no_such_attr_xyzzy")
        with pytest.raises(AttributeError):
            proxy.__getattr__("__doc__")


class TestLazyImportThreadSafety:
    def test_concurrent_resolution_returns_same_object(self):
        proxy = lazy_import("os")
        results = []

        def resolve():
            results.append(proxy._resolve())

        threads = [threading.Thread(target=resolve) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(r is results[0] for r in results)


# ---------------------------------------------------------------------------
# create_lazy_getattr
# ---------------------------------------------------------------------------


class TestCreateLazyGetattr:
    def _make_getattr(self, dispatch: dict[str, tuple[str, str]]) -> object:
        g: dict[str, object] = {}
        return create_lazy_getattr(MappingProxyType(dispatch), g, "test_module"), g

    def test_returns_callable(self):
        fn, _ = self._make_getattr({})
        assert callable(fn)

    def test_getattr_module_is_set(self):
        fn, _ = self._make_getattr({})
        assert fn.__module__ == "test_module"

    def test_unknown_attr_raises_attribute_error(self):
        fn, _ = self._make_getattr({})
        with pytest.raises(AttributeError, match="no attribute 'missing'"):
            fn("missing")

    def test_resolves_attribute_from_module(self):
        # dispatch format: attr_name -> (package, submodule)
        # imports `package.submodule` then does getattr(submodule, attr_name)
        # so {"join": ("os", "path")} → getattr(os.path, "join") → os.path.join
        import os.path

        dispatch = {"join": ("os", "path")}
        module_globals: dict[str, object] = {}
        fn = create_lazy_getattr(MappingProxyType(dispatch), module_globals, "test_module")

        result = fn("join")
        assert result is os.path.join

    def test_caches_result_in_globals(self):
        dispatch = {"join": ("os", "path")}
        module_globals: dict[str, object] = {}
        fn = create_lazy_getattr(MappingProxyType(dispatch), module_globals, "test_module")

        fn("join")
        assert "join" in module_globals

    def test_module_mode(self):
        """target_module == '__module__' imports the submodule itself."""
        dispatch = {"path": ("os", "__module__")}
        module_globals: dict[str, object] = {}
        fn = create_lazy_getattr(MappingProxyType(dispatch), module_globals, "test_module")

        import os.path

        result = fn("path")
        assert result is os.path

    def test_error_message_includes_module_name(self):
        fn, _ = self._make_getattr({})
        with pytest.raises(AttributeError, match="test_module"):
            fn("nope")
