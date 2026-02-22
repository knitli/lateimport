# SPDX-FileCopyrightText: 2026 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tests for lateimport."""

from __future__ import annotations

import threading

from types import MappingProxyType

import pytest

from lateimport import LazyImport, create_lazy_getattr, lazy_import


# ---------------------------------------------------------------------------
# LazyImport / lazy_import
# ---------------------------------------------------------------------------


class TestLazyImportCreation:
    """Test creation and basic properties of LazyImport objects returned by lazy_import()."""

    def test_not_resolved_on_creation(self) -> None:
        """Test that a new LazyImport is not resolved."""
        proxy = lazy_import("os")
        assert not proxy.is_resolved()

    def test_repr_pending(self) -> None:
        """Test that the repr of a pending LazyImport indicates it's pending."""
        proxy = lazy_import("os")
        assert "pending" in repr(proxy)
        assert "os" in repr(proxy)

    def test_repr_resolved(self) -> None:
        """Test that the repr of a resolved LazyImport indicates it's resolved."""
        proxy = lazy_import("os")
        _ = proxy._resolve()
        assert "resolved" in repr(proxy)

    def test_lazy_import_returns_lazy_import_instance(self) -> None:
        """Test that lazy_import() returns an instance of LazyImport."""
        proxy = lazy_import("os")
        assert isinstance(proxy, LazyImport)


class TestLazyImportResolution:
    """Test the resolution behavior of LazyImport objects, including caching and error handling."""

    def test_resolves_module(self) -> None:
        """Test that a LazyImport can resolve a module."""
        import os

        proxy = lazy_import("os")
        assert proxy._resolve() is os

    def test_resolves_attribute(self) -> None:
        """Test that a LazyImport can resolve an attribute of a module."""
        import os.path

        proxy = lazy_import("os.path", "join")
        assert proxy._resolve() is os.path.join

    def test_call_resolves_and_calls(self) -> None:
        """Test that calling a LazyImport resolves it and calls the resulting object."""
        proxy = lazy_import("os.path", "join")
        result = proxy("a", "b")
        assert result == "a/b"

    def test_is_resolved_after_resolution(self) -> None:
        """Test that is_resolved() returns True after the LazyImport has been resolved."""
        proxy = lazy_import("os")
        proxy._resolve()
        assert proxy.is_resolved()

    def test_resolution_cached(self) -> None:
        """Test that resolving a LazyImport multiple times returns the same object (cached)."""
        proxy = lazy_import("os")
        first = proxy._resolve()
        second = proxy._resolve()
        assert first is second


class TestLazyImportAttributeChaining:
    """Test that accessing attributes on a LazyImport returns new LazyImport proxies that can be resolved correctly."""

    def test_chained_attribute_creates_child_proxy(self) -> None:
        """Test that accessing an attribute on a LazyImport returns a new LazyImport proxy for that attribute."""
        proxy = lazy_import("os")
        child = proxy.path
        assert isinstance(child, LazyImport)
        assert not child.is_resolved()

    def test_chained_resolution(self) -> None:
        """Test that accessing an attribute on a LazyImport and then resolving it works correctly."""
        import os.path

        proxy = lazy_import("os")
        assert proxy.path.join._resolve() is os.path.join

    def test_introspection_attr_resolves_immediately(self) -> None:
        """Test that accessing an introspection attribute (like __name__) on a LazyImport resolves the module immediately."""
        proxy = lazy_import("os")
        # __name__ is in INTROSPECTION_ATTRIBUTES — should resolve the module
        assert proxy.__name__ == "os"
        assert proxy.is_resolved()


class TestLazyImportErrors:
    """Test error handling behavior of LazyImport, including import errors and attribute errors."""

    def test_bad_module_raises_import_error(self) -> None:
        """Test that a LazyImport raises ImportError for a non-existent module."""
        proxy = lazy_import("no_such_module_xyzzy")
        with pytest.raises(ImportError, match="no_such_module_xyzzy"):
            proxy._resolve()

    def test_bad_attribute_raises_attribute_error(self) -> None:
        """Test that a LazyImport raises AttributeError for a non-existent attribute."""
        proxy = lazy_import("os", "no_such_attr_xyzzy")
        with pytest.raises(AttributeError, match="no_such_attr_xyzzy"):
            proxy._resolve()

    def test_bad_introspection_attr_raises_attribute_error(self) -> None:
        """Test that a LazyImport raises AttributeError for a non-existent introspection attribute."""
        # Python resolves dunder attrs on the type, not the instance, so we
        # call __getattr__ directly to exercise the introspection branch.
        proxy = lazy_import("os", "no_such_attr_xyzzy")
        with pytest.raises(AttributeError):
            proxy.__getattr__("__doc__")


class TestLazyImportThreadSafety:
    """Test that LazyImport resolution is thread-safe and that concurrent resolutions return the same object."""

    def test_concurrent_resolution_returns_same_object(self) -> None:
        """Test that resolving the same LazyImport concurrently from multiple threads returns the same resolved object."""
        proxy = lazy_import("os")
        results = []

        def resolve() -> None:
            """Resolve the proxy and append the result to the results list."""
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
    """Test the behavior of the create_lazy_getattr function, which creates a __getattr__ function that lazily imports attributes based on a dispatch mapping."""

    def _make_getattr(self, dispatch: dict[str, tuple[str, str]]) -> object:
        """Helper method to create a lazy __getattr__ function and its associated globals dict for testing."""
        g: dict[str, object] = {}
        return create_lazy_getattr(MappingProxyType(dispatch), g, "test_module"), g

    def test_returns_callable(self) -> None:
        """Test that create_lazy_getattr returns a callable function."""
        fn, _ = self._make_getattr({})
        assert callable(fn)

    def test_getattr_module_is_set(self) -> None:
        """Test that the __module__ attribute of the returned function is set to the provided module name."""
        fn, _ = self._make_getattr({})
        assert fn.__module__ == "test_module"

    def test_unknown_attr_raises_attribute_error(self) -> None:
        """Test that accessing an attribute not in the dispatch mapping raises an AttributeError with an appropriate message."""
        fn, _ = self._make_getattr({})
        with pytest.raises(AttributeError, match="no attribute 'missing'"):
            fn("missing")

    def test_resolves_attribute_from_module(self) -> None:
        """Test that create_lazy_getattr can resolve an attribute from a specified module and cache it in the globals dict."""
        # dispatch format: attr_name -> (package, submodule)
        # imports `package.submodule` then does getattr(submodule, attr_name)
        # so {"join": ("os", "path")} → getattr(os.path, "join") → os.path.join
        import os.path

        dispatch = {"join": ("os", "path")}
        module_globals: dict[str, object] = {}
        fn = create_lazy_getattr(MappingProxyType(dispatch), module_globals, "test_module")

        result = fn("join")
        assert result is os.path.join

    def test_caches_result_in_globals(self) -> None:
        """Test that the result of resolving an attribute is cached in the provided globals dict."""
        dispatch = {"join": ("os", "path")}
        module_globals: dict[str, object] = {}
        fn = create_lazy_getattr(MappingProxyType(dispatch), module_globals, "test_module")

        fn("join")
        assert "join" in module_globals

    def test_module_mode(self) -> None:
        """Test that when the target module is '__module__', the submodule itself is imported.

        target_module == '__module__' imports the submodule itself.
        """
        dispatch = {"path": ("os", "__module__")}
        module_globals: dict[str, object] = {}
        fn = create_lazy_getattr(MappingProxyType(dispatch), module_globals, "test_module")

        import os.path

        result = fn("path")
        assert result is os.path

    def test_error_message_includes_module_name(self) -> None:
        """Test that the error message for an unknown attribute includes the module name."""
        fn, _ = self._make_getattr({})
        with pytest.raises(AttributeError, match="test_module"):
            fn("nope")
