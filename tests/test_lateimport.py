# SPDX-FileCopyrightText: 2026 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tests for lateimport."""

from __future__ import annotations

import threading

from types import MappingProxyType

import pytest

from lateimport import LateImport, create_late_getattr, lateimport


# ---------------------------------------------------------------------------
# LateImport / lateimport
# ---------------------------------------------------------------------------


class TestLateImportCreation:
    """Test creation and basic properties of LateImport objects returned by lateimport()."""

    def test_not_resolved_on_creation(self) -> None:
        """Test that a new LateImport is not resolved."""
        proxy = lateimport("os")
        assert not proxy.is_resolved()

    def test_repr_pending(self) -> None:
        """Test that the repr of a pending LateImport indicates it's pending."""
        proxy = lateimport("os")
        assert "pending" in repr(proxy)
        assert "os" in repr(proxy)

    def test_repr_resolved(self) -> None:
        """Test that the repr of a resolved LateImport indicates it's resolved."""
        proxy = lateimport("os")
        _ = proxy._resolve()
        assert "resolved" in repr(proxy)

    def test_lateimport_returns_lateimport_instance(self) -> None:
        """Test that lateimport() returns an instance of LateImport."""
        proxy = lateimport("os")
        assert isinstance(proxy, LateImport)


class TestLateImportResolution:
    """Test the resolution behavior of LateImport objects, including caching and error handling."""

    def test_resolves_module(self) -> None:
        """Test that a LateImport can resolve a module."""
        import os

        proxy = lateimport("os")
        assert proxy._resolve() is os

    def test_resolves_attribute(self) -> None:
        """Test that a LateImport can resolve an attribute of a module."""
        import os.path

        proxy = lateimport("os.path", "join")
        assert proxy._resolve() is os.path.join

    def test_call_resolves_and_calls(self) -> None:
        """Test that calling a LateImport resolves it and calls the resulting object."""
        proxy = lateimport("os.path", "join")
        result = proxy("a", "b")
        assert result == "a/b"

    def test_is_resolved_after_resolution(self) -> None:
        """Test that is_resolved() returns True after the LateImport has been resolved."""
        proxy = lateimport("os")
        proxy._resolve()
        assert proxy.is_resolved()

    def test_resolution_cached(self) -> None:
        """Test that resolving a LateImport multiple times returns the same object (cached)."""
        proxy = lateimport("os")
        first = proxy._resolve()
        second = proxy._resolve()
        assert first is second

    def test_do_resolve_double_check(self) -> None:
        """Test the double-checked locking path: _do_resolve returns early if already resolved.

        This exercises the branch where a second thread acquires the lock after the first
        thread has already set ``_resolved``, hitting the early-return guard.
        """
        import os

        proxy = lateimport("os")
        object.__setattr__(proxy, "_resolved", os)
        assert proxy._do_resolve() is os


class TestLateImportAttributeChaining:
    """Test that accessing attributes on a LateImport returns new LateImport proxies that can be resolved correctly."""

    def test_chained_attribute_creates_child_proxy(self) -> None:
        """Test that accessing an attribute on a LateImport returns a new LateImport proxy for that attribute."""
        proxy = lateimport("os")
        child = proxy.path
        assert isinstance(child, LateImport)
        assert not child.is_resolved()

    def test_chained_resolution(self) -> None:
        """Test that accessing an attribute on a LateImport and then resolving it works correctly."""
        import os.path

        proxy = lateimport("os")
        assert proxy.path.join._resolve() is os.path.join

    def test_introspection_attr_resolves_immediately(self) -> None:
        """Test that accessing an introspection attribute (like __name__) on a LateImport resolves the module immediately."""
        proxy = lateimport("os")
        # __name__ is in INTROSPECTION_ATTRIBUTES — should resolve the module
        assert proxy.__name__ == "os"
        assert proxy.is_resolved()


class TestLateImportErrors:
    """Test error handling behavior of LateImport, including import errors and attribute errors."""

    def test_bad_module_raises_import_error(self) -> None:
        """Test that a LateImport raises ImportError for a non-existent module."""
        proxy = lateimport("no_such_module_xyzzy")
        with pytest.raises(ImportError, match="no_such_module_xyzzy"):
            proxy._resolve()

    def test_bad_attribute_raises_attribute_error(self) -> None:
        """Test that a LateImport raises AttributeError for a non-existent attribute."""
        proxy = lateimport("os", "no_such_attr_xyzzy")
        with pytest.raises(AttributeError, match="no_such_attr_xyzzy"):
            proxy._resolve()

    def test_bad_introspection_attr_raises_attribute_error(self) -> None:
        """Test that a LateImport raises AttributeError for a non-existent introspection attribute."""
        # Python resolves dunder attrs on the type, not the instance, so we
        # call __getattr__ directly to exercise the introspection branch.
        proxy = lateimport("os", "no_such_attr_xyzzy")
        with pytest.raises(AttributeError):
            proxy.__getattr__("__doc__")


class TestLateImportThreadSafety:
    """Test that LateImport resolution is thread-safe and that concurrent resolutions return the same object."""

    def test_concurrent_resolution_returns_same_object(self) -> None:
        """Test that resolving the same LateImport concurrently from multiple threads returns the same resolved object."""
        proxy = lateimport("os")
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
# create_late_getattr
# ---------------------------------------------------------------------------


class TestCreateLazyGetattr:
    """Test the behavior of the create_late_getattr function, which creates a __getattr__ function that lazily imports attributes based on a dispatch mapping."""

    def _make_getattr(self, dispatch: dict[str, tuple[str, str]]) -> object:
        """Helper method to create a lazy __getattr__ function and its associated globals dict for testing."""
        g: dict[str, object] = {}
        return create_late_getattr(MappingProxyType(dispatch), g, "test_module"), g

    def test_returns_callable(self) -> None:
        """Test that create_late_getattr returns a callable function."""
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
        """Test that create_late_getattr can resolve an attribute from a specified module and cache it in the globals dict."""
        # dispatch format: attr_name -> (package, submodule)
        # imports `package.submodule` then does getattr(submodule, attr_name)
        # so {"join": ("os", "path")} → getattr(os.path, "join") → os.path.join
        import os.path

        dispatch = {"join": ("os", "path")}
        module_globals: dict[str, object] = {}
        fn = create_late_getattr(MappingProxyType(dispatch), module_globals, "test_module")

        result = fn("join")
        assert result is os.path.join

    def test_caches_result_in_globals(self) -> None:
        """Test that the result of resolving an attribute is cached in the provided globals dict."""
        dispatch = {"join": ("os", "path")}
        module_globals: dict[str, object] = {}
        fn = create_late_getattr(MappingProxyType(dispatch), module_globals, "test_module")

        fn("join")
        assert "join" in module_globals

    def test_module_mode(self) -> None:
        """Test that when the target module is '__module__', the submodule itself is imported.

        target_module == '__module__' imports the submodule itself.
        """
        dispatch = {"path": ("os", "__module__")}
        module_globals: dict[str, object] = {}
        fn = create_late_getattr(MappingProxyType(dispatch), module_globals, "test_module")

        import os.path

        result = fn("path")
        assert result is os.path

    def test_error_message_includes_module_name(self) -> None:
        """Test that the error message for an unknown attribute includes the module name."""
        fn, _ = self._make_getattr({})
        with pytest.raises(AttributeError, match="test_module"):
            fn("nope")


# ---------------------------------------------------------------------------
# Version metadata
# ---------------------------------------------------------------------------


class TestVersionInfo:
    """Test that version metadata is available."""

    def test_version_is_string(self) -> None:
        """Test that __version__ is defined as a string in lateimport._version."""
        from lateimport._version import __version__

        assert isinstance(__version__, str)
        assert __version__  # non-empty
