# SPDX-FileCopyrightText: 2026 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Lazy import utilities for deferred module loading and attribute access.

Provides two complementary tools:

``LazyImport`` / ``lazy_import``
    A proxy object for explicit lazy imports in application code. The module
    is not imported (and the attribute is not accessed) until the object is
    actually *used* — called, passed to isinstance, subscripted, etc.

    Usage::

        from lateimport import lazy_import

        numpy = lazy_import("numpy")           # no import yet
        result = numpy.array([1, 2, 3])        # imported here

``create_lazy_getattr``
    Factory for the ``__getattr__`` hook embedded in package ``__init__.py``
    files that use a ``_dynamic_imports`` dispatch table. Used by Exportify
    when generating ``__init__.py`` files; also available directly for
    hand-authored packages.

    Usage::

        from types import MappingProxyType
        from lateimport import create_lazy_getattr

        _dynamic_imports = MappingProxyType({
            "MyClass": ("mypackage.core", "models"),
        })
        __getattr__ = create_lazy_getattr(_dynamic_imports, globals(), __name__)
"""

from __future__ import annotations

import threading

from importlib import import_module
from types import MappingProxyType, ModuleType
from typing import Any, cast


INTROSPECTION_ATTRIBUTES = frozenset({
    "__annotations__",
    "__class__",
    "__closure__",
    "__code__",
    "__defaults__",
    "__dict__",
    "__doc__",
    "__func__",
    "__globals__",
    "__kwdefaults__",
    "__module__",
    "__name__",
    "__qualname__",
    "__self__",
    "__signature__",
    "__text_signature__",
    "__wrapped__",
})
"""Dunder attributes resolved immediately rather than lazily.

These are commonly accessed during introspection (by pydantic, dataclasses,
typing machinery, etc.) and must not be proxied.
"""


class LazyImport[Import: Any]:
    """Proxy that defers both module import and attribute access until use.

    The import is triggered the first time the object is *used* — called,
    passed to ``isinstance``, subscripted, etc. — not when it is referenced.

    Thread-safe: concurrent resolution is serialised with a lock and the
    result is cached after the first successful import.

    Example::

        heavy = lazy_import("heavy_module", "SomeClass")
        # heavy_module is not imported yet
        instance = heavy()
        # heavy_module.SomeClass is imported and called here
    """

    __slots__ = ("_attrs", "_lock", "_module_name", "_parent", "_resolved")

    def __init__(self, module_name: str, *attrs: str) -> None:
        """Create a lazy import for ``module_name``, optionally drilling into ``attrs``.

        Args:
            module_name: Dotted module path to import (e.g. ``"os.path"``).
            *attrs: Attribute chain to traverse after import
                    (e.g. ``"join"`` → ``os.path.join``).
        """
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_attrs", attrs)
        object.__setattr__(self, "_resolved", None)
        object.__setattr__(self, "_parent", None)
        object.__setattr__(self, "_lock", threading.Lock())

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _resolve(self) -> Import:
        """Return the imported object, importing it on first call."""
        if object.__getattribute__(self, "_resolved") is not None:
            return object.__getattribute__(self, "_resolved")
        with object.__getattribute__(self, "_lock"):
            return self._do_resolve()

    def _do_resolve(self) -> Import:
        """Perform the actual import under the lock (double-checked)."""
        resolved = object.__getattribute__(self, "_resolved")
        if resolved is not None:
            return resolved

        module_name = object.__getattribute__(self, "_module_name")
        attrs = object.__getattribute__(self, "_attrs")

        try:
            result = __import__(module_name, fromlist=[""])
        except ImportError as e:
            msg = f"lateimport: cannot import module {module_name!r}"
            raise ImportError(msg) from e

        for i, attr in enumerate(attrs):
            try:
                result = getattr(result, attr)
            except AttributeError as e:
                attr_path = ".".join(attrs[: i + 1])
                msg = f"lateimport: module {module_name!r} has no attribute path {attr_path!r}"
                raise AttributeError(msg) from e

        object.__setattr__(self, "_resolved", result)
        self._propagate_resolved()
        return cast(Import, result)

    def _propagate_resolved(self) -> None:
        """Mark parent proxies resolved so introspection on them works."""
        parent = object.__getattribute__(self, "_parent")
        if parent is not None:
            if object.__getattribute__(parent, "_resolved") is None:
                object.__setattr__(parent, "_resolved", True)
            parent._propagate_resolved()

    # ------------------------------------------------------------------
    # Proxy protocol
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> LazyImport[Import]:
        """Return a child proxy for ``name``, or resolve immediately for introspection attrs."""
        if name in INTROSPECTION_ATTRIBUTES:
            try:
                return getattr(self._resolve(), name)
            except AttributeError as e:
                module_name = object.__getattribute__(self, "_module_name")
                attrs = object.__getattribute__(self, "_attrs")
                path = f"{module_name}.{'.'.join(attrs)}" if attrs else module_name
                raise AttributeError(
                    f"lateimport: attribute {name!r} not found on {path!r}"
                ) from e

        module_name = object.__getattribute__(self, "_module_name")
        attrs = object.__getattribute__(self, "_attrs")
        child: LazyImport[Import] = LazyImport(module_name, *attrs, name)
        object.__setattr__(child, "_parent", self)
        return child

    def __call__(self, *args: Any, **kwargs: Any) -> Import:
        """Resolve and call the imported object."""
        return self._resolve()(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """Resolve and set an attribute on the imported object."""
        setattr(self._resolve(), name, value)

    def __dir__(self) -> list[str]:
        """Resolve and delegate dir() to the imported object."""
        return dir(self._resolve())

    def __repr__(self) -> str:
        module_name = object.__getattribute__(self, "_module_name")
        attrs = object.__getattribute__(self, "_attrs")
        resolved = object.__getattribute__(self, "_resolved")
        path = f"{module_name}.{'.'.join(attrs)}" if attrs else module_name
        status = "resolved" if resolved is not None else "pending"
        return f"<LazyImport {path!r} ({status})>"

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_resolved(self) -> bool:
        """Return ``True`` if the import has been resolved."""
        return object.__getattribute__(self, "_resolved") is not None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def lazy_import[Import: Any](module_name: str, *attrs: str) -> LazyImport[Import]:
    """Create a :class:`LazyImport` proxy for *module_name*.

    Args:
        module_name: Dotted module path to import.
        *attrs: Optional attribute chain to traverse after import.

    Returns:
        A :class:`LazyImport` proxy that resolves on first use.

    Example::

        np = lazy_import("numpy")
        join = lazy_import("os.path", "join")
    """
    return LazyImport(module_name, *attrs)


def create_lazy_getattr(
    dynamic_imports: MappingProxyType[str, tuple[str, str]],
    module_globals: dict[str, object],
    module_name: str,
) -> object:
    """Create a ``__getattr__`` hook for package-level lazy imports.

    Designed to be assigned as ``__getattr__`` in a package ``__init__.py``
    alongside a ``_dynamic_imports`` dispatch table of the form::

        _dynamic_imports = MappingProxyType({
            "SymbolName": ("package.submodule", "module_file"),
        })
        __getattr__ = create_lazy_getattr(_dynamic_imports, globals(), __name__)

    The tuple value is ``(package, target_module)`` where:

    * ``package``: the dotted package path used as the ``package`` argument
      to :func:`importlib.import_module`.
    * ``target_module``: the submodule name, or ``"__module__"`` to import
      the submodule itself as the attribute.

    Args:
        dynamic_imports: Dispatch table mapping attribute names to
            ``(package, target_module)`` tuples.
        module_globals: The ``globals()`` dict of the calling module —
            used to cache resolved attributes for fast subsequent access.
        module_name: ``__name__`` of the calling module, used in error
            messages and on the generated function.

    Returns:
        A ``__getattr__`` callable suitable for assignment in ``__init__.py``.
    """

    def __getattr__(attr_name: str) -> object:  # noqa: N807
        try:
            package, target_module = dynamic_imports[attr_name]
        except KeyError as e:
            raise AttributeError(
                f"module {module_name!r} has no attribute {attr_name!r}"
            ) from e

        if target_module == "__module__":
            result = import_module(f".{attr_name}", package=package)
            module_globals[attr_name] = result
            return result

        module: ModuleType = import_module(f".{target_module}", package=package)
        result = getattr(module, attr_name)
        module_globals[attr_name] = result
        return result

    __getattr__.__module__ = module_name
    __getattr__.__doc__ = (
        f"Lazy-import ``__getattr__`` for {module_name!r}. "
        "Generated by lateimport.create_lazy_getattr."
    )
    return __getattr__


__all__ = (
    "INTROSPECTION_ATTRIBUTES",
    "LazyImport",
    "create_lazy_getattr",
    "lazy_import",
)
