"""Root conftest for pytest.

Prevents pytest from treating the project root as a Python package.
The root __init__.py uses relative imports that only work inside
ComfyUI's custom node loader.
"""

import _pytest.python

_original_package_setup = _pytest.python.Package.setup


def _safe_package_setup(self):
    try:
        _original_package_setup(self)
    except Exception:
        pass


_pytest.python.Package.setup = _safe_package_setup
