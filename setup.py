"""
Build configuration for the optional compiled backend.

Package metadata lives in pyproject.toml; this file exists only to declare the
C++ extension.

The extension is optional by design.  If it fails to build -- no compiler, an
unsupported platform -- installation still succeeds and VOCD runs pure-Python
at roughly a fifth the speed, with identical output.  A hard build requirement
would make a research package unusable for the people most likely to want it.
"""

from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext_modules = [
        Pybind11Extension(
            "vocd._viterbi",
            ["vocd/_viterbi.cpp"],
            cxx_std=17,
            extra_compile_args=["-O3"],
        )
    ]
    cmdclass = {"build_ext": build_ext}
except ImportError:  # pybind11 absent: ship pure Python
    ext_modules = []
    cmdclass = {}


class _OptionalBuildExt(cmdclass.get("build_ext", object)):  # type: ignore[misc]
    """Let the wheel build even where no working compiler exists."""

    def run(self):  # pragma: no cover - depends on the build host
        try:
            super().run()
        except Exception as e:
            print(f"warning: could not build vocd._viterbi ({e});"
                  " falling back to the pure-Python backend")


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": _OptionalBuildExt} if ext_modules else {},
)
