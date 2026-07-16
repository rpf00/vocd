"""
Build the compiled Viterbi backend in place.

    python setup_ext.py

The extension is optional: VOCD runs pure-Python without it, just slower.
``pip install -e .`` builds it automatically via setup.py; this script is for
rebuilding after editing the C++ without reinstalling.
"""

import os
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "vocd", "_viterbi.cpp")


def main() -> int:
    try:
        import pybind11
    except ImportError:
        print("pybind11 is required to build the extension:\n"
              "  pip install pybind11", file=sys.stderr)
        return 1

    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = os.path.join(HERE, "vocd", "_viterbi" + suffix)

    cmd = [
        os.environ.get("CXX", "c++"),
        "-O3", "-std=c++17", "-shared", "-fPIC",
        f"-I{pybind11.get_include()}",
        f"-I{sysconfig.get_paths()['include']}",
        SRC, "-o", out,
    ]
    if sys.platform == "darwin":
        cmd += ["-undefined", "dynamic_lookup"]

    print(" ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print("build failed", file=sys.stderr)
        return rc

    print(f"built {out}")
    sys.path.insert(0, HERE)
    from vocd import _backend
    print("backend available:", _backend.available())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
