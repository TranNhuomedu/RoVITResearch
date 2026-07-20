"""rovit.rotation (tai tao) -- re-export tu rotation.py o goc campaign."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import importlib
_m = importlib.import_module("rotation")
build_rotations = _m.build_rotations
make_matrix = _m.make_matrix
