import sys
import pathlib

root_dir = pathlib.Path(__file__).resolve().parents[2]
if root_dir not in sys.path:
    sys.path.insert(0, str(root_dir))