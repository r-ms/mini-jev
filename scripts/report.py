import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from minijev.report import build
print(build(sys.argv[1:]))
