import sys
from pathlib import Path

# Make `src/` importable as top-level modules (mall_segmentation, bigmart_sales_prediction)
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
