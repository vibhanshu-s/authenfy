"""Killable service worker for the trained fusion engine."""

import json
from pathlib import Path
import sys

from .errors import AnalysisError
from .fusion_analysis import analyze_fusion
from .fusion_model import FusionDetector


if __name__ == "__main__":
    try:
        result = analyze_fusion(Path(sys.argv[1]), FusionDetector(sys.argv[2]))
    except AnalysisError as error:
        result = dict(error=str(error), status=error.status, code=error.code)
    print(json.dumps(result, allow_nan=False))
