"""Isolated bounded request worker. Receives only server-generated local paths."""

import json
import sys
from pathlib import Path

from .analysis import analyze
from .errors import AnalysisError
from .model import Detector


def main():
    try:
        result = analyze(Path(sys.argv[1]), Detector(sys.argv[2]))
    except AnalysisError as error:
        result = {"error": str(error), "status": error.status, "code": error.code}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
