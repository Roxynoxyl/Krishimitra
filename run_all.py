# Orchestrate the full pipeline:
#   validate -> evaluate (train + select + E1-E3) -> export web artifacts
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python src/run_all.py`

from src import validate, evaluate, export_artifacts


def main():
    print("=" * 70)
    print("STEP 1/3  data validation & profiling")
    print("=" * 70)
    validate.main()

    print("=" * 70)
    print("STEP 2/3  train, select, evaluate (E1-E3)")
    print("=" * 70)
    evaluate.main()

    print("=" * 70)
    print("STEP 3/3  export web artifacts")
    print("=" * 70)
    export_artifacts.main()
    print("DONE")


if __name__ == "__main__":
    main()
