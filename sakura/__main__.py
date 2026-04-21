"""Light-weight CLI for Sakura.

The package's primary interface is its Python APIs; the top-level CLI is an
informational entry point rather than a training orchestrator.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from sakura import __build__, __version__


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="sakura",
        description=(
            "Sakura provides Zakuro-backed asynchronous evaluation integrations "
            "for Lightning, HuggingFace Trainer, and TensorFlow/Keras."
        ),
        epilog=(
            "Use the Python APIs from sakura.lightning, sakura.huggingface, or "
            "sakura.tensorflow. The bundled benchmark entry point is "
            "`sakura-benchmark`."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} (build {__build__})",
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
