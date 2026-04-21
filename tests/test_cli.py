from __future__ import annotations

import subprocess
import sys


class TestSakuraCli:
    def test_module_help_is_non_destructive(self):
        result = subprocess.run(
            [sys.executable, "-m", "sakura"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "Zakuro-backed asynchronous evaluation" in result.stdout
        assert "sakura-benchmark" in result.stdout
        assert "mpirun" not in result.stdout
        assert "mpirun" not in result.stderr

    def test_module_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "sakura", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert result.stdout.startswith("sakura ")
        assert "build" in result.stdout
