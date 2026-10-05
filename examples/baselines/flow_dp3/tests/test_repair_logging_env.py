"""修复脚本的边界检查：下载失败不得更改包，系统 Python 不得执行修复。"""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import repair_logging_env as repair


class RepairTests(unittest.TestCase):
    def test_system_python_is_rejected_before_any_pip_command(self):
        with patch.object(sys, "argv", ["repair_logging_env.py"]), \
             patch.object(sys, "prefix", "/usr/local"), \
             patch.object(sys, "base_prefix", "/usr/local"), \
             patch.object(repair, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "激活本项目"):
                repair.main()
            run.assert_not_called()

    def test_failed_download_never_installs_or_uninstalls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".runtime").mkdir()
            (root / ".runtime/constraints-ppu.txt").write_text("torch==2.9.0\n")
            local_override = SimpleNamespace(metadata={"Name": "opentelemetry-api"}, version="1.45.0")
            with patch.object(repair, "ROOT", root), \
                 patch.object(sys, "prefix", str(root / ".venv-ppu")), \
                 patch.object(sys, "base_prefix", "/usr/local"), \
                 patch.object(sys, "argv", ["repair_logging_env.py"]), \
                 patch.object(repair.metadata, "version", return_value="2.9.0"), \
                 patch.object(repair.metadata, "distributions", side_effect=[[local_override], []]), \
                 patch.object(repair, "run", side_effect=subprocess.CalledProcessError(1, "download")) as run, \
                 redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    repair.main()
            self.assertEqual(run.call_count, 1)
            self.assertIn("download", run.call_args.args[0])
            self.assertFalse((root / ".runtime/wandb-dependency-repair/last-success.json").exists())


if __name__ == "__main__":
    unittest.main()
