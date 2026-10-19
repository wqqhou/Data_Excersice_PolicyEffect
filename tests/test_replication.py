"""Checks for portable, fail-fast orchestration without rerunning the analysis."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import replication


class ReplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.project = Path(self.temp_directory.name)
        raw = self.project / "data" / "raw"
        raw.mkdir(parents=True)
        for filename in ("monthly_individual_data.csv", "monthly_state_data.csv"):
            (raw / filename).touch()
        (self.project / "main.tex").touch()
        scripts = self.project / "scripts"
        scripts.mkdir()
        for script in (
            "data_clean.py", "plot.py", "balance_test.py", "estimate.py", "count_movers.py"
        ):
            (scripts / script).touch()
        root_patch = patch.object(replication, "PROJECT_ROOT", self.project)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_skip_pdf_runs_all_stages_with_project_cwd_and_headless_backend(self):
        with (
            patch("replication.shutil.which", return_value=None),
            patch("replication.subprocess.run") as run,
            patch.dict("os.environ", {"MPLBACKEND": "TkAgg"}),
        ):
            replication.run_replication(skip_pdf=True)

        self.assertEqual(
            [Path(call.args[0][1]).name for call in run.call_args_list],
            ["data_clean.py", "plot.py", "balance_test.py", "estimate.py", "count_movers.py"],
        )
        for call in run.call_args_list:
            self.assertEqual(call.args[0][0], sys.executable)
            self.assertEqual(call.kwargs["cwd"], self.project)
            self.assertEqual(call.kwargs["env"]["MPLBACKEND"], "Agg")
            self.assertIs(call.kwargs["check"], True)

    def test_failed_stage_stops_downstream_outputs(self):
        failure = subprocess.CalledProcessError(2, [sys.executable, "plot.py"])
        with patch("replication.subprocess.run", side_effect=[None, failure]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                replication.run_replication(skip_pdf=True)
        self.assertEqual(run.call_count, 2)

    def test_pdf_fallback_runs_two_passes_after_analysis(self):
        def executable(name):
            return "/tex/pdflatex" if name == "pdflatex" else None

        with (
            patch("replication.shutil.which", side_effect=executable),
            patch("replication.subprocess.run") as run,
        ):
            replication.run_replication()

        self.assertEqual(run.call_count, 7)
        self.assertEqual(run.call_args_list[-2], run.call_args_list[-1])
        self.assertEqual(run.call_args_list[-1].args[0][0], "/tex/pdflatex")
        self.assertEqual(run.call_args_list[-1].kwargs["cwd"], self.project)
        self.assertIs(run.call_args_list[-1].kwargs["check"], True)


if __name__ == "__main__":
    unittest.main()
