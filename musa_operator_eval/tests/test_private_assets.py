"""The maintainer-side assets must stay out of the agent-visible repository.

This is the whole of the publication separation: the task package is committed
and therefore agent-visible, and the expert dispatch, gap evidence, hidden
manifest, environment snapshot and admission decision live under `private/`,
which `.gitignore` excludes. There is no export step that could filter them out
later, so if one of them is ever tracked it is already published.

These tests check the rule rather than the current file listing, so they still
mean something on a fresh clone where `private/` holds only its README.

Run with either:
    python -m unittest musa_operator_eval.tests.test_private_assets -v
    python musa_operator_eval/tests/test_private_assets.py -v
"""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_TOP = ROOT.parent
PRIVATE_PREFIX = "musa_operator_eval/private/"
PRIVATE_README = PRIVATE_PREFIX + "README.md"

# A path that must be ignored if the rule is in place. It does not have to exist:
# `git check-ignore` answers from the pattern, not the filesystem.
PROBE_UNDER_PRIVATE = PRIVATE_PREFIX + "sdpa_forward_b_v0/expert/model_new.py"

# Files that exist only on the maintainer side and must not appear in a task
# package, whatever the ignore rules say.
MAINTAINER_ONLY_NAMES = {"cases.private.json", "baseline.json", "admission.json", "private_gap_evidence.json"}


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO_TOP, capture_output=True, text=True)


def git_available():
    try:
        return git("rev-parse", "--is-inside-work-tree").returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@unittest.skipUnless(git_available(), "requires a git working tree")
class IgnoreRuleTests(unittest.TestCase):
    def test_arbitrary_private_files_are_ignored(self):
        """Checked against the pattern, so this holds on a fresh clone too."""
        self.assertEqual(
            git("check-ignore", "-q", PROBE_UNDER_PRIVATE).returncode, 0,
            f"{PROBE_UNDER_PRIVATE} is not ignored; maintainer assets would be committed",
        )

    def test_the_private_readme_is_deliberately_not_ignored(self):
        self.assertEqual(
            git("check-ignore", "-q", PRIVATE_README).returncode, 1,
            "the private README is the documented exception and must stay tracked",
        )


@unittest.skipUnless(git_available(), "requires a git working tree")
class TrackedFileTests(unittest.TestCase):
    def _tracked_under_private(self):
        result = git("ls-files", "--", "musa_operator_eval/private")
        return [line for line in result.stdout.splitlines() if line.strip()]

    def test_only_the_readme_is_tracked(self):
        tracked = self._tracked_under_private()
        self.assertEqual(
            tracked, [PRIVATE_README],
            f"private assets are tracked and therefore published: {tracked}",
        )

    def test_every_present_private_file_is_ignored(self):
        """Catches a negated pattern added above the directory rule."""
        private_dir = ROOT / "private"
        if not private_dir.is_dir():
            self.skipTest("no private directory on this checkout")
        for path in private_dir.rglob("*"):
            if not path.is_file() or path.name == "__pycache__":
                continue
            relative = path.relative_to(REPO_TOP).as_posix()
            if relative == PRIVATE_README:
                continue
            with self.subTest(path=relative):
                self.assertEqual(
                    git("check-ignore", "-q", relative).returncode, 0,
                    f"{relative} is not ignored",
                )

    def test_maintainer_only_files_are_not_in_task_packages(self):
        for task_dir in sorted((ROOT / "tasks").glob("*")):
            if not task_dir.is_dir():
                continue
            for path in task_dir.rglob("*"):
                if path.name in MAINTAINER_ONLY_NAMES:
                    self.fail(f"{path.relative_to(REPO_TOP)} is a maintainer-only file inside a task package")


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
