"""The B tier's link whitelist stage: read a build product's dynamic section.

§4.7's stage reads the export table of what was built, which means an ELF reader
that has to be right about two things: which sonames the object needs, and that a
file it cannot read is not a file that links nothing. Both are checked here, and
the reader is checked against `readelf` where the tool is installed -- a reader
that agrees with the tool it replaces is the only evidence that the replacement
did not invent its own answer.

Run with:
    python musa_operator_eval/tests/test_link_whitelist.py -v
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_TOP = ROOT.parent

if str(ROOT / "evaluator") not in sys.path:
    sys.path.insert(0, str(ROOT / "evaluator"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


link_whitelist = _load("link_whitelist", ROOT / "evaluator" / "link_whitelist.py")
checker = _load("kernelbench_static_checker", REPO_TOP / "src" / "kernelbench" / "kernel_static_checker.py")

# A real ELF object that exists on any Linux host. It is used as a fixture rather
# than a file built for the test, because the reader's whole job is to agree with
# the linker about files it did not produce.
CANDIDATE_SYSTEM_OBJECTS = ("/usr/bin/ls", "/bin/ls", "/usr/bin/env", "/bin/sh")


def system_object() -> Path:
    for candidate in CANDIDATE_SYSTEM_OBJECTS:
        path = Path(candidate)
        if path.is_file():
            return path
    raise unittest.SkipTest("no system ELF object to read")


def readelf_available() -> bool:
    return shutil.which("readelf") is not None


class ElfReaderTests(unittest.TestCase):
    def test_the_reader_agrees_with_readelf_on_the_needed_libraries(self):
        """The tool the stage replaces is the tool it is checked against."""
        if not readelf_available():
            self.skipTest("readelf is not installed")
        target = system_object()
        described = link_whitelist.read_elf_report(target)
        output = subprocess.run(["readelf", "-d", str(target)], capture_output=True, text=True).stdout
        expected = sorted(line.split("[")[1].split("]")[0] for line in output.splitlines() if "NEEDED" in line)
        self.assertEqual(sorted(described["needed"]), expected)
        self.assertTrue(described["needed"], "a system binary with no dynamic dependencies is not a fixture")

    def test_the_reader_agrees_with_nm_on_the_undefined_symbols(self):
        if shutil.which("nm") is None:
            self.skipTest("nm is not installed")
        target = system_object()
        described = link_whitelist.read_elf_report(target)
        output = subprocess.run(["nm", "-D", "--undefined-only", str(target)], capture_output=True, text=True).stdout
        expected = sorted({line.split()[-1].split("@")[0] for line in output.splitlines() if line.strip()})
        self.assertEqual(sorted(described["undefined_symbols"]), expected)

    def test_a_file_that_is_not_an_object_is_an_error_not_an_empty_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "notes.so"
            path.write_text("this is not an ELF file\n", encoding="utf-8")
            with self.assertRaises(link_whitelist.ElfError):
                link_whitelist.read_elf_report(path)


class BuildDirectoryTests(unittest.TestCase):
    def _report(self, directory: Path, allowed=None, prefixes=None):
        return link_whitelist.check_build_dir(directory, allowed or [], prefixes or [])

    def test_a_directory_with_no_artifact_is_unchecked_rather_than_passed(self):
        """A whitelist check that passed because nothing was built claims too much."""
        with tempfile.TemporaryDirectory() as temporary:
            report = self._report(Path(temporary))
        self.assertFalse(report["ran"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["checked"], 0)
        self.assertIn("nothing was checked", report["detail"])

    def test_a_missing_directory_is_unchecked(self):
        report = self._report(Path("/nonexistent/build/dir"))
        self.assertFalse(report["ran"])
        self.assertEqual(report["checked"], 0)

    def test_an_unreadable_artifact_fails_the_stage(self):
        """'Could not be read' must not be filed under 'links nothing'."""
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "kernel.so").write_text("not an object\n", encoding="utf-8")
            report = self._report(Path(temporary))
        self.assertTrue(report["ran"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["checked"], 0)
        self.assertTrue(any("could not be read" in error for error in report["errors"]), report["errors"])

    def test_a_whitelisted_dependency_set_passes(self):
        target = system_object()
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "libattention.so"
            shutil.copyfile(target, artifact)
            described = link_whitelist.read_elf_report(artifact)
            report = self._report(Path(temporary), allowed=list(described["needed"]), prefixes=["attention"])
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["checked"], 1)

    def test_a_non_whitelisted_dependency_fails_and_names_the_library(self):
        target = system_object()
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "libattention.so"
            shutil.copyfile(target, artifact)
            report = self._report(Path(temporary), allowed=["libmusa"], prefixes=["musa"])
        self.assertFalse(report["passed"])
        self.assertTrue(any("non-whitelisted shared library" in error for error in report["errors"]), report["errors"])
        self.assertTrue(any("lib" in error for error in report["errors"]))

    def test_the_stage_reads_the_contract_it_is_given(self):
        """The tool takes a task package and reads its whitelist, not a default."""
        parser = link_whitelist.build_parser()
        args = parser.parse_args(["--build-dir", "/tmp/x", "--task-dir", str(ROOT / "tasks" / "kb_l3_43_b")])
        task = json.loads((args.task_dir / "task.json").read_text(encoding="utf-8"))
        self.assertEqual(task["library_policy"]["allowed_libraries"], ["libmusa", "libmudnn"])


class BuildProductTests(unittest.TestCase):
    """Which files under a build directory the stage has an opinion about."""

    def test_a_suffixless_executable_is_a_build_product(self):
        """The linker's output has no suffix, and the object files have no dynamic section."""
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            shutil.copyfile("/bin/true", build / "runner")
            (build / "runner.o").write_bytes(b"\x7fELF" + b"\x00" * 60)
            (build / "notes.txt").write_text("a log, not a build product")
            names = sorted(path.name for path in link_whitelist.build_products(build))
            self.assertEqual(names, ["runner", "runner.o"])

    def test_a_mangled_object_is_still_read(self):
        """A file that lost its magic but kept its suffix is reported, not skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            (build / "libbroken.so").write_bytes(b"not an elf at all")
            names = sorted(path.name for path in link_whitelist.build_products(build))
            self.assertEqual(names, ["libbroken.so"])

    def test_a_linked_executable_is_checked_and_its_libraries_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            shutil.copyfile("/bin/true", build / "runner")
            report = link_whitelist.check_build_dir(build, ["libc"], [])
            self.assertTrue(report["ran"])
            self.assertEqual(report["checked"], 1)
            artifact = report["artifacts"][0]
            self.assertTrue(artifact["readable"], artifact)
            self.assertTrue(report["passed"], report["errors"])


class LinkPolicyTests(unittest.TestCase):
    """The rule the stage applies, at the level the contract states it."""

    ALLOWED = ["libmusa", "libmudnn"]
    PREFIXES = ["musa", "mudnn"]

    def check(self, libraries, allowed=None, prefixes=None):
        return checker.check_linked_libraries(
            libraries,
            self.ALLOWED if allowed is None else allowed,
            self.PREFIXES if prefixes is None else prefixes,
        )

    def test_the_c_runtime_and_the_toolchain_are_not_a_library_choice(self):
        for library in ("libc.so.6", "libstdc++.so.6", "libgcc_s.so.1", "ld-linux-x86-64.so.2", "libm.so.6"):
            with self.subTest(library=library):
                self.assertEqual(self.check([library]), (False, ""))

    def test_the_torch_stack_is_plumbing(self):
        for library in ("libtorch.so", "libtorch_cpu.so", "libc10_musa.so"):
            with self.subTest(library=library):
                self.assertEqual(self.check([library]), (False, ""))

    def test_a_whitelisted_library_passes_by_name_and_by_prefix(self):
        self.assertEqual(self.check(["libmudnn.so.2"]), (False, ""))
        self.assertEqual(self.check(["libmusa_rt.so"]), (False, ""))

    def test_a_library_outside_the_whitelist_is_rejected_and_named(self):
        has_issue, message = self.check(["libc.so.6", "libflashattention.so"])
        self.assertTrue(has_issue)
        self.assertIn("libflashattention.so", message)

    def test_a_whitelist_of_nothing_rejects_the_task_own_libraries(self):
        self.assertTrue(self.check(["libmudnn.so.2"], allowed=[], prefixes=[])[0])

    def test_the_version_suffix_is_not_part_of_the_library_name(self):
        self.assertEqual(checker.normalize_library_name("libc.so.6"), "c")
        self.assertEqual(checker.normalize_library_name("/opt/musa/lib/libmudnn.so.2"), "mudnn")
        self.assertEqual(checker.normalize_library_name("libstdc++.so.6"), "stdc++")


if __name__ == "__main__":
    unittest.main()
