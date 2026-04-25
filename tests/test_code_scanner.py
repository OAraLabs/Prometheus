"""DangerousCodeScanner — AST-based detection of risky constructs."""

from __future__ import annotations

import pytest

from prometheus.symbiote.code_scanner import (
    DangerousCodeScanner,
    ScanVerdict,
)


class TestDangerousCalls:
    def test_exec_at_any_scope_dangerous(self):
        s = DangerousCodeScanner()
        result = s.scan_content(
            "def f():\n    exec('print(1)')\n",
            file_path="x.py",
        )
        assert result.verdict == ScanVerdict.DANGEROUS
        assert any(f.rule == "exec_call" for f in result.findings)

    def test_eval_at_module_scope_dangerous(self):
        s = DangerousCodeScanner()
        result = s.scan_content("x = eval('1+1')\n", file_path="x.py")
        assert result.verdict == ScanVerdict.DANGEROUS

    def test_compile_dangerous(self):
        s = DangerousCodeScanner()
        result = s.scan_content("compile('x', '<s>', 'exec')\n", file_path="x.py")
        assert result.verdict == ScanVerdict.DANGEROUS

    def test_dunder_import_dangerous(self):
        s = DangerousCodeScanner()
        result = s.scan_content('m = __import__("os")\n', file_path="x.py")
        assert result.verdict == ScanVerdict.DANGEROUS

    def test_os_system_dangerous(self):
        s = DangerousCodeScanner()
        result = s.scan_content('import os\nos.system("ls")\n', file_path="x.py")
        assert result.verdict == ScanVerdict.DANGEROUS

    def test_pty_spawn_dangerous(self):
        s = DangerousCodeScanner()
        result = s.scan_content('import pty\npty.spawn(["bash"])\n', file_path="x.py")
        assert result.verdict == ScanVerdict.DANGEROUS


class TestModuleScopeSuspicious:
    def test_subprocess_at_module_scope_suspicious(self):
        s = DangerousCodeScanner()
        result = s.scan_content(
            'import subprocess\nsubprocess.run(["ls"])\n',
            file_path="x.py",
        )
        assert result.verdict == ScanVerdict.SUSPICIOUS
        assert any("subprocess" in f.rule for f in result.findings)

    def test_subprocess_inside_function_clean(self):
        s = DangerousCodeScanner()
        result = s.scan_content(
            "import subprocess\n"
            "def run_it():\n"
            "    subprocess.run(['ls'])\n",
            file_path="x.py",
        )
        # subprocess inside a function is normal tool behavior.
        assert result.verdict == ScanVerdict.CLEAN

    def test_httpx_module_scope_suspicious(self):
        s = DangerousCodeScanner()
        result = s.scan_content(
            'import httpx\nhttpx.get("https://example.com")\n',
            file_path="x.py",
        )
        assert result.verdict == ScanVerdict.SUSPICIOUS

    def test_socket_module_scope_suspicious(self):
        s = DangerousCodeScanner()
        result = s.scan_content(
            "import socket\ns = socket.socket()\n",
            file_path="x.py",
        )
        assert result.verdict == ScanVerdict.SUSPICIOUS


class TestCleanCode:
    def test_normal_module_clean(self):
        s = DangerousCodeScanner()
        result = s.scan_content(
            '"""docstring"""\n'
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "class Foo:\n"
            "    def bar(self):\n"
            "        return 1\n",
            file_path="x.py",
        )
        assert result.verdict == ScanVerdict.CLEAN
        assert result.findings == []

    def test_non_python_clean(self):
        s = DangerousCodeScanner()
        result = s.scan_content(
            "this is not Python; it's prose with eval() in it",
            file_path="readme.md",
        )
        assert result.verdict == ScanVerdict.CLEAN


class TestSyntaxError:
    def test_syntax_error_suspicious(self):
        s = DangerousCodeScanner()
        result = s.scan_content("def broken(:\n    pass\n", file_path="x.py")
        assert result.verdict == ScanVerdict.SUSPICIOUS
        assert any(f.rule == "syntax_error" for f in result.findings)


class TestScanFile:
    def test_scan_file_reads_and_scans(self, tmp_path):
        p = tmp_path / "x.py"
        p.write_text('exec("hi")\n')
        s = DangerousCodeScanner()
        result = s.scan_file(p)
        assert result.verdict == ScanVerdict.DANGEROUS
        assert result.file_path == str(p)
