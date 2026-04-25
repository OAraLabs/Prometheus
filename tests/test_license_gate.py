"""LicenseGate — verdict mapping, file inspection, obligations."""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.symbiote.license_gate import (
    LicenseCheck,
    LicenseGate,
    LicenseVerdict,
)


class TestClassifySpdx:
    def test_mit_allowed(self):
        gate = LicenseGate()
        assert gate.classify_spdx("MIT") == LicenseVerdict.ALLOW

    def test_apache_allowed(self):
        gate = LicenseGate()
        assert gate.classify_spdx("Apache-2.0") == LicenseVerdict.ALLOW

    def test_bsd_allowed(self):
        gate = LicenseGate()
        assert gate.classify_spdx("BSD-3-Clause") == LicenseVerdict.ALLOW

    def test_isc_allowed(self):
        gate = LicenseGate()
        assert gate.classify_spdx("ISC") == LicenseVerdict.ALLOW

    def test_gpl3_blocked(self):
        gate = LicenseGate()
        assert gate.classify_spdx("GPL-3.0-only") == LicenseVerdict.BLOCK

    def test_agpl_blocked(self):
        gate = LicenseGate()
        assert gate.classify_spdx("AGPL-3.0-or-later") == LicenseVerdict.BLOCK

    def test_sspl_blocked(self):
        gate = LicenseGate()
        assert gate.classify_spdx("SSPL-1.0") == LicenseVerdict.BLOCK

    def test_busl_blocked(self):
        gate = LicenseGate()
        assert gate.classify_spdx("BUSL-1.1") == LicenseVerdict.BLOCK

    def test_lgpl3_warn(self):
        gate = LicenseGate()
        assert gate.classify_spdx("LGPL-3.0-only") == LicenseVerdict.WARN

    def test_mpl_warn(self):
        gate = LicenseGate()
        assert gate.classify_spdx("MPL-2.0") == LicenseVerdict.WARN

    def test_none_unknown(self):
        gate = LicenseGate()
        assert gate.classify_spdx(None) == LicenseVerdict.UNKNOWN

    def test_empty_unknown(self):
        gate = LicenseGate()
        assert gate.classify_spdx("") == LicenseVerdict.UNKNOWN

    def test_alias_gpl3(self):
        gate = LicenseGate()
        assert gate.classify_spdx("GPL-3") == LicenseVerdict.BLOCK
        assert gate.classify_spdx("GPLv3") == LicenseVerdict.BLOCK

    def test_alias_apache(self):
        gate = LicenseGate()
        assert gate.classify_spdx("Apache 2.0") == LicenseVerdict.ALLOW

    def test_extra_block_overrides_allow(self):
        gate = LicenseGate(extra_block=["MIT"])
        # extra_block is checked before allow.
        assert gate.classify_spdx("MIT") == LicenseVerdict.BLOCK


class TestCheckRepoData:
    def test_check_with_github_api_format(self):
        gate = LicenseGate()
        result = gate.check({"license": {"spdx_id": "MIT", "key": "mit"}})
        assert result.verdict == LicenseVerdict.ALLOW
        assert result.spdx_id == "MIT"
        assert result.source == "github_api"
        assert "Include copyright notice" in result.obligations

    def test_check_with_no_license(self):
        gate = LicenseGate()
        result = gate.check({"license": None})
        assert result.verdict == LicenseVerdict.UNKNOWN
        assert result.source == "none"

    def test_check_with_empty_dict(self):
        gate = LicenseGate()
        result = gate.check({})
        assert result.verdict == LicenseVerdict.UNKNOWN

    def test_check_with_none(self):
        gate = LicenseGate()
        result = gate.check(None)
        assert result.verdict == LicenseVerdict.UNKNOWN
        assert result.spdx_id is None

    def test_check_blocks_gpl(self):
        gate = LicenseGate()
        result = gate.check({"license": {"spdx_id": "GPL-3.0-only"}})
        assert result.verdict == LicenseVerdict.BLOCK


class TestCheckFile:
    def test_file_does_not_exist(self, tmp_path):
        gate = LicenseGate()
        result = gate.check_file(tmp_path / "nonexistent.md")
        assert result.verdict == LicenseVerdict.UNKNOWN

    def test_spdx_header_priority(self, tmp_path):
        f = tmp_path / "LICENSE"
        f.write_text("// SPDX-License-Identifier: Apache-2.0\nSome other text\n")
        gate = LicenseGate()
        result = gate.check_file(f)
        assert result.verdict == LicenseVerdict.ALLOW
        assert result.spdx_id == "Apache-2.0"
        assert result.source == "spdx_header"

    def test_mit_keyword(self, tmp_path):
        f = tmp_path / "LICENSE"
        f.write_text(
            "MIT License\n\n"
            "Permission is hereby granted, free of charge, to any person...\n"
        )
        gate = LicenseGate()
        result = gate.check_file(f)
        assert result.verdict == LicenseVerdict.ALLOW
        assert result.spdx_id == "MIT"

    def test_gpl_keyword_blocked(self, tmp_path):
        f = tmp_path / "COPYING"
        f.write_text(
            "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n"
        )
        gate = LicenseGate()
        result = gate.check_file(f)
        assert result.verdict == LicenseVerdict.BLOCK
        assert result.spdx_id == "GPL-3.0-only"

    def test_agpl_keyword_blocked(self, tmp_path):
        f = tmp_path / "LICENSE"
        f.write_text("GNU AFFERO GENERAL PUBLIC LICENSE\nblah blah\n")
        gate = LicenseGate()
        result = gate.check_file(f)
        assert result.verdict == LicenseVerdict.BLOCK


class TestObligations:
    def test_apache_obligations(self):
        gate = LicenseGate()
        obs = gate.format_obligations("Apache-2.0")
        assert "State changes made" in obs
        assert "Include NOTICE file if present" in obs

    def test_mit_obligations(self):
        gate = LicenseGate()
        obs = gate.format_obligations("MIT")
        assert "Include copyright notice" in obs
        assert "Include license text" in obs

    def test_unknown_obligations(self):
        gate = LicenseGate()
        assert gate.format_obligations(None) == []
        assert gate.format_obligations("") == []
        assert gate.format_obligations("nonexistent-spdx") == []
