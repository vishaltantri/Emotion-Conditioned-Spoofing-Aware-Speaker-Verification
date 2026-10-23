"""Tests for the ASVspoof 2019 LA protocol parser.

The parser once assumed ``speaker - utt label attack`` and returned ``'-'`` as
every utterance id, which would have produced a manifest pointing at files that
can never exist. These tests pin the real column order.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_manifests.py"


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_manifests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bonafide_line(builder):
    got = builder.parse_cm_protocol_line("LA_0079 LA_T_1138215 - - bonafide")
    assert got == ("LA_0079", "LA_T_1138215", "bonafide", "-")


def test_spoof_line_keeps_attack(builder):
    got = builder.parse_cm_protocol_line("LA_0022 LA_T_1202073 - A01 spoof")
    assert got == ("LA_0022", "LA_T_1202073", "spoof", "A01")


def test_tabs_and_extra_spaces(builder):
    got = builder.parse_cm_protocol_line("LA_0011\tLA_T_9 -\tA09   spoof")
    assert got == ("LA_0011", "LA_T_9", "spoof", "A09")


def test_utterance_id_is_never_a_placeholder(builder):
    for line in ("LA_0001 LA_T_1 - - bonafide", "LA_0001 LA_T_2 - A05 spoof"):
        _, utt, _, _ = builder.parse_cm_protocol_line(line)
        assert utt.startswith("LA_")


def test_malformed_lines_are_skipped(builder):
    assert builder.parse_cm_protocol_line("") is None
    assert builder.parse_cm_protocol_line("LA_0001 LA_T_1 -") is None
    assert builder.parse_cm_protocol_line("la_0001 la_t_1 - - nonsense") is None
