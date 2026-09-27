"""Printable tag sheets: the right tags, at the right size.

A sheet is the one artefact in the live setup nobody can inspect by eye. A tag
printed a few percent small still decodes perfectly - and puts every rack-frame
distance a few percent wrong. So the test measures the black square in pixels
against the physical size it is supposed to be at the page's dpi, and decodes
every tag back to the id it was meant to carry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_tag_sheets as sheets  # noqa: E402

from parikshak.perception.markers import PropMarkers  # noqa: E402
from parikshak.perception.rackframe import TagLayout  # noqa: E402

DPI = 150


def decode(page: np.ndarray) -> dict[int, float]:
    """tag id -> mean edge length of its black square, in pixels."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    corners, ids, _ = cv2.aruco.ArucoDetector(
        dictionary, cv2.aruco.DetectorParameters()).detectMarkers(page)
    out: dict[int, float] = {}
    for c, i in zip(corners, [] if ids is None else ids):
        pts = np.asarray(c, dtype=float).reshape(4, 2)
        edges = [np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]
        out[int(np.ravel(i)[0])] = float(np.mean(edges))
    return out


def test_every_rack_fiducial_prints_at_its_true_size():
    layout = TagLayout.load(ROOT / "racks" / "msg_a_fiducials.json")
    expected_px = sheets.mm_to_px(layout.tag_size_m * 1000.0, DPI)
    pages = dict(sheets.rack_pages(layout, DPI))
    assert set(pages) == {f"rack_{t}" for t in layout.tag_ids}
    for tag_id in layout.tag_ids:
        found = decode(pages[f"rack_{tag_id}"])
        assert set(found) == {tag_id}
        assert found[tag_id] == pytest.approx(expected_px, abs=2.0)


def test_every_prop_tag_appears_once_at_its_true_size():
    props = PropMarkers.load(ROOT / "racks" / "props_csp1.json")
    expected_px = sheets.mm_to_px(props.tag_size_m * 1000.0, DPI)
    found: dict[int, float] = {}
    for _name, page in sheets.prop_pages(props, DPI):
        for tag_id, edge in decode(page).items():
            assert tag_id not in found, f"tag {tag_id} printed twice"
            found[tag_id] = edge
    assert set(found) == set(props.tags)
    assert all(edge == pytest.approx(expected_px, abs=2.0) for edge in found.values())


def test_the_tool_writes_every_page(tmp_path):
    assert sheets.main(["--props", str(ROOT / "racks" / "props_crx2.json"),
                        "--procedure", str(ROOT / "procedures" / "crx2_colloid_resuspension.yaml"),
                        "--out", str(tmp_path), "--dpi", str(DPI), "--no-pdf"]) == 0
    names = {p.stem for p in tmp_path.glob("*.png")}
    assert {"rack_101", "rack_102", "rack_103", "rack_104", "guide"} <= names
    assert any(n.startswith("props_") for n in names)
