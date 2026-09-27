#!/usr/bin/env python3
"""Printable AprilTag sheets for a live run: the rack fiducials and the props.

    python tools/make_tag_sheets.py --props racks/props_csp1.json
    python tools/make_tag_sheets.py --props racks/props_crx2.json --out runs/tags --dpi 300

Writes one PNG per page at the chosen dpi, and tags.pdf with exact millimetre
sizing when PySide6 is installed. Print at 100% ("actual size"), then measure a
black square with a ruler before sticking anything down: a sheet printed with
"fit to page" is a few percent small, and every rack-frame distance comes out a
few percent wrong with nothing on screen to say so.

Pages:
  - one per rack fiducial, labelled with where its centre goes on the rack face;
  - prop tags gridded on A4, labelled with the entity and state each stands for;
  - a placement guide: the rack face from the crew's side, with fiducials and the
    procedure's zones drawn to scale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parikshak.perception.markers import PropMarkers  # noqa: E402
from parikshak.perception.rackframe import TagLayout  # noqa: E402

A4_MM = (210.0, 297.0)
MARGIN_MM = 12.0
#: White border around each black square, as a fraction of its edge. ArUco needs
#: a quiet zone to find the square at all; a quarter-edge is comfortably enough.
QUIET_FRACTION = 0.25


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def blank_page(dpi: int) -> np.ndarray:
    w, h = (mm_to_px(v, dpi) for v in A4_MM)
    return np.full((h, w), 255, dtype=np.uint8)


def tag_image(tag_id: int, size_m: float, dpi: int) -> np.ndarray:
    """The black square at exactly `size_m`, with no quiet zone."""
    import cv2
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.generateImageMarker(dictionary, tag_id, mm_to_px(size_m * 1000.0, dpi))


def put_text(page: np.ndarray, text: str, x: int, y: int, dpi: int, scale: float = 1.0) -> None:
    import cv2
    cv2.putText(page, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale * dpi / 300.0,
                0, max(1, int(2 * dpi / 300.0)), cv2.LINE_AA)


def rack_pages(layout: TagLayout, dpi: int) -> list[tuple[str, np.ndarray]]:
    pages = []
    for tag_id in layout.tag_ids:
        page = blank_page(dpi)
        marker = tag_image(tag_id, layout.tag_size_m, dpi)
        side = marker.shape[0]
        x0 = (page.shape[1] - side) // 2
        y0 = mm_to_px(60.0, dpi)
        page[y0:y0 + side, x0:x0 + side] = marker
        centre = layout.corners[tag_id].mean(axis=0)
        m = mm_to_px(MARGIN_MM, dpi)
        put_text(page, f"RACK FIDUCIAL {tag_id}  ({layout.family})", m, m + mm_to_px(8, dpi), dpi, 1.3)
        put_text(page, f"black square must measure {layout.tag_size_m * 1000:.1f} mm - "
                       f"print at 100%", m, m + mm_to_px(18, dpi), dpi)
        put_text(page, f"centre at rack x = {centre[0]:+.3f} m, y = {centre[1]:+.3f} m "
                       f"(+Y up, facing the crew)", m, y0 + side + mm_to_px(15, dpi), dpi)
        pages.append((f"rack_{tag_id}", page))
    return pages


def prop_pages(props: PropMarkers, dpi: int) -> list[tuple[str, np.ndarray]]:
    side = mm_to_px(props.tag_size_m * 1000.0, dpi)
    # Neighbours share the white between them: one quiet zone per gap, not two,
    # plus room for the label. Two quiet zones and a wide label gap fitted two
    # 60 mm tags on an A4 page, and CSP-1's props came out as ten sheets.
    cell = int(side * (1 + QUIET_FRACTION)) + mm_to_px(8, dpi)
    m = mm_to_px(MARGIN_MM, dpi)
    probe = blank_page(dpi)
    cols = max(1, (probe.shape[1] - 2 * m) // cell)
    rows = max(1, (probe.shape[0] - 2 * m - mm_to_px(20, dpi)) // cell)
    per_page = cols * rows

    tags = sorted(props.tags.values(), key=lambda t: t.tag_id)
    pages = []
    for p in range(0, len(tags), per_page):
        page = blank_page(dpi)
        put_text(page, f"PROP TAGS - {props.tag_size_m * 1000:.1f} mm squares, print at 100%",
                 m, m + mm_to_px(8, dpi), dpi, 1.2)
        for i, tag in enumerate(tags[p:p + per_page]):
            r, c = divmod(i, cols)
            x = m + c * cell + int(side * QUIET_FRACTION)
            y = m + mm_to_px(20, dpi) + r * cell + int(side * QUIET_FRACTION)
            page[y:y + side, x:x + side] = tag_image(tag.tag_id, props.tag_size_m, dpi)
            label = f"{tag.tag_id} {tag.entity}" + (f" [{tag.state}]" if tag.state else "")
            put_text(page, label, x, y + side + mm_to_px(6, dpi), dpi, 0.8)
        pages.append((f"props_{p // per_page + 1}", page))
    return pages


def guide_page(layout: TagLayout, props: PropMarkers | None, procedure, dpi: int
               ) -> tuple[str, np.ndarray]:
    """The rack face from the crew's side, to scale, with fiducials and zones."""
    import cv2
    page = blank_page(dpi)
    m = mm_to_px(MARGIN_MM, dpi)
    pts = np.vstack(list(layout.corners.values()))[:, :2]
    boxes = []
    if procedure is not None:
        for name, zone in procedure.zones.items():
            if zone.kind == "box":
                boxes.append((name, np.asarray(zone.lo)[:2], np.asarray(zone.hi)[:2]))
    extent = np.vstack([pts] + [np.vstack([lo, hi]) for _, lo, hi in boxes])
    lo, hi = extent.min(axis=0) - 0.05, extent.max(axis=0) + 0.05
    usable_w = page.shape[1] - 2 * m
    usable_h = page.shape[0] - 2 * m - mm_to_px(40, dpi)
    scale = min(usable_w / (hi[0] - lo[0]), usable_h / (hi[1] - lo[1]))

    def to_px(xy):
        return (int(m + (xy[0] - lo[0]) * scale),
                int(m + mm_to_px(40, dpi) + (hi[1] - xy[1]) * scale))

    title = "PLACEMENT GUIDE - rack face seen by the crew (+X right, +Y up)"
    put_text(page, title, m, m + mm_to_px(8, dpi), dpi, 1.1)
    put_text(page, "zones are drawn by their X-Y footprint; Z is out of the face toward the crew",
             m, m + mm_to_px(18, dpi), dpi, 0.8)
    for name, zlo, zhi in boxes:
        cv2.rectangle(page, to_px((zlo[0], zhi[1])), to_px((zhi[0], zlo[1])), 150,
                      max(1, dpi // 150))
        put_text(page, name, *to_px((zlo[0] + 0.01, zhi[1] - 0.03)), dpi, 0.6)
    for tag_id, corners in layout.corners.items():
        poly = np.array([to_px(c[:2]) for c in corners], dtype=np.int32)
        cv2.fillPoly(page, [poly], 0)
        cx, cy = to_px(corners.mean(axis=0)[:2])
        put_text(page, str(tag_id), cx + mm_to_px(4, dpi), cy - mm_to_px(4, dpi), dpi, 0.8)
    return "guide", page


def write_pdf(pages: list[tuple[str, np.ndarray]], path: Path, dpi: int) -> bool:
    """One PDF, each page placed at exact A4 millimetres. False without Qt."""
    try:
        from PySide6 import QtCore, QtGui
    except ImportError:
        return False
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])
    writer = QtGui.QPdfWriter(str(path))
    writer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.A4))
    writer.setPageMargins(QtCore.QMarginsF(0, 0, 0, 0))
    writer.setResolution(dpi)
    painter = QtGui.QPainter(writer)
    for i, (_name, page) in enumerate(pages):
        if i:
            writer.newPage()
        h, w = page.shape
        image = QtGui.QImage(page.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
        painter.drawImage(QtCore.QRectF(0, 0, w, h), image)
    painter.end()
    del app
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rack", type=Path, default=ROOT / "racks" / "msg_a_fiducials.json")
    ap.add_argument("--props", type=Path, default=None)
    ap.add_argument("--procedure", type=Path, default=None,
                    help="draw this procedure's zones on the placement guide")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "tags")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args(argv)

    import cv2

    layout = TagLayout.load(args.rack)
    props = PropMarkers.load(args.props) if args.props else None
    procedure = None
    if args.procedure:
        from parikshak.pdl import load_procedure
        procedure = load_procedure(args.procedure)

    pages = rack_pages(layout, args.dpi)
    if props is not None:
        pages += prop_pages(props, args.dpi)
    pages.append(guide_page(layout, props, procedure, args.dpi))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, page in pages:
        cv2.imwrite(str(args.out / f"{name}.png"), page)
    made_pdf = (not args.no_pdf) and write_pdf(pages, args.out / "tags.pdf", args.dpi)
    print(f"{len(pages)} page(s) -> {args.out}" + ("  (+ tags.pdf)" if made_pdf else ""))
    print("print at 100% / actual size, then measure a black square before use")
    return 0


if __name__ == "__main__":
    sys.exit(main())
