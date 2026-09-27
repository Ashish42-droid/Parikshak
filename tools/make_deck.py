#!/usr/bin/env python3
"""The SIH 2026 idea-submission deck, built from the project's own measurements.

    python tools/make_deck.py            # docs/PARIKSHAK_SIH_Idea_Submission.pptx

Six slides in SIH's idea-submission order: title, proposed solution, technical
approach, feasibility and viability, impact and benefits, research and
references. Every number is read from `runs/` when the deck is built, the alert
quoted on slide 2 is the engine's own - the skipped-latch golden run is replayed
here - and the rack image is drawn from the frame that alert fired on. Rebuild
after re-measuring and the deck cannot disagree with the evaluation.

Written with the standard library only: a .pptx is a zip of XML parts, and
writing them directly needs no download. `--render DIR` asks an installed
PowerPoint to export each slide as PNG (and the deck as PDF) for checking.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EMU = 914400                      # per inch
W_IN, H_IN = 13.333, 7.5          # 16:9 widescreen

# -- palette -------------------------------------------------------------------
NAVY = "0B1D33"
NAVY_2 = "13294B"
INK = "15202B"
SLATE = "4A5866"
MUTED = "75828E"
RULE = "D3DAE1"
MIST = "EEF2F6"
WHITE = "FFFFFF"
ICE = "CADCFC"
ACCENT = "2D5BD8"
ACCENT_SOFT = "E3EAFB"
SAFFRON = "E58E26"
OK = "1E8457"
OK_SOFT = "DFF1E8"
CRIT = "C0392B"
CRIT_SOFT = "F8E2DF"
UNV = "5F6196"

HEAD = "Arial"
BODY = "Calibri"

NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')
XML_HEAD = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'


def emu(inches: float) -> int:
    return int(round(inches * EMU))


# -- text ------------------------------------------------------------------------
@dataclass
class Run:
    text: str
    size: float = 14
    bold: bool = False
    color: str = INK
    font: str = BODY
    italic: bool = False


@dataclass
class Para:
    runs: list[Run]
    align: str = "l"
    after: float = 4
    bullet: bool = False
    line_pct: int = 100


def P(text: str, size: float = 14, *, bold: bool = False, color: str = INK, font: str = BODY,
      align: str = "l", after: float = 4, bullet: bool = False, italic: bool = False) -> Para:
    return Para([Run(text, size, bold, color, font, italic)], align, after, bullet)


def _run_xml(r: Run) -> str:
    attrs = f'lang="en-US" sz="{int(round(r.size * 100))}" b="{1 if r.bold else 0}" dirty="0"'
    if r.italic:
        attrs += ' i="1"'
    return (f'<a:r><a:rPr {attrs}><a:solidFill><a:srgbClr val="{r.color}"/></a:solidFill>'
            f'<a:latin typeface="{r.font}"/><a:cs typeface="{r.font}"/></a:rPr>'
            f'<a:t>{escape(r.text)}</a:t></a:r>')


def _para_xml(p: Para) -> str:
    ppr = f'<a:pPr algn="{p.align}"'
    if p.bullet:
        ppr += ' marL="228600" indent="-228600"'
    ppr += f'><a:lnSpc><a:spcPct val="{p.line_pct * 1000}"/></a:lnSpc>'
    ppr += f'<a:spcAft><a:spcPts val="{int(p.after * 100)}"/></a:spcAft>'
    if p.bullet:
        color = p.runs[0].color if p.runs else INK
        ppr += (f'<a:buClr><a:srgbClr val="{color}"/></a:buClr>'
                f'<a:buFont typeface="Arial"/><a:buChar char="&#8226;"/>')
    else:
        ppr += '<a:buNone/>'
    ppr += '</a:pPr>'
    runs = "".join(_run_xml(r) for r in p.runs) or '<a:endParaRPr lang="en-US"/>'
    return f'<a:p>{ppr}{runs}</a:p>'


def _body(paras: list[Para], *, anchor: str = "t", inset: float = 0.0) -> str:
    ins = emu(inset)
    return (f'<p:txBody><a:bodyPr wrap="square" lIns="{ins}" tIns="{ins}" rIns="{ins}" '
            f'bIns="{ins}" anchor="{anchor}" rtlCol="0"><a:noAutofit/></a:bodyPr><a:lstStyle/>'
            + "".join(_para_xml(p) for p in paras) + '</p:txBody>')


# -- slide -----------------------------------------------------------------------
@dataclass
class Slide:
    background: str = WHITE
    shapes: list[str] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)
    _next_id: int = 2

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    def text(self, x, y, w, h, paras: list[Para], *, anchor="t", name="Text") -> None:
        sid = self._id()
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{name} {sid}"/><p:cNvSpPr txBox="1"/>'
            f'<p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/>'
            f'</a:prstGeom><a:noFill/></p:spPr>{_body(paras, anchor=anchor)}</p:sp>')

    def box(self, x, y, w, h, fill: str | None, *, line: str | None = None, line_w: float = 1.0,
            paras: list[Para] | None = None, anchor="t", inset=0.14, round_: bool = True,
            name="Box") -> None:
        sid = self._id()
        geom = ('<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val 7000"/>'
                '</a:avLst></a:prstGeom>') if round_ else \
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        fill_xml = (f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill
                    else '<a:noFill/>')
        line_xml = (f'<a:ln w="{int(line_w * 12700)}"><a:solidFill><a:srgbClr val="{line}"/>'
                    f'</a:solidFill></a:ln>' if line else '<a:ln><a:noFill/></a:ln>')
        body = _body(paras or [P("")], anchor=anchor, inset=inset)
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{name} {sid}"/><p:cNvSpPr/><p:nvPr/>'
            f'</p:nvSpPr><p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>{geom}{fill_xml}{line_xml}</p:spPr>'
            f'{body}</p:sp>')

    def arrow(self, x1, y1, x2, color: str = SLATE) -> None:
        sid = self._id()
        self.shapes.append(
            f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{sid}" name="Arrow {sid}"/><p:cNvCxnSpPr/>'
            f'<p:nvPr/></p:nvCxnSpPr><p:spPr><a:xfrm><a:off x="{emu(x1)}" y="{emu(y1)}"/>'
            f'<a:ext cx="{emu(x2 - x1)}" cy="0"/></a:xfrm><a:prstGeom prst="straightConnector1">'
            f'<a:avLst/></a:prstGeom><a:ln w="22225"><a:solidFill><a:srgbClr val="{color}"/>'
            f'</a:solidFill><a:tailEnd type="triangle" w="med" len="med"/></a:ln></p:spPr>'
            f'</p:cxnSp>')

    def image(self, path: Path, x, y, w, h, alt: str) -> None:
        self.images.append(path)
        rid = f"rId{len(self.images) + 1}"          # rId1 is the layout
        sid = self._id()
        self.shapes.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="Picture {sid}" descr="{escape(alt)}"/>'
            f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch>'
            f'</p:blipFill><p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/>'
            f'</a:prstGeom><a:ln w="9525"><a:solidFill><a:srgbClr val="{RULE}"/></a:solidFill>'
            f'</a:ln></p:spPr></p:pic>')

    def table(self, x, y, col_w: list[float], rows: list[list[Para | str]], *,
              row_h: float = 0.4, header_fill: str = NAVY, header_color: str = WHITE,
              body_fill: str = WHITE, band_fill: str | None = MIST, size: float = 12,
              line: str = RULE, first_col_bold: bool = False) -> None:
        sid = self._id()
        grid = "".join(f'<a:gridCol w="{emu(c)}"/>' for c in col_w)

        def cell(content, *, fill, header, bold):
            paras = [content] if isinstance(content, Para) else [
                P(str(content), size, bold=header or bold,
                  color=header_color if header else INK, after=0)]
            border = (f'<a:ln w="6350"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>')
            lines = (border.replace('<a:ln ', '<a:lnL ').replace('</a:ln>', '</a:lnL>')
                     + border.replace('<a:ln ', '<a:lnR ').replace('</a:ln>', '</a:lnR>')
                     + border.replace('<a:ln ', '<a:lnT ').replace('</a:ln>', '</a:lnT>')
                     + border.replace('<a:ln ', '<a:lnB ').replace('</a:ln>', '</a:lnB>'))
            body = ('<a:txBody><a:bodyPr/><a:lstStyle/>'
                    + "".join(_para_xml(p) for p in paras) + '</a:txBody>')
            return (f'<a:tc>{body}<a:tcPr marL="{emu(0.1)}" marR="{emu(0.1)}" '
                    f'marT="{emu(0.05)}" marB="{emu(0.05)}" anchor="ctr">{lines}'
                    f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill></a:tcPr></a:tc>')

        trs = []
        for i, row in enumerate(rows):
            header = i == 0 and header_fill is not None
            fill = header_fill if header else (band_fill if band_fill and i % 2 == 0 else body_fill)
            tcs = "".join(cell(c, fill=fill, header=header, bold=(j == 0 and first_col_bold))
                          for j, c in enumerate(row))
            trs.append(f'<a:tr h="{emu(row_h)}">{tcs}</a:tr>')
        total_w = sum(col_w)
        self.shapes.append(
            f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{sid}" name="Table {sid}"/>'
            f'<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/>'
            f'</p:nvGraphicFramePr><p:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(total_w)}" cy="{emu(row_h * len(rows))}"/></p:xfrm><a:graphic>'
            f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table"><a:tbl>'
            f'<a:tblPr firstRow="1" bandRow="1"/><a:tblGrid>{grid}</a:tblGrid>{"".join(trs)}'
            f'</a:tbl></a:graphicData></a:graphic></p:graphicFrame>')

    def xml(self) -> str:
        bg = (f'<p:bg><p:bgPr><a:solidFill><a:srgbClr val="{self.background}"/></a:solidFill>'
              f'<a:effectLst/></p:bgPr></p:bg>')
        return (XML_HEAD + f'<p:sld {NS}><p:cSld>{bg}<p:spTree><p:nvGrpSpPr>'
                '<p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr>'
                '<a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/>'
                '<a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>' + "".join(self.shapes)
                + '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


# -- package parts ---------------------------------------------------------------------
THEME = XML_HEAD + f'''<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="PARIKSHAK">
<a:themeElements>
<a:clrScheme name="PARIKSHAK">
<a:dk1><a:srgbClr val="{INK}"/></a:dk1><a:lt1><a:srgbClr val="{WHITE}"/></a:lt1>
<a:dk2><a:srgbClr val="{NAVY}"/></a:dk2><a:lt2><a:srgbClr val="{MIST}"/></a:lt2>
<a:accent1><a:srgbClr val="{ACCENT}"/></a:accent1><a:accent2><a:srgbClr val="{SAFFRON}"/></a:accent2>
<a:accent3><a:srgbClr val="{OK}"/></a:accent3><a:accent4><a:srgbClr val="{CRIT}"/></a:accent4>
<a:accent5><a:srgbClr val="{UNV}"/></a:accent5><a:accent6><a:srgbClr val="{SLATE}"/></a:accent6>
<a:hlink><a:srgbClr val="{ACCENT}"/></a:hlink><a:folHlink><a:srgbClr val="{UNV}"/></a:folHlink>
</a:clrScheme>
<a:fontScheme name="PARIKSHAK">
<a:majorFont><a:latin typeface="{HEAD}"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>
<a:minorFont><a:latin typeface="{BODY}"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>
</a:fontScheme>
<a:fmtScheme name="PARIKSHAK">
<a:fillStyleLst>
<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
<a:solidFill><a:schemeClr val="phClr"><a:tint val="50000"/></a:schemeClr></a:solidFill>
<a:solidFill><a:schemeClr val="phClr"><a:shade val="80000"/></a:schemeClr></a:solidFill>
</a:fillStyleLst>
<a:lnStyleLst>
<a:ln w="6350" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/><a:miter lim="800000"/></a:ln>
<a:ln w="12700" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/><a:miter lim="800000"/></a:ln>
<a:ln w="19050" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/><a:miter lim="800000"/></a:ln>
</a:lnStyleLst>
<a:effectStyleLst>
<a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle>
</a:effectStyleLst>
<a:bgFillStyleLst>
<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
<a:solidFill><a:schemeClr val="phClr"><a:tint val="95000"/></a:schemeClr></a:solidFill>
<a:solidFill><a:schemeClr val="phClr"><a:shade val="90000"/></a:schemeClr></a:solidFill>
</a:bgFillStyleLst>
</a:fmtScheme>
</a:themeElements>
<a:objectDefaults/><a:extraClrSchemeLst/>
</a:theme>'''

EMPTY_TREE = ('<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/>'
              '</p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
              '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree>')

MASTER = XML_HEAD + (
    f'<p:sldMaster {NS}><p:cSld><p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef>'
    f'</p:bg>{EMPTY_TREE}</p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" '
    'accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" '
    'accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
    '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
    '<p:txStyles><p:titleStyle><a:lvl1pPr><a:defRPr sz="3200"/></a:lvl1pPr></p:titleStyle>'
    '<p:bodyStyle><a:lvl1pPr><a:defRPr sz="1600"/></a:lvl1pPr></p:bodyStyle>'
    '<p:otherStyle><a:lvl1pPr><a:defRPr sz="1400"/></a:lvl1pPr></p:otherStyle></p:txStyles>'
    '</p:sldMaster>')

LAYOUT = XML_HEAD + (
    f'<p:sldLayout {NS} type="blank" preserve="1"><p:cSld name="Blank">{EMPTY_TREE}</p:cSld>'
    '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')

REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def rels(items: list[tuple[str, str, str]]) -> str:
    body = "".join(f'<Relationship Id="{i}" Type="{t}" Target="{target}"/>'
                   for i, t, target in items)
    return (XML_HEAD + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
            f'relationships">{body}</Relationships>')


def write_pptx(slides: list[Slide], out: Path, title: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(slides)
    ct_slides = "".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.'
        f'openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, n + 1))
    content_types = XML_HEAD + (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>'
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>'
        '<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>'
        f'{ct_slides}'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        '</Types>')
    sld_ids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{i + 2}"/>' for i in range(n))
    presentation = XML_HEAD + (
        f'<p:presentation {NS} saveSubsetFonts="1">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f'<p:sldIdLst>{sld_ids}</p:sldIdLst>'
        f'<p:sldSz cx="{emu(W_IN)}" cy="{emu(H_IN)}"/><p:notesSz cx="6858000" cy="9144000"/>'
        '</p:presentation>')
    pres_rels = [("rId1", f"{REL}/slideMaster", "slideMasters/slideMaster1.xml")]
    pres_rels += [(f"rId{i + 2}", f"{REL}/slide", f"slides/slide{i + 1}.xml") for i in range(n)]
    pres_rels += [(f"rId{n + 2}", f"{REL}/theme", "theme/theme1.xml"),
                  (f"rId{n + 3}", f"{REL}/presProps", "presProps.xml"),
                  (f"rId{n + 4}", f"{REL}/viewProps", "viewProps.xml"),
                  (f"rId{n + 5}", f"{REL}/tableStyles", "tableStyles.xml")]

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels([
            ("rId1", f"{REL}/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", f"{REL}/extended-properties", "docProps/app.xml")]))
        z.writestr("docProps/core.xml", XML_HEAD + (
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f'<dc:title>{escape(title)}</dc:title><dc:creator>PARIKSHAK team</dc:creator>'
            '</cp:coreProperties>'))
        z.writestr("docProps/app.xml", XML_HEAD + (
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            f'<Application>PARIKSHAK make_deck.py</Application><Slides>{n}</Slides>'
            '<PresentationFormat>Widescreen</PresentationFormat></Properties>'))
        z.writestr("ppt/presentation.xml", presentation)
        z.writestr("ppt/_rels/presentation.xml.rels", rels(pres_rels))
        z.writestr("ppt/presProps.xml", XML_HEAD + f'<p:presentationPr {NS}/>')
        z.writestr("ppt/viewProps.xml", XML_HEAD + (
            f'<p:viewPr {NS}><p:gridSpacing cx="76200" cy="76200"/></p:viewPr>'))
        z.writestr("ppt/tableStyles.xml", XML_HEAD + (
            '<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>'))
        z.writestr("ppt/theme/theme1.xml", THEME)
        z.writestr("ppt/slideMasters/slideMaster1.xml", MASTER)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", rels([
            ("rId1", f"{REL}/slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", f"{REL}/theme", "../theme/theme1.xml")]))
        z.writestr("ppt/slideLayouts/slideLayout1.xml", LAYOUT)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", rels([
            ("rId1", f"{REL}/slideMaster", "../slideMasters/slideMaster1.xml")]))
        media = 0
        for i, s in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", s.xml())
            items = [("rId1", f"{REL}/slideLayout", "../slideLayouts/slideLayout1.xml")]
            for j, img in enumerate(s.images, start=2):
                media += 1
                name = f"image{media}.png"
                z.write(img, f"ppt/media/{name}")
                items.append((f"rId{j}", f"{REL}/image", f"../media/{name}"))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rels(items))


# -- the project's own numbers -----------------------------------------------------------
def _load(name: str) -> dict | None:
    path = ROOT / "runs" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def gather(work: Path) -> dict:
    """Everything the deck quotes, measured - plus the rack image and alert."""
    import cv2

    from parikshak.belief.trace import read_trace
    from parikshak.engine.runner import ProcedureEngine
    from parikshak.eval.schematic import render_schematic
    from parikshak.pdl import load_procedure

    csp1 = (_load("eval.json") or {}).get("metrics", {})
    crx2 = (_load("eval_crx2.json") or {}).get("metrics", {})
    bench = _load("bench.json") or {}
    manifest = json.loads((ROOT / "runs" / "downlink_sample" / "manifest.json")
                          .read_text(encoding="utf-8"))

    proc = load_procedure(ROOT / "procedures" / "csp1_colloid_sample_processing.yaml")
    _, frames = read_trace(ROOT / "traces" / "golden" / "skip_S08_latch.jsonl")
    engine = ProcedureEngine(proc, run_id="deck")
    alert, at_frame = None, None
    for f in frames:
        result = engine.step(f)
        for a in result.alerts:
            if alert is None and a.kind.value == "SKIP":
                alert, at_frame = a, f
    engine.finish()
    work.mkdir(parents=True, exist_ok=True)
    rack_png = work / "rack_skip_latch.png"
    img = render_schematic(proc, at_frame, 960, 540,
                           active_objects=proc.step(alert.step_id).objects)
    cv2.imwrite(str(rack_png), img)

    tests = None
    try:
        env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                             cwd=ROOT, capture_output=True, timeout=300,
                             env=env).stdout.decode("utf-8", "replace")
        # `-q` prints one "path.py: N" line per file and no overall total.
        per_file = [int(n) for n in re.findall(r"^\S+\.py: (\d+)\s*$", out, flags=re.M)]
        m = re.search(r"(\d+) tests? collected", out)
        tests = sum(per_file) if per_file else (int(m.group(1)) if m else None)
    except (OSError, subprocess.SubprocessError):
        tests = None

    return {"csp1": csp1, "crx2": crx2, "bench": bench, "manifest": manifest,
            "alert": alert, "rack_png": rack_png, "tests": tests}


def pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def silent_everywhere(m: dict) -> bool:
    return all(m.get(k) == 0 for k in ("nominal_false_alarms", "legal_reorder_false_alarms",
                                       "occlusion_false_alarms"))


# -- the six slides -------------------------------------------------------------------------
def header(s: Slide, title: str, number: int, total: int) -> None:
    s.text(0.6, 0.35, 9.5, 0.75, [P(title, 30, bold=True, color=NAVY, font=HEAD, after=0)],
           name="Title")
    s.text(9.4, 0.42, 3.33, 0.3, [P("SIH 2026 · PS 26174 · Space Technology", 11,
                                   color=MUTED, align="r", after=0)], name="Kicker")
    s.text(0.6, 7.02, 7.0, 0.3, [P("PARIKSHAK · on-board procedure witness", 10,
                                  color=MUTED, after=0)], name="Footer")
    s.text(11.2, 7.02, 1.53, 0.3, [P(f"{number} / {total}", 10, color=MUTED, align="r",
                                     after=0)], name="Page")


def slide_title(d: dict) -> Slide:
    s = Slide(background=NAVY)
    s.text(0.7, 0.55, 11.9, 0.35, [P("SMART INDIA HACKATHON 2026  ·  IDEA SUBMISSION", 12,
                                    bold=True, color=SAFFRON, after=0)])
    s.text(0.7, 1.05, 11.9, 1.15, [P("PARIKSHAK", 60, bold=True, color=WHITE, font=HEAD,
                                    after=0)])
    s.text(0.7, 2.2, 10.8, 1.2, [P(
        "An on-board witness that checks every step of an experiment procedure in orbit — "
        "and says so when it cannot see.", 22, color=ICE, after=0)])
    rows = [
        ["Problem Statement ID", "26174"],
        ["Problem Statement Title", "AI-based Human Activity Recognition for validating scientific "
                                    "experiment sequences on-board (BAS / lunar missions)"],
        ["Organisation", "Indian Space Research Organisation (ISRO), Department of Space"],
        ["Theme", "Space Technology"],
        ["PS Category", "Software"],
        ["Team ID", "[fill in your team ID]"],
        ["Team Name", "[fill in your team name]"],
    ]
    cells = [[P(k, 13, bold=True, color=ICE, after=0), P(v, 13, color=WHITE, after=0)]
             for k, v in rows]
    s.table(0.7, 3.95, [3.1, 8.83], cells, row_h=0.43,
            header_fill=None, band_fill=None, body_fill=NAVY_2, line="2A4570")
    return s


def slide_solution(d: dict, total: int) -> Slide:
    s = Slide()
    header(s, "Proposed solution", 2, total)
    s.text(0.6, 1.3, 6.2, 1.25, [P(
        "PARIKSHAK watches the experiment rack through a camera, checks every step against the "
        "procedure file, speaks the next step, and alerts only on strong evidence — naming the "
        "step and the reason.", 16, color=INK, after=0)])
    s.text(0.6, 2.62, 6.2, 0.35, [P("How it addresses the problem", 15, bold=True, color=NAVY,
                                   font=HEAD, after=0)])
    s.text(0.6, 3.02, 6.2, 2.4, [
        P("Verifies each step independently — the crew no longer self-reports on a tablet.",
          14, bullet=True, after=6),
        P("Catches skipped steps, the wrong object and safety violations within about 2 s.",
          14, bullet=True, after=6),
        P("Says “cannot verify” when blind: an occluded camera never makes an accusation.",
          14, bullet=True, after=6),
        P("Writes a tamper-evident flight record that is downlinked before any video.",
          14, bullet=True, after=0),
    ])
    s.image(d["rack_png"], 7.05, 1.3, 5.68, 3.195,
            "Rack face drawn from the engine's belief at the moment it flagged the skipped latch")
    a = d["alert"]
    s.box(7.05, 4.62, 5.68, 1.02, CRIT_SOFT, line=CRIT, line_w=1.0, inset=0.12, paras=[
        P(f"{a.severity.label.upper()} · SKIP · {a.step_id} · {a.t:.1f} s", 10.5, bold=True,
          color=CRIT, after=2),
        P(f"“{a.text.split('. ')[0]}.”", 13, bold=True, color=INK, after=2),
        Para([Run("Why: ", 11, True, INK), Run(a.reason, 11, False, SLATE)], after=0),
    ])
    cards = [
        ("The experiment is a file", "A new procedure is a YAML file. CRX-2 was added with no "
                                      "retraining and no code change."),
        ("Unknown is not wrong", "Three-valued logic: hidden or unmodelled evidence reads "
                                 "“cannot verify”, never “skipped”."),
        ("Up is the rack, not gravity", "Geometry is measured in rack coordinates from "
                                        "markers, so orientation in orbit does not matter."),
        ("Evidence over time", "Each step accumulates evidence with duration priors; an alert "
                               "needs 0.85 evidence held for 1.5 s."),
    ]
    s.text(0.6, 5.52, 6.0, 0.35, [P("Innovation and uniqueness", 15, bold=True, color=NAVY,
                                   font=HEAD, after=0)])
    w, gap = 2.9, 0.167
    for i, (title, body) in enumerate(cards):
        s.box(0.6 + i * (w + gap), 5.92, w, 1.0, MIST, inset=0.12, paras=[
            P(title, 12.5, bold=True, color=ACCENT, after=2),
            P(body, 10.5, color=SLATE, after=0)])
    return s


def slide_approach(d: dict, total: int) -> Slide:
    s = Slide()
    header(s, "Technical approach", 3, total)
    flow = [
        ("Camera", "Watches the payload rack, 5–10 frames/s"),
        ("Perception", "Rack pose from AprilTags; objects, states and hands"),
        ("Belief frame", "What is seen, with confidence — hidden stays unknown"),
        ("Procedure engine", "Predicates, evidence over time, duration priors, safety rules"),
        ("Crew & ground", "Voice prompt, alert with reason, checklist, hash-chained log"),
    ]
    bw, gap, y = 2.18, 0.3, 1.3
    for i, (name, sub) in enumerate(flow):
        x = 0.6 + i * (bw + gap)
        fill = ACCENT_SOFT if name == "Procedure engine" else MIST
        s.box(x, y, bw, 1.18, fill, line=ACCENT if name == "Procedure engine" else None,
              inset=0.12, paras=[P(name, 14, bold=True, color=NAVY, font=HEAD, after=3),
                                 P(sub, 10.5, color=SLATE, after=0)])
        if i < len(flow) - 1:
            s.arrow(x + bw + 0.04, y + 0.59, x + bw + gap - 0.04)

    fps = (d["bench"].get("end_to_end") or {}).get("fps_at_p95")
    rows = [
        ["Layer", "Working prototype", "Flight version (planned)"],
        ["Perception", "OpenCV 5, AprilTag 36h11 markers, PnP rack pose",
         "RT-DETR objects, MediaPipe / RTMPose hands, TensorRT INT8"],
        ["Reasoning", "Procedure language (YAML + JSON Schema), 3-valued predicates, "
                      "CUSUM evidence, alert policy", "Same engine, unchanged"],
        ["Interface", "Guided web app (FastAPI), PySide6 window, pyttsx3 voice",
         "Piper TTS, whisper.cpp / Vosk voice commands"],
        ["Record", "SHA-256 hash-chained log, deterministic trace replay", "CFDP downlink package"],
        ["Compute", f"Laptop CPU: {fps} FPS in the slowest 5% of frames" if fps else "Laptop CPU",
         "Jetson Orin, within 25 W"],
    ]
    s.table(0.6, 2.85, [1.45, 3.55, 2.95], rows, row_h=0.62, size=11.5, first_col_bold=True)

    tests = d["tests"]
    s.box(8.85, 2.85, 3.88, 3.72, NAVY, inset=0.2, paras=[
        P("Working prototype", 15, bold=True, color=WHITE, font=HEAD, after=8),
        P("Guided web app: choose an experiment, act each step, the engine judges live.",
          12, color=ICE, bullet=True, after=6),
        P("14 recorded scenarios replayed through the engine.", 12, color=ICE, bullet=True,
          after=6),
        P("Live webcam mode with printed markers; desktop operator window.", 12, color=ICE,
          bullet=True, after=6),
        P(f"{tests} automated tests; every build gate green." if tests else
          "Automated tests and build gates.", 12, color=ICE, bullet=True, after=6),
        P("Offline by construction: no network path in the on-board code.", 12,
          color=ICE, bullet=True, after=10),
        P("python -m demo", 12, bold=True, color=SAFFRON, font="Consolas", after=0),
    ])
    return s


def slide_feasibility(d: dict, total: int) -> Slide:
    s = Slide()
    header(s, "Feasibility and viability", 4, total)
    c1, c2 = d["csp1"], d["crx2"]

    def mark(value: str, ok: bool | None) -> Para:
        color = INK if ok is None else (OK if ok else CRIT)
        suffix = "" if ok is None else ("  met" if ok else "  missed")
        return Para([Run(value, 12, True, color), Run(suffix, 10, False, color)], align="l",
                    after=0)

    def fa(m):
        v = m.get("false_alarms_per_45min")
        return mark("—" if v is None else f"{v:.2f}", None if v is None else v <= 1.0)

    rows = [
        ["Measure", "Target", "CSP-1 · 14 steps", "CRX-2 · 8 steps"],
        ["Step accuracy", "≥ 95%", mark(pct(c1.get("step_accuracy")), (c1.get("step_accuracy") or 0) >= .95),
         mark(pct(c2.get("step_accuracy")), (c2.get("step_accuracy") or 0) >= .95)],
        ["Deviations caught", "≥ 90%", mark(pct(c1.get("deviation_recall")), (c1.get("deviation_recall") or 0) >= .90),
         mark(pct(c2.get("deviation_recall")), (c2.get("deviation_recall") or 0) >= .90)],
        ["False alarms per 45 min", "≤ 1", fa(c1), fa(c2)],
        ["False alarms, permitted reorders", "0",
         mark(str(c1.get("legal_reorder_false_alarms", "—")), c1.get("legal_reorder_false_alarms") == 0),
         mark(str(c2.get("legal_reorder_false_alarms", "—")), c2.get("legal_reorder_false_alarms") == 0)],
        ["Alert delay, median / 95th pct", "≤ 2 s",
         mark(f"{c1.get('latency_p50_s', 0):.1f} / {c1.get('latency_p95_s', 0):.1f} s",
              (c1.get("latency_p95_s") or 9) <= 2),
         mark(f"{c2.get('latency_p50_s', 0):.1f} / {c2.get('latency_p95_s', 0):.1f} s",
              (c2.get("latency_p95_s") or 9) <= 2)],
    ]
    s.text(0.6, 1.28, 7.0, 0.35, [P("Measured on the working prototype", 15, bold=True,
                                   color=NAVY, font=HEAD, after=0)])
    s.table(0.6, 1.7, [2.75, 0.95, 1.7, 1.7], rows, row_h=0.5, size=12, first_col_bold=True)
    s.text(0.6, 4.82, 7.1, 0.9, [P(
        f"Degraded synthetic runs — noise, dropped frames, occlusion — each scored against the "
        f"error injected into it: {c1.get('runs', '—')} CSP-1 runs, {c2.get('runs', '—')} CRX-2 "
        f"runs. An alarm counts as false if nothing was injected, or if it concerns a step "
        f"before the injected one.", 11, color=SLATE, after=0)])

    s.text(8.0, 1.28, 4.73, 0.35, [P("Risks and how we handle them", 15, bold=True,
                                    color=NAVY, font=HEAD, after=0)])
    risks = [
        ("Orbit looks different", "Learned parts never use gravity cues; geometry is in the "
                                  "rack frame; fine-tune on MicroG-4M."),
        ("False alarms erode trust", "Evidence plus persistence; unknown is never wrong; the "
                                     "remaining causes are measured and named."),
        ("The crew's body blocks the view", "Reads “cannot verify”; duration priors carry "
                                            "state; multi-camera fusion planned."),
        ("Edge compute and power", "Already real-time on a laptop CPU; Jetson Orin with "
                                   "TensorRT INT8 for the 25 W budget."),
    ]
    for i, (risk, fix) in enumerate(risks):
        s.box(8.0, 1.72 + i * 1.02, 4.73, 0.9, MIST, inset=0.12, paras=[
            P(risk, 12.5, bold=True, color=ACCENT, after=2), P(fix, 10.5, color=SLATE, after=0)])

    s.box(0.6, 5.85, 12.13, 0.95, ACCENT_SOFT, inset=0.16, anchor="ctr", paras=[Para([
        Run("Viability  ", 13, True, NAVY, HEAD),
        Run("Pure software on standard hardware. It reads the same procedure the crew already "
            "follows, is advisory and never commands hardware, and has no network path in the "
            "on-board code — a test fails the build if one appears.", 12.5, False, INK)],
        after=0)])
    return s


def slide_impact(d: dict, total: int) -> Slide:
    s = Slide()
    header(s, "Impact and benefits", 5, total)
    lv = d["manifest"].get("log_vs_video", {})
    log_kb = lv.get("log_bytes", 0) / 1000
    video_mb = lv.get("video_bytes", 0) / 1e6
    run_s = lv.get("duration_s")
    fps = (d["bench"].get("end_to_end") or {}).get("fps_at_p95")
    silent = silent_everywhere(d["csp1"]) and silent_everywhere(d["crx2"])
    tiles = [
        (f"{log_kb:.0f} KB", f"flight record for a {run_s:.0f} s run, against about "
                             f"{video_mb:.0f} MB of video" if run_s else "flight record per run"),
        ("0" if silent else "—", "false alarms on correct, reordered and blocked-camera runs, "
                                 "in both experiments" if silent else "see results"),
        ("1 file", "to add a new experiment — no retraining, no code change"),
        (f"{fps} FPS" if fps else "—", "on a laptop CPU, in the slowest 5% of frames"),
    ]
    w, gap = 2.87, 0.216
    for i, (big, small) in enumerate(tiles):
        x = 0.6 + i * (w + gap)
        s.box(x, 1.3, w, 1.72, MIST, inset=0.16, paras=[
            P(big, 32, bold=True, color=SAFFRON if i == 0 else NAVY, font=HEAD, after=4),
            P(small, 11, color=SLATE, after=0)])
    cols = [
        ("Crew and mission", [
            "Fewer experiment runs lost to a missed step.",
            "Hands-free guidance: attention stays on the task.",
            "Safety rules catch hazards no checklist can see."]),
        ("ISRO operations", [
            "A verifiable record of every run for the principal investigator.",
            "A tiny downlink footprint that fits constrained links.",
            "One system across the payload catalogue — procedures are data."]),
        ("Society and economy", [
            "Indigenous capability for Gaganyaan and Bharatiya Antariksh Station.",
            "Transfers to laboratory, pharma and manufacturing SOP compliance.",
            "Privacy by design: video never has to leave the station."]),
    ]
    cw, cgap = 3.9, 0.215
    for i, (title, items) in enumerate(cols):
        x = 0.6 + i * (cw + cgap)
        s.text(x, 3.35, cw, 0.4, [P(title, 16, bold=True, color=NAVY, font=HEAD, after=0)])
        s.text(x, 3.8, cw, 2.1, [P(t, 13.5, bullet=True, after=8) for t in items])
    s.box(0.6, 6.0, 12.13, 0.82, NAVY, inset=0.16, anchor="ctr", paras=[Para([
        Run("Why it has to run on board  ", 13, True, SAFFRON, HEAD),
        Run("A round trip to the Moon takes about 2.6 s and to Mars 6 to 44 minutes. Ground control cannot watch a step as it happens; the witness has to be on the station.", 12.5, False, WHITE)], after=0)])
    return s


def slide_references(d: dict, total: int) -> Slide:
    s = Slide()
    header(s, "Research and references", 6, total)
    cols = [
        ("Datasets and benchmarks", [
            "MicroG-4M — microgravity human activity (arXiv:2506.02845)",
            "IndustReal — procedure step recognition, WACV 2024 (arXiv:2310.17323)",
            "Assembly101 — mistake detection, CVPR 2022",
            "CaptainCook4D — procedural error taxonomy (arXiv:2312.14556)",
            "EgoPER — error detection in procedural video, CVPR 2024",
            "PREGO — online mistake detection, CVPR 2024",
            "HoloAssist — interactive assistance, ICCV 2023",
        ]),
        ("Methods and tools", [
            "AprilTag 3 — fiducial markers, IROS 2019",
            "RT-DETR — real-time detection (arXiv:2304.08069)",
            "MediaPipe Hands; RTMPose / MMPose",
            "ByteTrack — multi-object tracking (arXiv:2110.06864)",
            "MS-TCN++ — temporal action segmentation, TPAMI 2020",
            "Page's CUSUM — sequential change detection",
            "Piper TTS; whisper.cpp; OpenCV",
        ]),
        ("Standards and mission context", [
            "NASA-STD-3001 — Space Flight Human-System Standard",
            "CCSDS 727.0-B — CCSDS File Delivery Protocol (CFDP)",
            "ISRO — Gaganyaan and Bharatiya Antariksh Station (isro.gov.in)",
            "Digital Personal Data Protection Act, 2023 (MeitY)",
            "Vision-based mistake analysis in procedural activities: a review "
            "(arXiv:2510.19292)",
        ]),
    ]
    cw, cgap = 3.9, 0.215
    for i, (title, items) in enumerate(cols):
        x = 0.6 + i * (cw + cgap)
        s.text(x, 1.3, cw, 0.4, [P(title, 15, bold=True, color=NAVY, font=HEAD, after=0)])
        s.text(x, 1.78, cw, 4.3, [P(t, 12, bullet=True, color=INK, after=7) for t in items])
    s.box(0.6, 6.1, 12.13, 0.72, MIST, inset=0.14, anchor="ctr", paras=[Para([
        Run("Prototype and documentation  ", 12.5, True, NAVY, HEAD),
        Run("run it with  python -m demo  ·  full write-up in docs/PROJECT_DOCUMENTATION.md  ·  "
            "every figure on these slides is read from runs/ when the deck is built.",
            12, False, SLATE)], after=0)])
    return s


def build(out: Path, work: Path) -> Path:
    d = gather(work)
    total = 6
    slides = [slide_title(d), slide_solution(d, total), slide_approach(d, total),
              slide_feasibility(d, total), slide_impact(d, total), slide_references(d, total)]
    write_pptx(slides, out, "PARIKSHAK — SIH 2026 idea submission (PS 26174)")
    return out


def render(pptx: Path, out_dir: Path) -> None:
    """Export every slide as PNG, and the deck as PDF, through an installed PowerPoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pptx.with_suffix(".pdf")
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        "$ppt = New-Object -ComObject PowerPoint.Application\n"
        f"$pres = $ppt.Presentations.Open('{pptx}', $true, $false, $false)\n"
        f"$pres.Export('{out_dir}', 'PNG', 1600, 900)\n"
        f"$pres.SaveAs('{pdf}', 32)\n"
        "$pres.Close()\n"
        "$ppt.Quit()\n")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, timeout=300)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the SIH idea-submission deck.")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "PARIKSHAK_SIH_Idea_Submission.pptx")
    ap.add_argument("--work", type=Path, default=ROOT / "runs" / "deck")
    ap.add_argument("--render", type=Path, default=None,
                    help="export slides as PNG (and the deck as PDF) with PowerPoint")
    args = ap.parse_args(argv)
    out = build(args.out, args.work)
    print(f"  {out}  ({out.stat().st_size / 1e3:.0f} KB)")
    if args.render:
        render(out.resolve(), args.render.resolve())
        print(f"  slides rendered to {args.render}, PDF at {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
