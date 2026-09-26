"""Build the final competition report as a verified-layout PDF."""

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "final_report.md"
DESTINATION = ROOT / "reports" / "蛋白质功能预测分析报告.pdf"
FONT_DIR = Path("C:/Windows/Fonts")


def register_fonts() -> tuple[str, str]:
    regular_candidates = [FONT_DIR / "simsun.ttc", FONT_DIR / "msyh.ttc"]
    bold_candidates = [FONT_DIR / "simhei.ttf", FONT_DIR / "msyhbd.ttc"]
    regular_path = next(path for path in regular_candidates if path.exists())
    bold_path = next(path for path in bold_candidates if path.exists())
    pdfmetrics.registerFont(TTFont("ReportChinese", str(regular_path), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("ReportChineseBold", str(bold_path), subfontIndex=0))
    return "ReportChinese", "ReportChineseBold"


REGULAR_FONT, BOLD_FONT = register_fonts()


def inline_markup(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font name="ReportChinese">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=10.5,
            leading=18,
            alignment=TA_LEFT,
            firstLineIndent=21,
            spaceAfter=5,
            textColor=colors.black,
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverSubtitle",
            parent=styles["Title"],
            fontName=BOLD_FONT,
            fontSize=16,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=10,
            textColor=colors.black,
        )
    )
    styles.add(
        ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName=BOLD_FONT,
            fontSize=22,
            leading=32,
            alignment=TA_CENTER,
            spaceAfter=22,
            textColor=colors.black,
        )
    )
    for name, size, leading, before, after in (
        ("Chapter", 16, 24, 4, 14),
        ("Section", 13, 20, 10, 8),
        ("Subsection", 11.5, 18, 8, 6),
    ):
        styles.add(
            ParagraphStyle(
                name,
                parent=styles["Heading1"],
                fontName=BOLD_FONT,
                fontSize=size,
                leading=leading,
                alignment=TA_LEFT,
                spaceBefore=before,
                spaceAfter=after,
                textColor=colors.black,
                keepWithNext=True,
            )
        )
    styles.add(
        ParagraphStyle(
            "Caption",
            parent=styles["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=9.5,
            leading=15,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            "TableCell",
            parent=styles["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=8.8,
            leading=13,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            "TableHeader",
            parent=styles["BodyText"],
            fontName=BOLD_FONT,
            fontSize=8.8,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            "Reference",
            parent=styles["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=8.8,
            leading=14,
            leftIndent=17,
            firstLineIndent=-17,
            spaceAfter=3,
            alignment=TA_LEFT,
        )
    )
    return styles


STYLES = make_styles()


def page_decor(canvas, document) -> None:
    canvas.saveState()
    page_number = canvas.getPageNumber()
    if page_number > 1:
        canvas.setFont(REGULAR_FONT, 8.5)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawCentredString(A4[0] / 2, A4[1] - 1.35 * cm, "蛋白质功能预测第三次实验报告")
        canvas.drawCentredString(A4[0] / 2, 1.25 * cm, str(page_number - 1))
    canvas.restoreState()


def cover_story(title: str) -> list:
    return [
        Spacer(1, 3.0 * cm),
        Paragraph("2026 年全国大学生海豚杯数智应用创新大赛", STYLES["CoverSubtitle"]),
        Spacer(1, 1.0 * cm),
        Paragraph(title, STYLES["ReportTitle"]),
        Spacer(1, 2.0 * cm),
        Table(
            [
                ["团队名称", "____________________________"],
                ["参赛高校", "____________________________"],
                ["参赛队员", "____________________________"],
                ["指导教师", "____________________________"],
            ],
            colWidths=[3.2 * cm, 9.0 * cm],
            style=TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), REGULAR_FONT),
                    ("FONTSIZE", (0, 0), (-1, -1), 12),
                    ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                    ("ALIGN", (1, 0), (1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                ]
            ),
        ),
        Spacer(1, 2.0 * cm),
        Paragraph("2026 年 9 月", STYLES["Caption"]),
        PageBreak(),
    ]


def parse_table(lines: list[str], start: int) -> tuple[list, int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            rows.append(cells)
        index += 1
    if not rows:
        return [], index
    columns = len(rows[0])
    available = 16.0 * cm
    if columns == 2:
        widths = [4.2 * cm, 11.8 * cm]
    elif columns == 3:
        widths = [4.1 * cm, 3.0 * cm, 8.9 * cm]
    elif columns == 4:
        widths = [4.3 * cm, 3.0 * cm, 3.6 * cm, 5.1 * cm]
    elif columns == 6:
        widths = [2.1 * cm, 2.1 * cm, 2.0 * cm, 3.7 * cm, 2.6 * cm, 3.5 * cm]
    else:
        widths = [available / columns] * columns
    formatted = []
    for row_index, row in enumerate(rows):
        style = STYLES["TableHeader"] if row_index == 0 else STYLES["TableCell"]
        formatted.append([Paragraph(inline_markup(value), style) for value in row])
    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#D9D9D9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return [table, Spacer(1, 9)], index


def build_story(lines: list[str]) -> list:
    title = lines[0].removeprefix("# ").strip()
    story = cover_story(title)
    chapter_seen = False
    figure_number = 0
    table_caption: Paragraph | None = None
    references = False
    index = 1
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("|"):
            table_elements, index = parse_table(lines, index)
            if table_caption is not None:
                story.append(KeepTogether([table_caption, *table_elements]))
                table_caption = None
            else:
                story.extend(table_elements)
            continue
        if stripped.startswith("## "):
            heading = stripped[3:].strip()
            references = heading == "参考文献"
            if chapter_seen:
                story.append(Spacer(1, 10))
            chapter_seen = True
            figure_number = 0
            story.append(Paragraph(inline_markup(heading), STYLES["Chapter"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(inline_markup(stripped[4:].strip()), STYLES["Section"]))
        elif stripped.startswith("#### "):
            story.append(Paragraph(inline_markup(stripped[5:].strip()), STYLES["Subsection"]))
        elif re.fullmatch(r"表\d+\.\d+\s+.+", stripped):
            table_caption = Paragraph(inline_markup(stripped), STYLES["Caption"])
        elif stripped.startswith("!["):
            match = re.fullmatch(r"!\[(.+)]\((.+)\)", stripped)
            if match:
                figure_number += 1
                image_path = ROOT / "reports" / match.group(2)
                image = Image(str(image_path), width=15.2 * cm, height=8.7 * cm)
                image._restrictSize(15.2 * cm, 10.0 * cm)
                caption = Paragraph(
                    inline_markup(f"图 {figure_number}  {match.group(1)}"),
                    STYLES["Caption"],
                )
                story.append(KeepTogether([image, caption]))
        elif references and re.match(r"^\[\d+]", stripped):
            story.append(Paragraph(inline_markup(stripped), STYLES["Reference"]))
        else:
            story.append(Paragraph(inline_markup(stripped), STYLES["ReportBody"]))
        index += 1
    return story


def main() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    document = SimpleDocTemplate(
        str(DESTINATION),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.3 * cm,
        bottomMargin=2.2 * cm,
        title=lines[0].removeprefix("# ").strip(),
        author="参赛团队",
        subject="蛋白质功能预测第三次实验",
    )
    document.build(build_story(lines), onFirstPage=page_decor, onLaterPages=page_decor)
    print(DESTINATION)


if __name__ == "__main__":
    main()
