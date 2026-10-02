from pathlib import Path
import re
import xml.etree.ElementTree as ElementTree
import zipfile


WORKBOOK_PATH = Path(__file__).resolve().parents[1] / "reports" / "tables" / "蛋白质功能预测迭代趋势.xlsx"
NAMESPACES = {
    "chart": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "drawing": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "sheet": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
}


def _charts(archive):
    return [
        ElementTree.fromstring(archive.read(name))
        for name in archive.namelist()
        if "/charts/" in name and name.endswith(".xml") and "/_rels/" not in name
    ]


def test_iteration_trends_has_separate_range_backed_validation_charts():
    with zipfile.ZipFile(WORKBOOK_PATH) as archive:
        charts = _charts(archive)
    assert len(charts) == 3
    titles = {
        "".join(node.text or "" for node in chart.findall("./chart:chart/chart:title//drawing:t", NAMESPACES))
        for chart in charts
    }
    assert titles == {
        "500 标签尾段 Macro F1（目标 0.5）",
        "100 标签筛选 Macro F1（非全量成绩）",
        "历史随机验证 Macro F1（不同口径）",
    }
    for chart in charts:
        references = chart.findall(".//chart:f", NAMESPACES)
        assert references
        assert all("趋势图数据" in node.text for node in references)


def test_iteration_trend_helpers_link_all_measured_rows_to_source_cells():
    with zipfile.ZipFile(WORKBOOK_PATH) as archive:
        formulas = [
            node.text or ""
            for name in archive.namelist()
            if name.startswith("xl/worksheets/") and name.endswith(".xml")
            for node in ElementTree.fromstring(archive.read(name)).findall(".//sheet:f", NAMESPACES)
        ]
    linked_rows = {
        int(match.group(1))
        for formula in formulas
        for match in [re.fullmatch(r"'?迭代趋势'?!\$?E\$?(\d+)", formula)]
        if match
    }
    assert {6, 7, 8, 9, 10, 11, 35, 36, 37, 38, 39, 40}.issubset(linked_rows)


def test_full_label_chart_target_is_formula_backed_not_a_fake_experiment():
    with zipfile.ZipFile(WORKBOOK_PATH) as archive:
        chart = next(
            chart for chart in _charts(archive)
            if "目标 0.5" in "".join(node.text or "" for node in chart.findall("./chart:chart/chart:title//drawing:t", NAMESPACES))
        )
    series = chart.findall(".//chart:ser", NAMESPACES)
    assert len(series) == 2
    assert "目标" in "".join(node.text or "" for node in series[1].findall(".//chart:tx//chart:v", NAMESPACES))
    references = [node.text for node in series[1].findall(".//chart:f", NAMESPACES)]
    assert any("$C$" in reference for reference in references)
