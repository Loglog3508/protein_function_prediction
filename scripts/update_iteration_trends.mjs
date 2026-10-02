import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [runtimeRoot, entriesPath, workbookPath, outputPath, previewPath] = process.argv.slice(2);
if (!runtimeRoot || !entriesPath || !workbookPath || !outputPath || !previewPath) {
  throw new Error('Usage: node update_iteration_trends.mjs runtime-root entries.json input.xlsx output.xlsx preview.png');
}
if (path.resolve(workbookPath) === path.resolve(outputPath)) {
  throw new Error('Export to a staging file and verify before replacing the workbook');
}
const require = createRequire(path.join(path.resolve(runtimeRoot), 'package.json'));
const { FileBlob, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')));
const entries = JSON.parse(await fs.readFile(entriesPath, 'utf8'));
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const sheet = workbook.worksheets.getItem('迭代趋势');
const original = sheet.getRange('A1:Q31').values;
const current = sheet.getUsedRange().values;
const existingNames = new Set(current.map(row => row[0]).filter(Boolean));
if (!existingNames.has('后续已完成迭代')) {
  sheet.getRange('A33').values = [['后续已完成迭代']];
  sheet.getRange('A33').format.font.bold = true;
  sheet.getRange('A34:H34').copyFrom(sheet.getRange('A5:H5'), 'all');
}
let nextRow = Math.max(35, current.length + 1);
for (const entry of entries) {
  if (existingNames.has(entry.iteration)) continue;
  if (entry.status !== 'complete' || typeof entry.f1 !== 'number' || typeof entry.auc !== 'number') {
    throw new Error(`Only completed, measured iterations can be appended: ${entry.iteration}`);
  }
  const source = JSON.parse(await fs.readFile(entry.source, 'utf8'));
  let result = source;
  for (const key of entry.sourceKeys) result = result[key];
  for (const [field, actual] of [['mean_crossfit_macro_f1', entry.f1], ['std_crossfit_macro_f1', entry.std], ['continuous_macro_auc', entry.auc]]) {
    if (!Number.isFinite(result[field]) || !Number.isFinite(actual) || Math.abs(result[field] - actual) > 1e-12) {
      throw new Error(`Source mismatch: ${entry.iteration} ${field}`);
    }
  }
  const destination = sheet.getRange(`A${nextRow}:H${nextRow}`);
  destination.copyFrom(sheet.getRange('A11:H11'), 'all');
  destination.values = [[entry.iteration, new Date(`${entry.date}T00:00:00Z`), entry.method, entry.validation, entry.f1, entry.std, entry.auc, entry.conclusion]];
  destination.format.wrapText = true;
  destination.format.rowHeight = 90;
  sheet.getRange(`B${nextRow}`).setNumberFormat('yyyy-mm-dd');
  sheet.getRange(`E${nextRow}:G${nextRow}`).setNumberFormat('0.000000');
  existingNames.add(entry.iteration);
  nextRow += 1;
}
if (JSON.stringify(sheet.getRange('A1:Q31').values) !== JSON.stringify(original)) {
  throw new Error('Historical cell values changed');
}
const latestDate = entries.map(entry => entry.date).sort().at(-1);
if (latestDate) {
  const currentNote = sheet.getRange('A3').values[0][0];
  sheet.getRange('A3').values = [[currentNote.replace(/截至 \d{4}-\d{2}-\d{2}/, `截至 ${latestDate}`)]];
}
let plotData;
try {
  plotData = workbook.worksheets.getItem('趋势图数据');
} catch {
  plotData = workbook.worksheets.add('趋势图数据');
}
plotData.getUsedRange().clear({applyTo: 'all'});
plotData.showGridLines = false;
plotData.getRange('A1').values = [['趋势图来源：公式链接登记表，分开显示不同验证口径']];
plotData.getRange('A2:B2').values = [['完整 500 标签目标 Macro F1', 0.5]];
plotData.getRange('A4:C4').values = [['迭代', 'Macro F1', '目标 0.5']];
plotData.getRange('E4:F4').values = [['筛选实验', 'Macro F1']];
plotData.getRange('H4:I4').values = [['历史随机验证', 'Macro F1']];
const registered = sheet.getUsedRange().values;
const fullRows = [8, 9, 10, 11];
const screenRows = [];
for (let row = 35; row <= registered.length; row += 1) {
  const values = registered[row - 1];
  if (!values[0]) continue;
  if (!Number.isFinite(values[4])) throw new Error(`Missing measured F1 at row ${row}`);
  const scope = String(values[3]);
  if (/\b500\b/.test(scope)) fullRows.push(row);
  else if (/\b100\b/.test(scope)) screenRows.push(row);
  else throw new Error(`Missing label-count scope for chart row ${row}`);
}
const populateSeries = (rows, categoryColumn, valueColumn, targetColumn) => {
  for (const [index, sourceRow] of rows.entries()) {
    const destinationRow = index + 5;
    plotData.getRange(`${categoryColumn}${destinationRow}`).formulas = [[`='迭代趋势'!A${sourceRow}`]];
    plotData.getRange(`${valueColumn}${destinationRow}`).formulas = [[`='迭代趋势'!E${sourceRow}`]];
    plotData.getRange(`${valueColumn}${destinationRow}`).setNumberFormat('0.000000');
    if (targetColumn) plotData.getRange(`${targetColumn}${destinationRow}`).formulas = [['=$B$2']];
  }
};
populateSeries(fullRows, 'A', 'B', 'C');
populateSeries(screenRows, 'E', 'F');
populateSeries([6, 7], 'H', 'I');
const noteRow = Math.max(fullRows.length, screenRows.length, 2) + 7;
plotData.getRange(`A${noteRow}`).values = [['尾段图只包含 500 标签结果；方法与校准协议仍会变化，不能当作线上成绩。']];
plotData.getRange(`A${noteRow + 1}`).values = [['100 标签筛选单列；随机验证历史单列，不与尾段成绩连接。']];
plotData.getRange('A1:I2').format.font.bold = true;
plotData.getRange('A:A').format.columnWidth = 22;
plotData.getRange('B:C').format.columnWidth = 16;
plotData.getRange('E:E').format.columnWidth = 27;
plotData.getRange('F:F').format.columnWidth = 16;
plotData.getRange('H:H').format.columnWidth = 22;
plotData.getRange('I:I').format.columnWidth = 16;
for (const range of ['A4:C4', 'E4:F4', 'H4:I4']) {
  plotData.getRange(range).format.font.bold = true;
  plotData.getRange(range).format.fill = '#FFF2CC';
}
sheet.charts.deleteAll();
const makeChart = (range, title, fromCell, toCell, color, target = false) => {
  const chart = sheet.charts.add('line', plotData.getRange(range));
  chart.title = title;
  chart.titleTextStyle.fontSize = 14;
  chart.titleTextStyle.typeface = 'Microsoft YaHei';
  chart.xAxis = {axisType: 'textAxis', textStyle: {typeface: 'Microsoft YaHei', fontSize: 11}};
  chart.yAxis = {numberFormatCode: '0.000', numberFormatSourceLinked: false, textStyle: {typeface: 'Microsoft YaHei'}};
  chart.hasLegend = target;
  if (target) chart.legend = {position: 'bottom', textStyle: {typeface: 'Microsoft YaHei', fontSize: 11}};
  chart.series.items[0].fill = color;
  chart.series.items[0].line = {fill: color, style: 'solid', width: 2};
  if (target) {
    chart.series.items[1].fill = '#C0504D';
    chart.series.items[1].line = {fill: '#C0504D', style: 'dashed', width: 2};
  }
  chart.setPosition(fromCell, toCell);
  return chart;
};
makeChart(`A4:C${fullRows.length + 4}`, '500 标签尾段 Macro F1（目标 0.5）', 'J2', 'Q16', '#1F4E78', true);
if (screenRows.length) makeChart(`E4:F${screenRows.length + 4}`, '100 标签筛选 Macro F1（非全量成绩）', 'J23', 'Q37', '#548235');
makeChart('H4:I6', '历史随机验证 Macro F1（不同口径）', 'J41', 'Q53', '#7F7F7F');
workbook.recalculate();
const errors = await workbook.inspect({kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?', options: {useRegex: true, maxResults: 30}, maxChars: 1500});
console.log(errors.ndjson);
console.log((await workbook.inspect({kind: 'table', range: `迭代趋势!A34:H${nextRow - 1}`, include: 'values,formulas', tableMaxRows: 20, tableMaxCols: 8, maxChars: 5500})).ndjson);
const preview = await workbook.render({sheetName: '迭代趋势', range: `A33:H${nextRow - 1}`, scale: 1, format: 'png'});
await fs.mkdir(path.dirname(path.resolve(outputPath)), {recursive: true});
await fs.mkdir(path.dirname(path.resolve(previewPath)), {recursive: true});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
for (const [suffix, range] of [['full-labels', 'J2:Q16'], ['screen', 'J23:Q37'], ['historical-random', 'J41:Q53']]) {
  const chartPreview = await workbook.render({sheetName: '迭代趋势', range, scale: 1.5, format: 'png'});
  const chartPreviewPath = path.join(path.dirname(previewPath), `${path.parse(previewPath).name}-${suffix}.png`);
  await fs.writeFile(chartPreviewPath, new Uint8Array(await chartPreview.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(workbook)).save(outputPath);
console.log(`Exported ${outputPath}; preserved historical records and updated measured trends and linked charts`);
