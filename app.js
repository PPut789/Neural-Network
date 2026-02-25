const $ = (id) => document.getElementById(id);

/**
 * =====================================================================
 * GLOBALS
 * =====================================================================
 */
let rawData = null;          // numeric 2D (legacy: all cols incl target at last)
let rawRows = null;          // string 2D (keeps original values for encoding)
let Xraw = null, yraw = null;
let Xnorm = null, scaler = null, yScaler = null;
let nIn = null, model = null, headers = [];
let isTrained = false;

// NEW (Preprocess / Split / History / Eval / Viz)
let selectedFeatureIdx = [];
let selectedTargetIdx = null;
let Xtrain = null, Xtest = null, ytrain = null, ytest = null;
let trainSize = 0, testSize = 0;

let trainHistory = {
  loss: [],
  acc: [],
  valLoss: [],
  valAcc: []
};

let lastEval = {
  yTrue: [],
  yPred: [],
  task: "binary"
};

/**
 * =====================================================================
 * UTIL
 * =====================================================================
 */
function log(msg) {
  const box = $("log");
  if (!box) return;
  box.textContent += msg + "\n";
  box.scrollTop = box.scrollHeight;
}
function safeSetText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}
function safeSetHTML(id, html) {
  const el = $(id);
  if (el) el.innerHTML = html;
}
function clamp01(x) { return Math.max(0, Math.min(1, x)); }
function isNumericLike(v) {
  if (v === null || v === undefined) return false;
  const s = String(v).trim();
  if (s === "" || s.toUpperCase() === "NULL") return false;
  return !isNaN(Number(s));
}
function shuffleInPlace(arr, seed = 42) {
  // simple seeded shuffle (LCG)
  let m = 0x80000000, a = 1103515245, c = 12345;
  let state = (seed >>> 0) || 1;
  function rand() { state = (a * state + c) % m; return state / (m - 1); }

  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}
function mean(arr) {
  const xs = arr.filter(v => !Number.isNaN(v));
  if (xs.length === 0) return NaN;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
function median(arr) {
  const xs = arr.filter(v => !Number.isNaN(v)).slice().sort((a, b) => a - b);
  if (xs.length === 0) return NaN;
  const mid = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
}
function mode(arr) {
  const freq = new Map();
  for (const v of arr) {
    if (v === null || v === undefined) continue;
    const s = String(v).trim();
    if (s === "" || s.toUpperCase() === "NULL") continue;
    freq.set(s, (freq.get(s) || 0) + 1);
  }
  let best = null, bestC = -1;
  for (const [k, c] of freq.entries()) {
    if (c > bestC) { bestC = c; best = k; }
  }
  return best;
}

/**
 * =====================================================================
 * COLLAPSIBLE WEIGHT SECTION
 * =====================================================================
 */
const weightToggleBtn = $("weightToggleBtn");
if (weightToggleBtn) {
  weightToggleBtn.onclick = () => {
    const content = $("weightContentArea");
    const icon = $("weightIcon");
    if (!content || !icon) return;
    if (content.style.display === "none") {
      content.style.display = "block";
      icon.classList.add("open");
    } else {
      content.style.display = "none";
      icon.classList.remove("open");
    }
  };
}

/**
 * =====================================================================
 * DOM & Drag-Drop
 * =====================================================================
 */
const dropZone = $("dropZone");
if (dropZone) {
  dropZone.onclick = () => $("fileInput")?.click();
  dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add("hover"); };
  dropZone.ondragleave = () => dropZone.classList.remove("hover");
  dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.classList.remove("hover");
    handleFile(e.dataTransfer.files[0]);
  };
}
const fileInput = $("fileInput");
if (fileInput) {
  fileInput.onchange = (e) => handleFile(e.target.files[0]);
}

async function handleFile(file) {
  if (!file) return;
  const fileNameDisplay = $("fileNameDisplay");
  if (fileNameDisplay) fileNameDisplay.innerHTML = `Selected file : <b style="color:var(--accent)">${file.name}</b>`;

  const text = await file.text();
  try {
    // NEW: parse to rawRows (string) + rawData (numeric attempt)
    const parsed = parseCSV(text);
    rawRows = parsed.rawRows;
    rawData = parsed.numericRows;

    // default legacy: last column is target, others features
    processDataLegacy();       // keeps your old behavior working
    renderTable(rawRows);      // show original (string) first 50 rows
    checkMissingNumeric(rawData);

    buildPredictUI();
    populatePreprocessSelectors(); // NEW: fill target/features dropdowns and viz axis

    $("dataStat") && ($("dataStat").style.display = "flex");
    $("tableWrapper") && ($("tableWrapper").style.display = "block");

    isTrained = false;
    resetHistory();
    log(`โหลดไฟล์ ${file.name} สำเร็จ`);
  } catch (e) {
    alert(e.message);
  }
}

/**
 * =====================================================================
 * CSV PARSE (UPDATED: keep strings + numeric)
 * =====================================================================
 */
function parseCSV(text) {
  const rows = text.trim().split(/\r?\n/).map(r => r.trim()).filter(Boolean);
  let data = rows.map(r => r.split(",").map(s => s.trim()));

  if (data.length === 0) throw new Error("CSV ว่าง หรืออ่านไม่สำเร็จ");

  // header detect: if first row has any non-numeric and not NULL/empty
  const first = data[0];
  const isHeader = first.some(val => !isNumericLike(val) && val !== "" && val.toUpperCase() !== "NULL");
  if (isHeader) {
    headers = data.shift();
  } else {
    headers = Array.from({ length: data[0].length }, (_, i) =>
      i === data[0].length - 1 ? "Target (y)" : `Feature ${i + 1}`
    );
  }

  // keep raw string table
  const rawRows = data;

  // numeric conversion (NaN for NULL/empty/non-numeric)
  const numericRows = rawRows.map(row =>
    row.map(v => {
      const s = String(v).trim();
      if (s === "" || s.toUpperCase() === "NULL") return NaN;
      const num = Number(s);
      return Number.isFinite(num) ? num : NaN;
    })
  );

  // sanity: must have >= 2 columns
  if (headers.length < 2) throw new Error("CSV ต้องมีอย่างน้อย 2 คอลัมน์ (Feature + Target)");

  return { rawRows, numericRows };
}

/**
 * =====================================================================
 * LEGACY PROCESS (keeps your current flow: last col as y, min-max for X, yScaler fit only)
 * =====================================================================
 */
function processDataLegacy() {
  if (!rawData || rawData.length === 0) return;

  Xraw = rawData.map(r => r.slice(0, -1));
  yraw = rawData.map(r => r[r.length - 1]);

  nIn = Xraw[0].length;
  scaler = minMaxFit(Xraw);
  Xnorm = minMaxTransform(Xraw, scaler);

  // yScaler for regression inverse
  const yraw2d = yraw.map(v => [v]);
  yScaler = minMaxFit(yraw2d);

  safeSetText("metaRows", rawData.length);
  safeSetText("metaNIn", nIn);
}

/**
 * =====================================================================
 * MISSING / CLEAN
 * =====================================================================
 */
function checkMissingNumeric(data) {
  let count = 0;
  data.forEach(row => row.forEach(v => { if (isNaN(v)) count++; }));
  safeSetText("missingCount", count);
}

const btnCleanData = $("btnCleanData");
if (btnCleanData) {
  btnCleanData.onclick = () => {
    if (!rawData || !rawRows) return;
    const beforeCount = rawData.length;

    // remove rows with ANY NaN in numeric interpretation
    const keepIdx = [];
    rawData.forEach((row, i) => { if (row.every(v => !isNaN(v))) keepIdx.push(i); });

    if (keepIdx.length === 0) {
      alert("⚠️ ไม่สามารถ Clean Data ได้!\nเพราะข้อมูลของคุณมี Missing Value แทรกอยู่ในทุกแถว ถ้ากดลบตอนนี้ข้อมูลจะเหลือ 0 แถวครับ (ต้องแก้ที่ไฟล์ CSV ก่อน)");
      return;
    }

    rawData = keepIdx.map(i => rawData[i]);
    rawRows = keepIdx.map(i => rawRows[i]);

    processDataLegacy();
    renderTable(rawRows);
    checkMissingNumeric(rawData);

    isTrained = false;
    resetHistory();
    log(`Clean Data: ลบแถวที่มีค่าว่างออก ${beforeCount - rawData.length} แถว เรียบร้อยแล้ว ✅`);

    populatePreprocessSelectors();
    buildPredictUI();
  };
}

/**
 * Export Clean CSV (optional, if you added button)
 */
const btnExportClean = $("btnExportClean");
if (btnExportClean) {
  btnExportClean.onclick = () => {
    if (!rawRows) return alert("ยังไม่มีข้อมูลให้ export");
    const csv = [headers.join(",")]
      .concat(rawRows.map(r => r.map(v => (v === null || v === undefined) ? "" : String(v)).join(",")))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "cleaned.csv";
    a.click();
    URL.revokeObjectURL(a.href);
    log("Export Clean CSV สำเร็จ ✅");
  };
}

/**
 * =====================================================================
 * TABLE RENDER (show strings)
 * =====================================================================
 */
function renderTable(dataRowsString) {
  const table = $("dataTable");
  if (!table) return;

  table.innerHTML = "";

  const thead = document.createElement("thead");
  const trHead = document.createElement("tr");
  headers.forEach(h => {
    const th = document.createElement("th");
    th.textContent = h;
    trHead.appendChild(th);
  });
  thead.appendChild(trHead);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  dataRowsString.slice(0, 50).forEach(row => {
    const tr = document.createElement("tr");
    row.forEach(cell => {
      const td = document.createElement("td");
      const s = String(cell).trim();
      td.textContent = (s === "" || s.toUpperCase() === "NULL") ? "NULL" : s;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

/**
 * =====================================================================
 * PREDICT UI (based on nIn + headers[feature])
 * =====================================================================
 */
function buildPredictUI() {
  const container = $("dynamicPredictInputsContainer");
  if (!container) return;
  container.innerHTML = "";

  if (!nIn) return;

  const grid = document.createElement("div");
  grid.className = "predict-grid";

  for (let i = 0; i < nIn; i++) {
    const wrapper = document.createElement("div");
    wrapper.className = "predict-item";

    const label = document.createElement("label");
    label.textContent = headers[i] || `Feature ${i + 1}`;

    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.className = "dyn-predict-input";
    input.placeholder = "0.0";

    wrapper.appendChild(label);
    wrapper.appendChild(input);
    grid.appendChild(wrapper);
  }
  container.appendChild(grid);
}

/**
 * =====================================================================
 * NORMALIZATION (MinMax) + Inverse
 * =====================================================================
 */
function minMaxFit(X) {
  const n = X[0].length;
  const minv = Array(n).fill(+Infinity), maxv = Array(n).fill(-Infinity);
  for (const row of X) {
    for (let j = 0; j < n; j++) {
      minv[j] = Math.min(minv[j], row[j]);
      maxv[j] = Math.max(maxv[j], row[j]);
    }
  }
  return { minv, maxv };
}
function minMaxTransform(X, s) {
  return X.map(r =>
    r.map((v, j) => (s.maxv[j] === s.minv[j]) ? 0 : (v - s.minv[j]) / (s.maxv[j] - s.minv[j]))
  );
}
function standardFit(X) {
  const n = X[0].length;
  const mu = Array(n).fill(0), sd = Array(n).fill(0);

  for (let j = 0; j < n; j++) {
    const col = X.map(r => r[j]);
    mu[j] = mean(col);
    const m = mu[j];
    const v = mean(col.map(x => (x - m) ** 2));
    sd[j] = Math.sqrt(v);
    if (!Number.isFinite(sd[j]) || sd[j] === 0) sd[j] = 1;
  }
  return { mu, sd };
}
function standardTransform(X, s) {
  return X.map(r => r.map((v, j) => (v - s.mu[j]) / s.sd[j]));
}

// inverse for y (minmax only used for y in this project)
function inverseMinMaxTransform(val, scaler) {
  const mn = scaler.minv[0], mx = scaler.maxv[0];
  return (val * (mx - mn)) + mn;
}

/**
 * =====================================================================
 * PREPROCESSING (NEW)
 * - choose feature columns + target column
 * - missing strategy
 * - encoding (label / onehot; auto tries numeric)
 * - scaling (none/minmax/standard)
 * - split train/test with optional shuffle + seed
 * =====================================================================
 */
function populatePreprocessSelectors() {
  // target dropdown
  const targetSel = $("targetCol");
  const featSel = $("featureCols");
  const vizX = $("vizX");
  const vizY = $("vizY");

  if (targetSel) {
    targetSel.innerHTML = "";
    headers.forEach((h, idx) => {
      const opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = h;
      targetSel.appendChild(opt);
    });
    // default target = last column
    targetSel.value = String(headers.length - 1);
  }

  if (featSel) {
    featSel.innerHTML = "";
    headers.forEach((h, idx) => {
      const opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = h;
      // default select all except last
      if (idx !== headers.length - 1) opt.selected = true;
      featSel.appendChild(opt);
    });
  }

  // viz axis selectors
  const fillAxis = (sel) => {
    if (!sel) return;
    sel.innerHTML = "";
    const optAuto = document.createElement("option");
    optAuto.value = "auto"; optAuto.textContent = "Auto";
    sel.appendChild(optAuto);
    headers.forEach((h, idx) => {
      const opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = h;
      sel.appendChild(opt);
    });
  };
  fillAxis(vizX);
  fillAxis(vizY);

  // update summary placeholder
  const list = $("preprocessSummaryList");
  if (list) {
    list.innerHTML = `<li>เลือก Target (y), Features (X) แล้วกด <b>Run Preprocessing</b></li>`;
  }
}

// encoding helpers
function labelEncodeColumn(colStrings) {
  const map = new Map();
  let next = 0;
  const encoded = colStrings.map(s => {
    const key = String(s).trim();
    if (key === "" || key.toUpperCase() === "NULL") return NaN;
    if (!map.has(key)) map.set(key, next++);
    return map.get(key);
  });
  return { encoded, map };
}

function oneHotEncodeColumn(colStrings) {
  const uniques = [];
  const idxMap = new Map();
  for (const s0 of colStrings) {
    const s = String(s0).trim();
    if (s === "" || s.toUpperCase() === "NULL") continue;
    if (!idxMap.has(s)) { idxMap.set(s, uniques.length); uniques.push(s); }
  }
  const oh = colStrings.map(s0 => {
    const s = String(s0).trim();
    if (s === "" || s.toUpperCase() === "NULL") return Array(uniques.length).fill(NaN);
    const v = Array(uniques.length).fill(0);
    v[idxMap.get(s)] = 1;
    return v;
  });
  return { oh, uniques };
}

function runPreprocessing() {
  if (!rawRows || rawRows.length === 0) return alert("กรุณาโหลด CSV ก่อน");

  const targetSel = $("targetCol");
  const featSel = $("featureCols");
  if (!targetSel || !featSel) {
    // if your HTML doesn't include preprocess section, fallback
    processDataLegacy();
    return;
  }

  selectedTargetIdx = Number(targetSel.value);
  selectedFeatureIdx = Array.from(featSel.selectedOptions).map(o => Number(o.value));

  if (!Number.isFinite(selectedTargetIdx)) return alert("กรุณาเลือก Target Column");
  if (selectedFeatureIdx.length === 0) return alert("กรุณาเลือก Feature อย่างน้อย 1 คอลัมน์");

  // prevent target included in features
  selectedFeatureIdx = selectedFeatureIdx.filter(i => i !== selectedTargetIdx);
  if (selectedFeatureIdx.length === 0) return alert("Feature ต้องไม่ว่าง (อย่าเลือก Target เป็น Feature)");

  const missingStrategy = $("missingStrategy")?.value || "drop";
  const scaling = $("scaling")?.value || "none";
  const encoding = $("encoding")?.value || "auto";
  const testSizeVal = Number($("testSize")?.value ?? 0.2);
  const doShuffle = ($("shuffle")?.value ?? "true") === "true";
  const seed = Number($("seed")?.value ?? 42);

  // build columns (string)
  const Xcols = selectedFeatureIdx.map(j => rawRows.map(r => r[j]));
  const ycol = rawRows.map(r => r[selectedTargetIdx]);

  // detect column types + encode
  // We will output numeric matrix Xnum (rows x featuresExpanded) and yNum (rows)
  const nRows = rawRows.length;

  // handle y (target): allow numeric; if non-numeric and binary task, label encode it
  let yNum = ycol.map(v => {
    const s = String(v).trim();
    if (s === "" || s.toUpperCase() === "NULL") return NaN;
    return isNumericLike(s) ? Number(s) : NaN;
  });

  // if y still has NaN due to strings, label encode as fallback (common for class labels)
  if (yNum.some(v => Number.isNaN(v))) {
    const le = labelEncodeColumn(ycol);
    yNum = le.encoded;
    log("ℹ️ Target (y) มีค่าไม่เป็นตัวเลข → ทำ Label Encoding ให้แล้ว");
  }

  // build X numeric with encoding for each column
  let XnumRows = Array.from({ length: nRows }, () => []);

  for (let c = 0; c < Xcols.length; c++) {
    const col = Xcols[c];
    const numericCol = col.map(v => {
      const s = String(v).trim();
      if (s === "" || s.toUpperCase() === "NULL") return NaN;
      return isNumericLike(s) ? Number(s) : NaN;
    });
    const isNumericCol = !numericCol.some(v => Number.isNaN(v));

    if (encoding === "auto") {
      if (isNumericCol) {
        for (let i = 0; i < nRows; i++) XnumRows[i].push(numericCol[i]);
      } else {
        // one-hot by default in auto
        const oh = oneHotEncodeColumn(col);
        for (let i = 0; i < nRows; i++) XnumRows[i].push(...oh.oh[i]);
        log(`ℹ️ Feature "${headers[selectedFeatureIdx[c]]}" เป็น categorical → One-Hot (${oh.uniques.length})`);
      }
    } else if (encoding === "label") {
      if (isNumericCol) {
        for (let i = 0; i < nRows; i++) XnumRows[i].push(numericCol[i]);
      } else {
        const le = labelEncodeColumn(col);
        for (let i = 0; i < nRows; i++) XnumRows[i].push(le.encoded[i]);
        log(`ℹ️ Feature "${headers[selectedFeatureIdx[c]]}" → Label Encoding (${le.map.size})`);
      }
    } else if (encoding === "onehot") {
      if (isNumericCol) {
        for (let i = 0; i < nRows; i++) XnumRows[i].push(numericCol[i]);
      } else {
        const oh = oneHotEncodeColumn(col);
        for (let i = 0; i < nRows; i++) XnumRows[i].push(...oh.oh[i]);
        log(`ℹ️ Feature "${headers[selectedFeatureIdx[c]]}" → One-Hot (${oh.uniques.length})`);
      }
    }
  }

  // missing handling (drop/impute)
  // determine which rows are valid after strategy
  const rowsIdx = Array.from({ length: nRows }, (_, i) => i);

  if (missingStrategy === "drop") {
    const kept = rowsIdx.filter(i => {
      const xok = XnumRows[i].every(v => !Number.isNaN(v));
      const yok = !Number.isNaN(yNum[i]);
      return xok && yok;
    });
    const removed = nRows - kept.length;
    XnumRows = kept.map(i => XnumRows[i]);
    yNum = kept.map(i => yNum[i]);
    trainSize = 0; testSize = 0;
    log(`Preprocess: Drop missing rows → ลบ ${removed} แถว`);
  } else {
    // impute X numeric NaN (by column)
    const nFeat = XnumRows[0].length;
    for (let j = 0; j < nFeat; j++) {
      const col = XnumRows.map(r => r[j]);
      let fill = 0;
      if (missingStrategy === "mean") fill = mean(col);
      else if (missingStrategy === "median") fill = median(col);
      else if (missingStrategy === "zero") fill = 0;
      else if (missingStrategy === "mode") {
        // mode for numeric is less meaningful but ok
        const m = mode(col.map(v => Number.isNaN(v) ? "" : String(v)));
        fill = isNumericLike(m) ? Number(m) : 0;
      }
      if (!Number.isFinite(fill)) fill = 0;
      for (let i = 0; i < XnumRows.length; i++) {
        if (Number.isNaN(XnumRows[i][j])) XnumRows[i][j] = fill;
      }
    }

    // impute y NaN
    if (yNum.some(v => Number.isNaN(v))) {
      let fillY = 0;
      if (missingStrategy === "mean") fillY = mean(yNum);
      else if (missingStrategy === "median") fillY = median(yNum);
      else if (missingStrategy === "zero") fillY = 0;
      else if (missingStrategy === "mode") {
        const m = mode(yNum.map(v => Number.isNaN(v) ? "" : String(v)));
        fillY = isNumericLike(m) ? Number(m) : 0;
      }
      if (!Number.isFinite(fillY)) fillY = 0;
      yNum = yNum.map(v => Number.isNaN(v) ? fillY : v);
      log("Preprocess: Impute missing y สำเร็จ");
    }
  }

  // scaling (X)
  let Xscaled = XnumRows;
  if (scaling === "minmax") {
    scaler = minMaxFit(XnumRows);
    Xscaled = minMaxTransform(XnumRows, scaler);
  } else if (scaling === "standard") {
    scaler = standardFit(XnumRows);
    Xscaled = standardTransform(XnumRows, scaler);
  } else {
    scaler = null; // no scaling
  }

  // set globals for model (Xnorm / Xraw / yraw)
  Xraw = XnumRows;
  yraw = yNum;
  Xnorm = Xscaled;
  nIn = Xscaled[0].length;

  // fit yScaler for regression inverse (always minmax for y to keep your inverse logic)
  const y2d = yraw.map(v => [v]);
  yScaler = minMaxFit(y2d);

  // split train/test
  const idx = Array.from({ length: Xnorm.length }, (_, i) => i);
  if (doShuffle) shuffleInPlace(idx, seed);

  const tSize = clamp01(testSizeVal);
  const nTest = Math.max(1, Math.floor(idx.length * tSize));
  const nTrain = Math.max(1, idx.length - nTest);

  const testIdx = idx.slice(0, nTest);
  const trainIdx = idx.slice(nTest);

  Xtrain = trainIdx.map(i => Xnorm[i]);
  ytrain = trainIdx.map(i => yraw[i]);
  Xtest = testIdx.map(i => Xnorm[i]);
  ytest = testIdx.map(i => yraw[i]);

  trainSize = Xtrain.length;
  testSize = Xtest.length;

  // update meta
  safeSetText("metaRows", Xnorm.length);
  safeSetText("metaNIn", nIn);

  // rebuild predict UI now based on expanded features
  buildPredictUI();

  // update preprocess summary
  const sum = $("preprocessSummaryList");
  if (sum) {
    sum.innerHTML = "";
    const items = [
      `Target: <b>${headers[selectedTargetIdx]}</b>`,
      `Features: <b>${selectedFeatureIdx.map(i => headers[i]).join(", ")}</b>`,
      `Encoding: <b>${encoding}</b>`,
      `Missing: <b>${missingStrategy}</b>`,
      `Scaling: <b>${scaling}</b>`,
      `Split: <b>train ${trainSize}</b> / <b>test ${testSize}</b> (testSize=${tSize})`,
      `n_in (after encode): <b>${nIn}</b>`
    ];
    for (const it of items) {
      const li = document.createElement("li");
      li.innerHTML = it;
      sum.appendChild(li);
    }
  }

  isTrained = false;
  resetHistory();
  log("✅ Run Preprocessing สำเร็จ");
}

const btnRunPreprocess = $("btnRunPreprocess");
if (btnRunPreprocess) btnRunPreprocess.onclick = runPreprocessing;

const btnResetPreprocess = $("btnResetPreprocess");
if (btnResetPreprocess) {
  btnResetPreprocess.onclick = () => {
    // back to legacy behavior
    if (!rawData) return;
    processDataLegacy();
    buildPredictUI();
    isTrained = false;
    resetHistory();
    const sum = $("preprocessSummaryList");
    if (sum) sum.innerHTML = `<li>Reset แล้ว → ตอนนี้ใช้ <b>คอลัมน์สุดท้ายเป็น Target</b> และที่เหลือเป็น Feature</li>`;
    log("Reset Preprocessing แล้ว");
  };
}

/**
 * =====================================================================
 * MODEL BUILD
 * - uses Xnorm (preprocessed) if available
 * =====================================================================
 */
const btnBuild = $("btnBuild");
if (btnBuild) {
  btnBuild.onclick = () => {
    if (!Xnorm) return alert("กรุณาโหลด CSV และ/หรือ Run Preprocessing ก่อน");

    const task = $("task")?.value || "binary";
    const hidden = Number($("hidden")?.value ?? 8);
    const lr = Number($("lr")?.value ?? 0.01);
    let activation = $("activation")?.value || "tanh";
    if (activation === "leakyRelu") activation = "relu";

    // optional loss/metric UI
    const lossSel = $("lossSelect")?.value || "auto";
    const metricSel = $("metricSelect")?.value || "auto";

    if (model) model.dispose();
    model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [nIn], units: hidden, activation }));
    model.add(tf.layers.dense({ units: 1, activation: task === "binary" ? "sigmoid" : "linear" }));

    const loss = (lossSel !== "auto")
      ? lossSel
      : (task === "binary" ? "binaryCrossentropy" : "meanSquaredError");

    const metrics = [];
    if (task === "binary") {
      metrics.push(metricSel !== "auto" ? metricSel : "accuracy");
    } else {
      if (metricSel !== "auto") metrics.push(metricSel);
    }

    model.compile({
      optimizer: tf.train.adam(lr),
      loss,
      metrics
    });

    isTrained = false;
    resetHistory();
    log(`สร้างโมเดลแล้ว: Hidden=${hidden}, Act=${activation}, Task=${task}, Loss=${loss}`);

    safeSetText("modelSummaryText", `Dense(${hidden}, ${activation}) → Dense(1, ${task === "binary" ? "sigmoid" : "linear"})`);

    dumpWeightsToUI();
  };
}

/**
 * =====================================================================
 * WEIGHTS UI
 * =====================================================================
 */
async function dumpWeightsToUI() {
  if (!model) return;
  const [l1, l2] = model.layers;
  const [w1, b1] = l1.getWeights(), [w2, b2] = l2.getWeights();
  const w1Arr = await w1.array(), hidden = b1.shape[0];

  let w1Text = "";
  for (let h = 0; h < hidden; h++) {
    let row = [];
    for (let i = 0; i < nIn; i++) row.push(w1Arr[i][h].toFixed(4));
    w1Text += row.join(", ") + "\n";
  }
  $("w1_input") && ($("w1_input").value = w1Text.trim());
  $("b1_input") && ($("b1_input").value = (await b1.array()).map(v => v.toFixed(4)).join(", "));
  $("w2_input") && ($("w2_input").value = (await w2.array()).map(r => r[0].toFixed(4)).join(", "));
  $("b2_input") && ($("b2_input").value = (await b2.array())[0].toFixed(4));
}

const btnDumpWeights = $("btnDumpWeights");
if (btnDumpWeights) btnDumpWeights.onclick = dumpWeightsToUI;

const btnApplyWeights = $("btnApplyWeights");
if (btnApplyWeights) {
  btnApplyWeights.onclick = () => {
    try {
      const hidden = Number($("hidden")?.value ?? 8);
      const w1Lines = ($("w1_input")?.value ?? "").trim().split("\n");
      const b1Vals = ($("b1_input")?.value ?? "").split(",").map(Number);
      const w2Vals = ($("w2_input")?.value ?? "").split(",").map(Number);
      const b2Val = Number($("b2_input")?.value ?? 0);

      const w1_tf = Array.from({ length: nIn }, (_, i) =>
        Array.from({ length: hidden }, (_, h) => Number(w1Lines[h].split(",")[i]))
      );

      model.layers[0].setWeights([tf.tensor2d(w1_tf), tf.tensor1d(b1Vals)]);
      model.layers[1].setWeights([tf.tensor2d(w2Vals.map(v => [v])), tf.tensor1d([b2Val])]);

      isTrained = true;
      log("อัปเดต Weight เรียบร้อย ✅ (ถือว่า Trained แล้ว)");
    } catch (e) {
      alert("รูปแบบข้อมูลผิดพลาด กรุณาตรวจสอบจำนวน Hidden Units / จำนวน input (n_in)");
    }
  };
}

/**
 * =====================================================================
 * LOG RESET
 * =====================================================================
 */
const btnResetLog = $("btnResetLog");
if (btnResetLog) btnResetLog.onclick = () => { const l = $("log"); if (l) l.textContent = ""; };

function resetHistory() {
  trainHistory = { loss: [], acc: [], valLoss: [], valAcc: [] };
  safeSetText("trainLoss", "-");
  safeSetText("trainMetric", "-");
  safeSetText("valLoss", "-");
}

/**
 * =====================================================================
 * TRAIN
 * - uses train split if available; otherwise uses all rows
 * - stores history for visualization
 * =====================================================================
 */
const btnTrain = $("btnTrain");
if (btnTrain) {
  btnTrain.onclick = async () => {
    if (!model || !Xnorm) return alert("ต้องโหลดข้อมูลและสร้างโมเดลก่อน");

    const task = $("task")?.value || "binary";

    // prefer split data
    const Xuse = (Xtrain && ytrain) ? Xtrain : Xnorm;
    const yuse = (Xtrain && ytrain) ? ytrain : yraw;

    if (!Xuse || !yuse) return alert("ยังไม่มีข้อมูลสำหรับ Train");

    let ys;
    if (task === "binary") {
      const uniqueY = [...new Set(yuse)];
      const isNotBinary = uniqueY.some(y => y !== 0 && y !== 1);
      if (isNotBinary) {
        const proceed = confirm(
          "⚠️ คำเตือน: Binary Mode ต้องมี Target เป็น 0/1 เท่านั้น!\n" +
          "แต่พบค่าอื่น เช่น: " + uniqueY.slice(0, 5).join(", ") + "\n" +
          "แนะนำให้เปลี่ยน Task เป็น Regression"
        );
        if (!proceed) return;
      }
      ys = tf.tensor2d(yuse, [yuse.length, 1]);
    } else {
      // normalize y for regression training (minmax)
      const ynorm = minMaxTransform(yuse.map(v => [v]), yScaler).map(a => a[0]);
      ys = tf.tensor2d(ynorm, [ynorm.length, 1]);
    }

    const epochs = Number($("epochs")?.value ?? 200);
    const xs = tf.tensor2d(Xuse);

    log("⏳ กำลัง Train กรุณารอสักครู่...");

    // optional validation with test set (if exists)
    let valData = null;
    if (Xtest && ytest && Xtest.length > 0) {
      const xVal = tf.tensor2d(Xtest);
      let yVal;
      if (task === "binary") yVal = tf.tensor2d(ytest, [ytest.length, 1]);
      else {
        const yn = minMaxTransform(ytest.map(v => [v]), yScaler).map(a => a[0]);
        yVal = tf.tensor2d(yn, [yn.length, 1]);
      }
      valData = [xVal, yVal];
    }

    resetHistory();

    await model.fit(xs, ys, {
      epochs,
      validationData: valData,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          // store
          trainHistory.loss.push(logs.loss ?? null);
          if (typeof logs.acc === "number") trainHistory.acc.push(logs.acc);
          if (typeof logs.accuracy === "number") trainHistory.acc.push(logs.accuracy); // tfjs sometimes uses accuracy
          if (typeof logs.val_loss === "number") trainHistory.valLoss.push(logs.val_loss);
          if (typeof logs.val_acc === "number") trainHistory.valAcc.push(logs.val_acc);
          if (typeof logs.val_accuracy === "number") trainHistory.valAcc.push(logs.val_accuracy);

          // UI quick stats
          if ((epoch + 1) % 10 === 0) log(`Epoch ${epoch + 1}: loss = ${(logs.loss ?? 0).toFixed(6)}`);
          safeSetText("trainLoss", (logs.loss ?? 0).toFixed(6));

          const metric = (logs.acc ?? logs.accuracy);
          safeSetText("trainMetric", typeof metric === "number" ? metric.toFixed(6) : "-");

          const vloss = (logs.val_loss);
          safeSetText("valLoss", typeof vloss === "number" ? vloss.toFixed(6) : "-");
        }
      }
    });

    xs.dispose();
    ys.dispose();
    if (valData) { valData[0].dispose(); valData[1].dispose(); }

    isTrained = true;
    log("Train เสร็จแล้ว ✅");
    dumpWeightsToUI();
  };
}

/**
 * =====================================================================
 * PREDICT
 * =====================================================================
 */
const btnPredict = $("btnPredict");
if (btnPredict) {
  btnPredict.addEventListener("click", async () => {
    try {
      if (!model) return alert("ต้องสร้างโมเดลก่อน");
      if (!isTrained) return alert("⚠️ กรุณากดปุ่ม 'Start Training' หรือ 'Apply Weights' ก่อนทำการทำนายผลครับ!");

      const inputElements = document.querySelectorAll(".dyn-predict-input");
      if (!inputElements || inputElements.length !== nIn) {
        return alert("ช่อง Input ไม่ตรงกับ Model (n_in) — ลอง Run Preprocessing / โหลดใหม่");
      }

      const vals = Array.from(inputElements).map(el => {
        const v = el.value.trim();
        return v === "" ? NaN : Number(v);
      });

      if (vals.some(v => Number.isNaN(v))) return alert("กรุณากรอกตัวเลขให้ครบทุกช่อง");

      // IMPORTANT:
      // - If you used preprocessing with scaling, predict inputs must be scaled the same way.
      // - Here we only support:
      //   * minmax scaling: scaler has minv/maxv
      //   * standard scaling: scaler has mu/sd
      //   * none: scaler == null (assume user inputs already in model scale)
      let xTrans = vals;

      if (scaler && scaler.minv && scaler.maxv) {
        // minmax
        xTrans = vals.map((v, j) => (scaler.maxv[j] === scaler.minv[j]) ? 0 : (v - scaler.minv[j]) / (scaler.maxv[j] - scaler.minv[j]));
      } else if (scaler && scaler.mu && scaler.sd) {
        // standard
        xTrans = vals.map((v, j) => (v - scaler.mu[j]) / scaler.sd[j]);
      } else {
        // none
        xTrans = vals;
      }

      const xTensor = tf.tensor2d([xTrans], [1, nIn], "float32");
      const pred = model.predict(xTensor);
      let yhat = (await pred.data())[0];

      xTensor.dispose();
      pred.dispose();

      const task = $("task")?.value || "binary";
      let displayVal = "";

      if (task === "binary") {
        const cls = yhat >= 0.5 ? 1 : 0;

        if (yhat > 0 && yhat < 0.00001) displayVal = "< 0.00001";
        else if (yhat < 1 && yhat > 0.99999) displayVal = "> 0.99999";
        else displayVal = yhat.toFixed(5);

        safeSetHTML("result", `
          <div style="display:flex; align-items:center; justify-content:center; gap:10px;">
            <span class="muted">Prediction:</span>
            <b style="font-size:1.2rem;">${displayVal}</b>
            <span style="font-size:1.2rem;">→ <b>Class ${cls}</b></span>
          </div>
        `);

        safeSetText("predictDetailsText", `prob=${displayVal}, threshold=0.5, class=${cls}`);
        log(`Predict: yhat=${displayVal} | class=${cls}`);
      } else {
        const realWorldValue = inverseMinMaxTransform(yhat, yScaler);
        displayVal = realWorldValue.toFixed(4);

        safeSetHTML("result", `
          <div style="display:flex; align-items:center; justify-content:center; gap:10px;">
            <span class="muted">Prediction:</span>
            <b style="font-size:1.4rem; color:var(--accent);">${displayVal}</b>
          </div>
        `);

        safeSetText("predictDetailsText", `normalized=${yhat.toFixed(6)} → real=${displayVal}`);
        log(`Predict (Regression): Normalized=${yhat.toFixed(6)} | Real Value=${displayVal}`);
      }
    } catch (err) {
      alert(err.message || String(err));
    }
  });
}

/**
 * =====================================================================
 * EVALUATION (NEW)
 * - computes metrics on test set
 * - supports binary: acc/prec/recall/f1 + confusion matrix
 * - supports regression: mse/mae/r2
 * =====================================================================
 */
function confusionMatrixBinary(yTrue, yPred, threshold = 0.5) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const yt = yTrue[i];
    const yp = yPred[i] >= threshold ? 1 : 0;
    if (yt === 1 && yp === 1) tp++;
    else if (yt === 0 && yp === 0) tn++;
    else if (yt === 0 && yp === 1) fp++;
    else if (yt === 1 && yp === 0) fn++;
  }
  return { tp, tn, fp, fn };
}
function safeDiv(a, b) { return b === 0 ? 0 : a / b; }

async function evaluateOnTest() {
  if (!model) return alert("ต้องสร้างโมเดลก่อน");
  if (!Xtest || !ytest || Xtest.length === 0) return alert("ยังไม่มีชุด Test — กรุณา Run Preprocessing เพื่อ split train/test");

  const task = $("task")?.value || "binary";

  const xs = tf.tensor2d(Xtest);
  const predT = model.predict(xs);
  const yHatArr = Array.from(await predT.data());

  xs.dispose();
  predT.dispose();

  lastEval = { yTrue: ytest.slice(), yPred: yHatArr.slice(), task };

  // update eval table
  const evalTable = $("evalTable");
  if (evalTable) {
    evalTable.innerHTML = "";
    const thead = document.createElement("thead");
    thead.innerHTML = `<tr><th>#</th><th>y_true</th><th>y_pred</th></tr>`;
    evalTable.appendChild(thead);
    const tbody = document.createElement("tbody");
    for (let i = 0; i < Math.min(50, ytest.length); i++) {
      const tr = document.createElement("tr");
      const yt = ytest[i];
      const yp = yHatArr[i];
      tr.innerHTML = `<td>${i + 1}</td><td>${yt}</td><td>${task === "binary" ? yp.toFixed(6) : inverseMinMaxTransform(yp, yScaler).toFixed(6)}</td>`;
      tbody.appendChild(tr);
    }
    evalTable.appendChild(tbody);
  }

  if (task === "binary") {
    const cm = confusionMatrixBinary(ytest, yHatArr, 0.5);
    const acc = safeDiv(cm.tp + cm.tn, cm.tp + cm.tn + cm.fp + cm.fn);
    const prec = safeDiv(cm.tp, cm.tp + cm.fp);
    const rec = safeDiv(cm.tp, cm.tp + cm.fn);
    const f1 = safeDiv(2 * prec * rec, prec + rec);

    safeSetText("mAcc", acc.toFixed(6));
    safeSetText("mPrec", prec.toFixed(6));
    safeSetText("mRec", rec.toFixed(6));
    safeSetText("mF1", f1.toFixed(6));

    // clear regression metrics
    safeSetText("mMse", "-"); safeSetText("mMae", "-"); safeSetText("mR2", "-");

    log(`Evaluate(Binary): acc=${acc.toFixed(4)} prec=${prec.toFixed(4)} rec=${rec.toFixed(4)} f1=${f1.toFixed(4)} | CM tp=${cm.tp} tn=${cm.tn} fp=${cm.fp} fn=${cm.fn}`);
  } else {
    // regression: compare real-world y
    const yTrue = ytest;
    const yPredReal = yHatArr.map(v => inverseMinMaxTransform(v, yScaler));
    const yTrueMean = mean(yTrue);

    const mse = mean(yTrue.map((yt, i) => (yPredReal[i] - yt) ** 2));
    const mae = mean(yTrue.map((yt, i) => Math.abs(yPredReal[i] - yt)));
    const ssRes = yTrue.reduce((s, yt, i) => s + (yPredReal[i] - yt) ** 2, 0);
    const ssTot = yTrue.reduce((s, yt) => s + (yt - yTrueMean) ** 2, 0);
    const r2 = ssTot === 0 ? 0 : (1 - ssRes / ssTot);

    safeSetText("mMse", mse.toFixed(6));
    safeSetText("mMae", mae.toFixed(6));
    safeSetText("mR2", r2.toFixed(6));

    // clear binary metrics
    safeSetText("mAcc", "-"); safeSetText("mPrec", "-"); safeSetText("mRec", "-"); safeSetText("mF1", "-");

    log(`Evaluate(Regression): mse=${mse.toFixed(4)} mae=${mae.toFixed(4)} r2=${r2.toFixed(4)}`);
  }
}

const btnEvaluate = $("btnEvaluate");
if (btnEvaluate) btnEvaluate.onclick = evaluateOnTest;

const btnExportReport = $("btnExportReport");
if (btnExportReport) {
  btnExportReport.onclick = () => {
    if (!lastEval || lastEval.yTrue.length === 0) return alert("ยังไม่มีผล Evaluate ให้ export");
    const lines = [];
    lines.push("index,y_true,y_pred");
    for (let i = 0; i < lastEval.yTrue.length; i++) {
      const yt = lastEval.yTrue[i];
      const yp = lastEval.yPred[i];
      lines.push(`${i + 1},${yt},${yp}`);
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "evaluation_report.csv";
    a.click();
    URL.revokeObjectURL(a.href);
    log("Export Report สำเร็จ ✅");
  };
}

/**
 * =====================================================================
 * VISUALIZATION (NEW) - Canvas
 * - loss curve / acc curve
 * - scatter2d (if 2 features)
 * - decision boundary (binary + 2 features)
 * - confusion matrix (binary)
 * - pred vs true (regression)
 * =====================================================================
 */
function getCanvasCtx() {
  const c = $("vizCanvas");
  if (!c) return null;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  return { c, ctx };
}
function drawAxes(ctx, c, pad = 40) {
  ctx.save();
  ctx.font = "12px sans-serif";
  ctx.strokeStyle = "#999";
  ctx.lineWidth = 1;

  // x axis
  ctx.beginPath();
  ctx.moveTo(pad, c.height - pad);
  ctx.lineTo(c.width - pad, c.height - pad);
  ctx.stroke();

  // y axis
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, c.height - pad);
  ctx.stroke();

  ctx.restore();
}
function drawLineChart(values, label = "loss") {
  const pack = getCanvasCtx();
  if (!pack) return;
  const { c, ctx } = pack;
  const pad = 40;

  drawAxes(ctx, c, pad);

  const n = values.length;
  if (n === 0) {
    ctx.fillText("No data yet (train first).", pad, pad + 20);
    return;
  }

  const vMin = Math.min(...values.filter(v => typeof v === "number"));
  const vMax = Math.max(...values.filter(v => typeof v === "number"));
  const minV = Number.isFinite(vMin) ? vMin : 0;
  const maxV = Number.isFinite(vMax) ? vMax : 1;
  const span = (maxV - minV) || 1;

  ctx.save();
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 2;

  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad + (i / Math.max(1, n - 1)) * (c.width - 2 * pad);
    const y = (c.height - pad) - ((values[i] - minV) / span) * (c.height - 2 * pad);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = "#111";
  ctx.font = "14px sans-serif";
  ctx.fillText(`${label} (min=${minV.toFixed(4)}, max=${maxV.toFixed(4)})`, pad, pad - 10);

  ctx.restore();
}

function drawScatter2D() {
  const pack = getCanvasCtx();
  if (!pack) return;
  const { c, ctx } = pack;
  const pad = 40;

  if (!Xnorm || !yraw) {
    ctx.fillText("Upload + preprocess first.", pad, pad + 20);
    return;
  }
  if (nIn < 2) {
    ctx.fillText("Need at least 2 features for scatter.", pad, pad + 20);
    return;
  }

  drawAxes(ctx, c, pad);

  const xs = Xnorm.map(r => r[0]);
  const ys = Xnorm.map(r => r[1]);

  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xSpan = (xMax - xMin) || 1;
  const ySpan = (yMax - yMin) || 1;

  ctx.save();
  for (let i = 0; i < Xnorm.length; i++) {
    const px = pad + ((xs[i] - xMin) / xSpan) * (c.width - 2 * pad);
    const py = (c.height - pad) - ((ys[i] - yMin) / ySpan) * (c.height - 2 * pad);
    const cls = (yraw[i] >= 0.5) ? 1 : 0;
    ctx.fillStyle = cls ? "#1f77b4" : "#d62728";
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

async function drawDecisionBoundary(resolution = 150) {
  const pack = getCanvasCtx();
  if (!pack) return;
  const { c, ctx } = pack;
  const pad = 40;

  if (!model || !isTrained) {
    ctx.fillText("Train the model first.", pad, pad + 20);
    return;
  }
  if (!Xnorm || nIn < 2) {
    ctx.fillText("Need 2 features (n_in>=2).", pad, pad + 20);
    return;
  }
  const task = $("task")?.value || "binary";
  if (task !== "binary") {
    ctx.fillText("Decision boundary is for binary classification only.", pad, pad + 20);
    return;
  }

  const xs = Xnorm.map(r => r[0]);
  const ys = Xnorm.map(r => r[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);

  // draw background grid
  const img = ctx.createImageData(c.width, c.height);
  const w = c.width, h = c.height;

  // only paint inside plot area
  for (let py = pad; py < h - pad; py++) {
    for (let px = pad; px < w - pad; px++) {
      const x = xMin + ((px - pad) / (w - 2 * pad)) * (xMax - xMin);
      const y = yMin + (1 - (py - pad) / (h - 2 * pad)) * (yMax - yMin);
      // we only feed first 2 dims, remaining set 0
      const vec = Array(nIn).fill(0);
      vec[0] = x;
      vec[1] = y;

      const t = tf.tensor2d([vec], [1, nIn], "float32");
      const p = model.predict(t);
      const prob = (await p.data())[0];
      t.dispose(); p.dispose();

      // color blend: red(0) to blue(1)
      const r = Math.floor((1 - prob) * 255);
      const b = Math.floor(prob * 255);
      const g = 240;

      const idx = (py * w + px) * 4;
      img.data[idx + 0] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 50;
    }
  }
  ctx.putImageData(img, 0, 0);

  // overlay axes + points
  drawAxes(ctx, c, pad);
  ctx.save();
  for (let i = 0; i < Xnorm.length; i++) {
    const px = pad + ((xs[i] - xMin) / ((xMax - xMin) || 1)) * (c.width - 2 * pad);
    const py = (c.height - pad) - ((ys[i] - yMin) / ((yMax - yMin) || 1)) * (c.height - 2 * pad);
    const cls = (yraw[i] >= 0.5) ? 1 : 0;
    ctx.fillStyle = cls ? "#1f77b4" : "#d62728";
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawConfusionMatrix() {
  const pack = getCanvasCtx();
  if (!pack) return;
  const { c, ctx } = pack;
  const pad = 40;

  if (!lastEval || lastEval.task !== "binary" || lastEval.yTrue.length === 0) {
    ctx.fillText("Run Evaluate (binary) first.", pad, pad + 20);
    return;
  }
  const cm = confusionMatrixBinary(lastEval.yTrue, lastEval.yPred, 0.5);

  ctx.save();
  ctx.font = "16px sans-serif";
  ctx.fillStyle = "#111";
  ctx.fillText("Confusion Matrix (threshold=0.5)", pad, pad);

  // matrix box
  const boxSize = 240;
  const x0 = pad, y0 = pad + 30;

  ctx.strokeStyle = "#333";
  ctx.strokeRect(x0, y0, boxSize, boxSize);
  ctx.beginPath();
  ctx.moveTo(x0 + boxSize / 2, y0);
  ctx.lineTo(x0 + boxSize / 2, y0 + boxSize);
  ctx.moveTo(x0, y0 + boxSize / 2);
  ctx.lineTo(x0 + boxSize, y0 + boxSize / 2);
  ctx.stroke();

  ctx.font = "14px sans-serif";
  ctx.fillText("Pred 0", x0 + 30, y0 - 8);
  ctx.fillText("Pred 1", x0 + 150, y0 - 8);
  ctx.fillText("True 0", x0 - 50, y0 + 70);
  ctx.fillText("True 1", x0 - 50, y0 + 190);

  // TN, FP, FN, TP
  ctx.fillText(`TN: ${cm.tn}`, x0 + 30, y0 + 70);
  ctx.fillText(`FP: ${cm.fp}`, x0 + 150, y0 + 70);
  ctx.fillText(`FN: ${cm.fn}`, x0 + 30, y0 + 190);
  ctx.fillText(`TP: ${cm.tp}`, x0 + 150, y0 + 190);

  ctx.restore();
}

function drawPredVsTrue() {
  const pack = getCanvasCtx();
  if (!pack) return;
  const { c, ctx } = pack;
  const pad = 40;

  if (!lastEval || lastEval.task !== "regression" || lastEval.yTrue.length === 0) {
    ctx.fillText("Run Evaluate (regression) first.", pad, pad + 20);
    return;
  }

  drawAxes(ctx, c, pad);

  const yT = lastEval.yTrue;
  const yP = lastEval.yPred.map(v => inverseMinMaxTransform(v, yScaler));

  const xMin = Math.min(...yT), xMax = Math.max(...yT);
  const yMin = Math.min(...yP), yMax = Math.max(...yP);
  const minV = Math.min(xMin, yMin);
  const maxV = Math.max(xMax, yMax);
  const span = (maxV - minV) || 1;

  // diagonal y=x
  ctx.save();
  ctx.strokeStyle = "#333";
  ctx.beginPath();
  ctx.moveTo(pad, c.height - pad);
  ctx.lineTo(c.width - pad, pad);
  ctx.stroke();

  // points
  ctx.fillStyle = "#1f77b4";
  for (let i = 0; i < yT.length; i++) {
    const px = pad + ((yT[i] - minV) / span) * (c.width - 2 * pad);
    const py = (c.height - pad) - ((yP[i] - minV) / span) * (c.height - 2 * pad);
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#111";
  ctx.font = "14px sans-serif";
  ctx.fillText("Pred vs True (Regression)", pad, pad - 10);
  ctx.restore();
}

async function renderVisualization() {
  const type = $("vizType")?.value || "lossCurve";
  const res = Number($("vizResolution")?.value ?? 150);

  if (type === "lossCurve") drawLineChart(trainHistory.loss, "loss");
  else if (type === "accCurve") drawLineChart(trainHistory.acc, "accuracy");
  else if (type === "scatter2d") drawScatter2D();
  else if (type === "decisionBoundary") await drawDecisionBoundary(res);
  else if (type === "confusionMatrix") drawConfusionMatrix();
  else if (type === "predVsTrue") drawPredVsTrue();
}

const btnRenderViz = $("btnRenderViz");
if (btnRenderViz) btnRenderViz.onclick = renderVisualization;

const btnClearViz = $("btnClearViz");
if (btnClearViz) {
  btnClearViz.onclick = () => {
    const pack = getCanvasCtx();
    if (!pack) return;
    const { c, ctx } = pack;
    ctx.clearRect(0, 0, c.width, c.height);
  };
}

/**
 * =====================================================================
 * ALGORITHM TABS (NEW)
 * =====================================================================
 */
function showAlgo(tab) {
  const ids = ["algoOverview", "algoForward", "algoLoss", "algoBackprop", "algoUpdate"];
  ids.forEach(id => { const el = $(id); if (el) el.style.display = "none"; });
  const el = $(tab);
  if (el) el.style.display = "block";
}
const algoTabOverview = $("algoTabOverview");
const algoTabForward = $("algoTabForward");
const algoTabLoss = $("algoTabLoss");
const algoTabBackprop = $("algoTabBackprop");
const algoTabUpdate = $("algoTabUpdate");

if (algoTabOverview) algoTabOverview.onclick = () => showAlgo("algoOverview");
if (algoTabForward) algoTabForward.onclick = () => showAlgo("algoForward");
if (algoTabLoss) algoTabLoss.onclick = () => showAlgo("algoLoss");
if (algoTabBackprop) algoTabBackprop.onclick = () => showAlgo("algoBackprop");
if (algoTabUpdate) algoTabUpdate.onclick = () => showAlgo("algoUpdate");

/**
 * =====================================================================
 * AUTO INIT (if preprocess UI exists)
 * =====================================================================
 */
window.addEventListener("load", () => {
  // avoid errors if preprocess section exists but not populated yet
  if ($("preprocessSummaryList")) {
    $("preprocessSummaryList").innerHTML = `<li>Upload dataset แล้วกด <b>Run Preprocessing</b></li>`;
  }
});
// =====================
// PAGE NAV (Dataset → Preprocess → Build → Train → Evaluate → Predict)
// =====================
let currentPage = 0;

function showPage(pageIndex){
  const pages = document.querySelectorAll(".page");
  const tabs  = document.querySelectorAll(".tab");
  if(!pages.length) return;

  // clamp
  const maxPage = 5; // เราใช้ 0..5 ตาม flow
  currentPage = Math.max(0, Math.min(pageIndex, maxPage));

  // toggle pages
  pages.forEach(p => p.classList.remove("active"));
  tabs.forEach(t => t.classList.remove("active"));

  // show all sections that share same data-page (เช่น Build มีหลายการ์ด)
  pages.forEach(p => {
    if(Number(p.dataset.page) === currentPage) p.classList.add("active");
  });

  const activeTab = Array.from(tabs).find(t => Number(t.dataset.go) === currentPage);
  if(activeTab) activeTab.classList.add("active");

  window.scrollTo({ top: 0, behavior: "smooth" });
}

// tab click
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => showPage(Number(btn.dataset.go)));
});

// start at Dataset
window.addEventListener("load", () => showPage(0));