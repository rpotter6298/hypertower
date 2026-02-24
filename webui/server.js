const http = require("http");
const fs = require("fs");
const path = require("path");
const url = require("url");
const { spawnSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const WEBROOT = path.resolve(__dirname);

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload, null, 2);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

function sendFile(res, filePath) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    const mime =
      ext === ".html"
        ? "text/html"
        : ext === ".css"
        ? "text/css"
        : ext === ".js"
        ? "application/javascript"
        : "application/octet-stream";
    res.writeHead(200, { "Content-Type": mime });
    res.end(data);
  });
}

function listDir(requestedPath) {
  const cleaned = (requestedPath || "").replace(/^\/+/, "");
  const safePath = path.resolve(ROOT, cleaned || ".");
  if (!safePath.startsWith(ROOT)) {
    return { error: "Path outside repo root." };
  }
  if (!fs.existsSync(safePath)) {
    return { error: "Path not found." };
  }
  const stat = fs.statSync(safePath);
  if (!stat.isDirectory()) {
    return { error: "Path is not a directory." };
  }
  const entries = fs.readdirSync(safePath, { withFileTypes: true });
  const formatted = entries.map((entry) => {
    const entryPath = path.join(safePath, entry.name);
    const relPath = path.relative(ROOT, entryPath);
    const isDir = entry.isDirectory();
    const size = isDir ? 0 : fs.statSync(entryPath).size;
    return {
      name: entry.name,
      type: isDir ? "dir" : "file",
      relPath: relPath.replace(/\\/g, "/"),
      size,
    };
  });
  formatted.sort((a, b) => {
    if (a.type !== b.type) return a.type === "dir" ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
  return { path: path.relative(ROOT, safePath).replace(/\\/g, "/"), entries: formatted };
}

function readTabularFile(requestedPath, limit = 50) {
  const cleaned = (requestedPath || "").replace(/^\/+/, "");
  const safePath = path.resolve(ROOT, cleaned || ".");
  if (!safePath.startsWith(ROOT)) {
    return { error: "Path outside repo root." };
  }
  if (!fs.existsSync(safePath)) {
    return { error: "Path not found." };
  }
  const stat = fs.statSync(safePath);
  if (!stat.isFile()) {
    return { error: "Path is not a file." };
  }
  const ext = path.extname(safePath).toLowerCase();
  if ([".xlsx", ".xls"].includes(ext)) {
    const script = `
import json
import pandas as pd
import sys

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
df = pd.read_excel(path)
df = df.fillna("")
header = [str(c) for c in df.columns.tolist()]
total_rows = int(df.shape[0])
if limit > 0:
    df = df.head(limit)
rows = df.astype(str).values.tolist()
print(json.dumps({"header": header, "rows": rows, "rows_total": total_rows}))
`;
    const proc = spawnSync("python3", ["-c", script, safePath, String(limit || 0)], {
      encoding: "utf8",
    });
    if (proc.status !== 0) {
      const err = proc.stderr || proc.stdout || "python3 failed to read excel";
      return { error: err.trim() };
    }
    try {
      const parsed = JSON.parse(proc.stdout);
      return {
        path: path.relative(ROOT, safePath).replace(/\\/g, "/"),
        delimiter: "excel",
        header: parsed.header || [],
        rows: parsed.rows || [],
        rows_total: parsed.rows_total,
      };
    } catch (err) {
      return { error: "Failed to parse Excel preview output." };
    }
  }
  if (![".csv", ".tsv", ".txt"].includes(ext)) {
    return { error: "Only .csv, .tsv, .txt, .xlsx, or .xls are supported for preview." };
  }
  const content = fs.readFileSync(safePath, "utf8");
  const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (!lines.length) {
    return { error: "File is empty." };
  }
  const delimiter = ext === ".tsv" ? "\t" : ",";
  const header = lines[0].split(delimiter).map((v) => v.trim());
  const totalRows = lines.length - 1;
  const rowLimit = limit && limit > 0 ? limit : totalRows;
  const rows = lines
    .slice(1, rowLimit + 1)
    .map((line) => line.split(delimiter).map((v) => v.trim()));
  return {
    path: path.relative(ROOT, safePath).replace(/\\/g, "/"),
    delimiter,
    header,
    rows,
    rows_total: totalRows,
  };
}

function runPython(script, args = []) {
  const proc = spawnSync("python3", ["-c", script, ...args], {
    cwd: ROOT,
    encoding: "utf8",
    env: { ...process.env, PYTHONWARNINGS: "ignore" },
  });
  if (proc.status !== 0) {
    const err = (proc.stderr || proc.stdout || "python3 failed").trim();
    return { error: err || "python3 failed" };
  }
  try {
    return JSON.parse(proc.stdout);
  } catch (err) {
    return { error: "Failed to parse python output." };
  }
}

function importPapila() {
  const script = `
import json
from classes.v2 import PapilaData

pd = PapilaData.from_dirs(
    image_dir="Papila/FundusImages",
    clinical_dir="Papila/ClinicalData",
    label_col="Diagnosis",
    cat_cols=["Gender", "Phakic/Pseudophakic"],
)
out = {
    "image_dir": str(pd.clinical.image_dir),
    "clinical_dir": str(pd.clinical.clinical_dir) if pd.clinical.clinical_dir else None,
    "label_col": pd.label_col,
    "df_rows": int(pd.df.shape[0]),
    "df_cols": int(pd.df.shape[1]),
    "df_columns": list(pd.df.columns),
}
print(json.dumps(out))
`;
  return runPython(script);
}

function papilaDfPreview(limit = 50) {
  const script = `
import json
from classes.v2 import PapilaData

pd = PapilaData.from_dirs(
    image_dir="Papila/FundusImages",
    clinical_dir="Papila/ClinicalData",
    label_col="Diagnosis",
    cat_cols=["Gender", "Phakic/Pseudophakic"],
)
df = pd.df.fillna("")
header = [str(c) for c in df.columns.tolist()]
limit = int(${Number(limit)}) if ${Number(limit)} else 0
if limit > 0:
    df = df.head(limit)
rows = df.astype(str).values.tolist()
print(json.dumps({"header": header, "rows": rows, "rows_total": int(pd.df.shape[0])}))
`;
  return runPython(script);
}

const server = http.createServer((req, res) => {
  const parsed = url.parse(req.url, true);
  if (parsed.pathname === "/api/fs" || parsed.pathname === "/api/fs/") {
    const requestedPath = parsed.query.path || "";
    const payload = listDir(requestedPath);
    if (payload.error) {
      sendJson(res, 400, payload);
      return;
    }
    sendJson(res, 200, payload);
    return;
  }
  if (parsed.pathname === "/api/import/papila" || parsed.pathname === "/api/import/papila/") {
    const payload = importPapila();
    if (payload.error) {
      sendJson(res, 400, payload);
      return;
    }
    sendJson(res, 200, payload);
    return;
  }
  if (parsed.pathname === "/api/papila/df" || parsed.pathname === "/api/papila/df/") {
    const rawLimit = parsed.query.limit;
    const limit = rawLimit ? Number.parseInt(rawLimit, 10) : 50;
    const payload = papilaDfPreview(Number.isFinite(limit) ? limit : 50);
    if (payload.error) {
      sendJson(res, 400, payload);
      return;
    }
    sendJson(res, 200, payload);
    return;
  }
  if (parsed.pathname === "/api/file" || parsed.pathname === "/api/file/") {
    const requestedPath = parsed.query.path || "";
    const rawLimit = parsed.query.limit;
    const limit = rawLimit ? Number.parseInt(rawLimit, 10) : 50;
    const payload = readTabularFile(requestedPath, Number.isFinite(limit) ? limit : 50);
    if (payload.error) {
      sendJson(res, 400, payload);
      return;
    }
    sendJson(res, 200, payload);
    return;
  }

  let filePath = parsed.pathname === "/" ? "index.html" : parsed.pathname.slice(1);
  filePath = path.join(WEBROOT, filePath);
  if (!filePath.startsWith(WEBROOT)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }
  sendFile(res, filePath);
});

const port = process.env.PORT || 5173;
server.listen(port, () => {
  console.log(`Hypertower UI running at http://localhost:${port}`);
  console.log(`Repo root: ${ROOT}`);
});
