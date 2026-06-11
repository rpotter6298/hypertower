"""fetch_tcga_brca — programmatic download of TCGA-BRCA multimodal data.

Two-phase fetch:

  Phase 1 — TABULAR (cBioPortal bulk distribution):
    - Clinical fields (~90 columns: stage, grade, treatment, vital status, etc.)
    - RPPA protein expression (~200 proteins)
    - mRNA expression (~20,000 genes; optional)
    - Mutations (MAF)
    - All pre-joined by sample ID and cleaned by Broad/MSK curation
    - One tarball, ~200 MB compressed, fast download
    - Source: https://cbioportal-datahub.s3.amazonaws.com/
    - Curated study: brca_tcga_pan_can_atlas_2018

  Phase 2 — PATHOLOGY IMAGES (GDC API):
    - Diagnostic image thumbnails (small, JPG-like, ~MBs each — manageable)
    - Or full SVS slide images (gigapixel, ~100s of MB each — heavy)
    - Uses GDC's REST API to build a manifest, then downloads files
    - Source: https://api.gdc.cancer.gov/

Usage:
    python -m v4.scripts.data.fetch_tcga_brca --out data/tcga_brca
    python -m v4.scripts.data.fetch_tcga_brca --out data/tcga_brca --skip-images
    python -m v4.scripts.data.fetch_tcga_brca --out data/tcga_brca --images diagnostic --limit 50

All TCGA-BRCA data downloaded here is in GDC's *open-access* tier — no DUA,
no controlled-access approval needed.  Standard NIH attribution required for
publications.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CBIOPORTAL_STUDY = "brca_tcga_pan_can_atlas_2018"
# cBioPortal datahub stores files in a GitHub repo with LFS.  The S3 bucket
# is no longer publicly accessible, so we pull individual files from GitHub.
# Large data files are stored via LFS (different endpoint); small meta/text
# files are regular git blobs.  We try LFS first, fall back to raw.
CBIOPORTAL_LFS_BASE = (
    f"https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/{CBIOPORTAL_STUDY}"
)
CBIOPORTAL_RAW_BASE = (
    f"https://raw.githubusercontent.com/cBioPortal/datahub/master/public/{CBIOPORTAL_STUDY}"
)

# Curated file list for the BRCA Pan-Cancer Atlas 2018 study.
CBIOPORTAL_FILES_ESSENTIAL = [
    "data_clinical_patient.txt",   # ~90 clinical fields per patient
    "data_clinical_sample.txt",    # sample-level annotations
    "data_rppa.txt",               # RPPA protein expression (~200 proteins)
    "data_rppa_zscores.txt",       # RPPA z-scored against normal samples
    "meta_clinical_patient.txt",
    "meta_clinical_sample.txt",
    "meta_rppa.txt",
    "meta_study.txt",
]
CBIOPORTAL_FILES_OPTIONAL = [
    "data_protein_quantification.txt",          # mass-spec proteomics (CPTAC) — richer than RPPA
    "data_phosphoprotein_quantification.txt",   # phosphoproteomics
    "data_protein_quantification_zscores.txt",
    "data_mutations.txt",                       # MAF — somatic mutations
    "data_cna.txt",                             # copy-number alterations (gistic)
    "data_mrna_seq_v2_rsem.txt",                # RNA-seq counts (LARGE, ~150 MB)
    "data_mrna_seq_v2_rsem_zscores_ref_normal_samples.txt",
]

GDC_API_FILES    = "https://api.gdc.cancer.gov/files"
GDC_API_DATA     = "https://api.gdc.cancer.gov/data"

USER_AGENT = "hypertower-data-fetch/1.0 (research; python urllib)"


# ---------------------------------------------------------------------------
# Phase 1: cBioPortal tabular bundle
# ---------------------------------------------------------------------------

def fetch_cbioportal(out_dir: Path, include_optional: bool = False) -> Path:
    """Download cBioPortal TCGA-BRCA Pan-Cancer Atlas files via GitHub LFS."""
    study_dir = out_dir / "cbioportal" / CBIOPORTAL_STUDY
    study_dir.mkdir(parents=True, exist_ok=True)

    files = list(CBIOPORTAL_FILES_ESSENTIAL)
    if include_optional:
        files += CBIOPORTAL_FILES_OPTIONAL

    print(f"[cBioPortal] downloading {len(files)} files from datahub")
    print(f"  → {study_dir}")
    failed = []
    for fname in files:
        dest = study_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  · {fname}  (already present, {dest.stat().st_size/1e6:.2f} MB)")
            continue
        # Try LFS first (for large data files), then raw (for small meta files).
        last_err = None
        for url in (f"{CBIOPORTAL_LFS_BASE}/{fname}",
                    f"{CBIOPORTAL_RAW_BASE}/{fname}"):
            try:
                print(f"  ↓ {fname}")
                _stream_download(url, dest)
                print(f"     {dest.stat().st_size/1e6:.2f} MB")
                last_err = None
                break
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                last_err = e
        if last_err is not None:
            print(f"     failed: {last_err}")
            failed.append(fname)

    print(f"\n[cBioPortal] {len(files) - len(failed)}/{len(files)} files retrieved.")
    if failed:
        print(f"[cBioPortal] failed files: {failed}")

    print(f"\nFiles under {study_dir}:")
    for f in sorted(study_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name:60s}  {size_mb:>8.2f} MB")
    return study_dir


# ---------------------------------------------------------------------------
# Phase 2: GDC pathology images
# ---------------------------------------------------------------------------

# Image-type aliases for convenience. "Diagnostic Slide" is the larger SVS;
# "Tissue Slide" is similar.  Diagnostic image thumbnails are not always
# listed as a separate type — they're embedded inside the slide files.
_IMAGE_TYPE_FILTERS = {
    "diagnostic":     "Diagnostic Slide",
    "tissue":         "Tissue Slide",
}


def build_image_manifest(image_type: str = "diagnostic",
                         limit: int | None = None,
                         max_size_mb: float | None = None,
                         out_path: Path | None = None) -> list[dict]:
    """Query GDC API for TCGA-BRCA pathology images, return file metadata.

    Returns a list of dicts: file_id, file_name, file_size, patient_id, sample_id.
    """
    filt_type = _IMAGE_TYPE_FILTERS.get(image_type, image_type)
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id",
                                     "value": ["TCGA-BRCA"]}},
            {"op": "in", "content": {"field": "data_format",  "value": ["SVS"]}},
            {"op": "in", "content": {"field": "experimental_strategy",
                                     "value": [filt_type]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    # Request more than `limit` so we can filter by size client-side first.
    page_size = max(limit or 1000, 1000)
    params = {
        "filters": json.dumps(filters),
        "fields": ("file_id,file_name,file_size,experimental_strategy,"
                   "cases.submitter_id,cases.samples.submitter_id"),
        "format": "JSON",
        "size":   str(page_size),
    }
    url = f"{GDC_API_FILES}?{urllib.parse.urlencode(params)}"
    print(f"[GDC] querying for image manifest "
          f"(type={image_type}, max_size={max_size_mb}MB, limit={limit})...")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    raw_hits = data.get("data", {}).get("hits", [])
    total    = data.get("data", {}).get("pagination", {}).get("total", len(raw_hits))
    print(f"[GDC] GDC reports {total} total matching files; fetched {len(raw_hits)}")

    # Flatten + filter
    hits = []
    for h in raw_hits:
        case = (h.get("cases") or [{}])[0]
        sample = ((case.get("samples") or [{}])[0])
        size_mb = h.get("file_size", 0) / 1e6
        if max_size_mb is not None and size_mb > max_size_mb:
            continue
        hits.append({
            "file_id":               h["file_id"],
            "file_name":             h["file_name"],
            "file_size":             h.get("file_size", 0),
            "experimental_strategy": h.get("experimental_strategy"),
            "patient_id":            case.get("submitter_id"),
            "sample_id":             sample.get("submitter_id"),
        })
    if limit is not None:
        hits = hits[:limit]

    print(f"[GDC] {len(hits)} files in manifest after filter "
          f"({sum(h['file_size'] for h in hits)/1e9:.2f} GB total)")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(hits, indent=2))
        print(f"[GDC] manifest saved → {out_path}")

    return hits


def download_images(manifest: list[dict], out_dir: Path) -> None:
    """Download images from a GDC manifest. Files are SVS (gigapixel)."""
    import time
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(manifest)
    if n == 0:
        return
    total_bytes = sum(item.get("file_size", 0) for item in manifest)
    print(f"[GDC] downloading {n} files ({total_bytes/1e9:.2f} GB total) → {out_dir}")
    done_bytes = 0
    t0 = time.time()
    for i, item in enumerate(manifest, 1):
        fid    = item["file_id"]
        name   = item["file_name"]
        sz     = item.get("file_size", 0)
        dest   = out_dir / name
        if dest.exists() and dest.stat().st_size == sz:
            print(f"  [{i:>3d}/{n}] {name}  (already complete, skip)")
            done_bytes += sz
            continue
        elif dest.exists():
            dest.unlink()  # partial / wrong size, redo
        url = f"{GDC_API_DATA}/{fid}"
        print(f"  [{i:>3d}/{n}] {name}  ({sz/1e6:.1f} MB)  "
              f"[total so far {done_bytes/1e9:.2f}/{total_bytes/1e9:.2f} GB, "
              f"elapsed {(time.time()-t0)/60:.1f} min]")
        try:
            _stream_download(url, dest)
            done_bytes += sz
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"      failed: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stream_download(url: str, dest: Path, chunk_size: int = 1 << 16) -> None:
    """Stream-download a URL to a destination path, with a progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        got   = 0
        last_pct = -1
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total > 0:
                    pct = int(got * 100 / total)
                    if pct >= last_pct + 5:
                        print(f"    ... {pct}% ({got/1e6:.1f}/{total/1e6:.1f} MB)", flush=True)
                        last_pct = pct
    tmp.rename(dest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/tcga_brca"),
                    help="Output directory (default: data/tcga_brca)")
    ap.add_argument("--skip-tabular", action="store_true",
                    help="Skip the cBioPortal tabular file downloads")
    ap.add_argument("--include-optional", action="store_true",
                    help="Also fetch optional larger files (mutations, CNA, RNA-seq)")
    ap.add_argument("--skip-images",  action="store_true",
                    help="Skip the GDC image download (manifest only is still built)")
    ap.add_argument("--images", choices=["diagnostic", "tissue"], default="diagnostic",
                    help="Image type to fetch — diagnostic (H&E, ~1.5 GB each) or "
                         "tissue (~200 MB each). Default: diagnostic")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of images downloaded (after size filter)")
    ap.add_argument("--max-size-mb", type=float, default=None,
                    help="Skip files larger than this many MB (useful for sampling smaller slides)")
    ap.add_argument("--manifest-only", action="store_true",
                    help="Build the GDC image manifest JSON but don't download images")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.skip_tabular:
        fetch_cbioportal(args.out, include_optional=args.include_optional)

    if args.skip_images:
        return

    manifest = build_image_manifest(
        image_type=args.images,
        limit=args.limit,
        max_size_mb=args.max_size_mb,
        out_path=args.out / "images" / args.images / f"manifest.json",
    )

    if args.manifest_only:
        print("[GDC] manifest-only mode, skipping downloads.")
        return

    download_images(manifest, args.out / "images" / args.images)


if __name__ == "__main__":
    main()
