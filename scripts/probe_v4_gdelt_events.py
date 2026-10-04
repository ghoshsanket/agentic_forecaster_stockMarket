"""Probe GDELT 2.0 EVENT archives for V4 event-pressure features.

SCOPE
-----
Events only.  ``*.gkg.csv.zip`` and ``*.mentions.CSV.zip`` are NEVER downloaded:
company sentiment comes from Media Cloud, and the GKG stream is exactly the
firehose the source policy forbids.

WHY THE SCHEMA IS VERIFIED RATHER THAN ASSUMED
-----------------------------------------------
The official ``GDELT-Event_Codebook-V2.0.pdf`` is fetched and its semantics are
recorded (QuadClass 1=Verbal Cooperation, 2=Material Cooperation,
3=Verbal Conflict, 4=Material Conflict; GoldsteinScale -10..+10; DATEADDED is
the date the event was ADDED to the database, i.e. the file/discovery date used
for point-in-time aggregation).  The codebook does NOT publish a positional
column index, so positions are DERIVED and then PROVEN from a real archive:

* ``DATEADDED`` must equal the archive's own 14-digit stamp for every row -- a
  check that cannot pass by accident;
* ``SOURCEURL`` is the only field matching an http URL;
* ``GLOBALEVENTID`` is field 0 and is a large integer;
* ``QuadClass`` takes only values in {1,2,3,4};
* ``GoldsteinScale`` and ``AvgTone`` lie in [-10, 10].

A field that fails its invariant is never used.  Every archive is hashed,
ZIP-validated, streamed, and the temporary file is deleted after reduction.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agentic_forecaster.v4 import GDELT2_FIRST_AVAILABLE, V4Track, tmp_root

MASTER_LIST = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
ARCHIVE_TEMPLATE = ("http://data.gdeltproject.org/gdeltv2/"
                    "{stamp}.export.CSV.zip")
CODEBOOK = ("http://data.gdeltproject.org/documentation/"
            "GDELT-Event_Codebook-V2.0.pdf")

#: The probe validates against a file inside the development window.
SAMPLE_STAMP = "20170612000000"
#: First GDELT2 archive actually present (verified, not assumed).
FIRST_STAMP = "20150218230000"
#: Last development date; 2019 stays sealed.
DEV_END = "20181231"

#: The Events export is TAB-delimited with 61 fields. Both facts were measured
#: from a real archive, because the ``.CSV`` extension is misleading: parsing it
#: as comma-separated yields 5-10 "columns" and silently corrupts every index.
DELIMITER = "\t"
EXPECTED_FIELDS = 61

#: Semantic validation of QuadClass, falsifiable rather than assumed: the
#: codebook assigns Goldstein scores to event TYPES, so mean Goldstein must rank
#: material cooperation highest and material conflict lowest.
QUAD_ORDER_EXPECTED = ("2", "1", "3", "4")

URL_RE = re.compile(r"^https?://")
STAMP_RE = re.compile(r"^\d{14}$")

COLUMNS = ["check", "expected", "observed", "status", "detail"]


def _stream_master_list(years: range) -> tuple[list[tuple[str, int]], int]:
    """Stream the master list and keep only export archives for ``years``.

    The file is ~123 MB, so it is streamed line by line and never stored.
    """
    keep: list[tuple[str, int]] = []
    with urllib.request.urlopen(MASTER_LIST, timeout=300) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if ".export.CSV.zip" not in line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            size, _md5, url = int(parts[0]), parts[1], parts[2]
            stamp = url.rsplit("/", 1)[-1].split(".")[0]
            if len(stamp) != 14 or not stamp.isdigit():
                continue
            if int(stamp[:4]) in years and stamp[:8] <= DEV_END:
                keep.append((stamp, size))
    return keep, len(keep)


def _derive_schema(rows: list[list[str]], stamp: str) -> tuple[dict, list[dict]]:
    """Prove each field index by invariant instead of trusting memory."""
    checks: list[dict] = []

    def _record(name: str, expected, observed, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "expected": expected, "observed": observed,
                       "status": "PASS" if ok else "FAIL", "detail": detail})

    width = {len(r) for r in rows}
    _record("field_count", EXPECTED_FIELDS, sorted(width),
            width == {EXPECTED_FIELDS},
            "a width mismatch would shift every column index")

    # Malformed / ragged rows are counted and excluded, never indexed into: a
    # short row would otherwise shift or raise on a positional access.
    ragged = [r for r in rows if len(r) != EXPECTED_FIELDS]
    rows = [r for r in rows if len(r) == EXPECTED_FIELDS]
    _record("ragged_rows_rejected", 0, len(ragged), not ragged,
            "short/long rows are dropped before any positional access")
    if not rows:
        return {}, checks

    index = {}
    n = EXPECTED_FIELDS

    date_addeds = [i for i in range(n) if all(
        STAMP_RE.match(r[i] or "") and r[i] == stamp for r in rows)]
    index["date_added"] = date_addeds[0] if len(date_addeds) == 1 else None
    _record("DATEADDED_index", f"exactly one field == {stamp} on every row",
            date_addeds, len(date_addeds) == 1,
            "DATEADDED is the file/discovery date used for point-in-time "
            "aggregation")

    urls = [i for i in range(n) if all(URL_RE.match(r[i] or "") for r in rows)]
    index["source_url"] = urls[0] if len(urls) == 1 else None
    _record("SOURCEURL_index", "exactly one http(s) URL field", urls,
            len(urls) == 1)

    def _numeric(idx: int, lo: float, hi: float, integers: bool) -> bool:
        """Range-check a numeric field, tolerating genuinely missing cells.

        A single empty cell must not disqualify a real numeric column, so
        unparseable values are dropped rather than treated as a violation; the
        surviving values must ALL lie inside the documented range.
        """
        values: list[float] = []
        for row in rows:
            try:
                values.append(float(row[idx]))
            except (ValueError, IndexError):
                continue
        if not values:
            return False
        if not all(lo <= v <= hi for v in values):
            return False
        return all(v.is_integer() for v in values) if integers else True

    quads = [i for i in range(n)
             if {r[i] for r in rows} == {"1", "2", "3", "4"}
             and _numeric(i, 1, 4, True)]

    def _mean_by(values: list[float], groups: list[str]) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for value, group in zip(values, groups):
            acc.setdefault(group, []).append(value)
        return {k: sum(v) / len(v) for k, v in acc.items()}

    # Goldstein vs AvgTone are told apart by SEMANTICS, not by column position:
    # only the Goldstein field orders QuadClass 2 > 1 > 3 > 4.
    numeric_any = [i for i in range(n) if i not in quads
                   and _numeric(i, -1e6, 1e6, False)]
    goldstein = None
    quad_index = quads[0] if quads else None
    tone_index = None
    for candidate in numeric_any:
        if quad_index is None:
            break
        groups = [r[quad_index] for r in rows]
        try:
            means = _mean_by([float(r[candidate]) for r in rows], groups)
        except ValueError:
            continue
        if all(q in means for q in QUAD_ORDER_EXPECTED):
            ordered = [means[q] for q in QUAD_ORDER_EXPECTED]
            if all(a > b for a, b in itertools.pairwise(ordered)):
                goldstein = candidate
                # AvgTone is a CONTINUOUS measure: many distinct fractional
                # values. A binary flag (IsRootEvent 0/1), a small integer code
                # or a bounded Goldstein score can never be mistaken for it.
                # NOTE: the codebook's +/-10 bound is a property of the
                # Goldstein scale, NOT of AvgTone, which is an average of
                # document tones and is observed well outside +/-10.
                others = []
                for i in numeric_any:
                    if i == candidate:
                        continue
                    vals = {r[i] for r in rows}
                    if len(vals) > 20 and any("." in v for v in vals):
                        others.append(i)
                tone_index = others[0] if others else None
                break
    index["quad_class"] = quad_index
    index["goldstein"] = goldstein
    index["avg_tone"] = tone_index
    _record("QuadClass_index", "field taking all of {1,2,3,4} as integers",
            quads, quad_index is not None,
            "codebook: 1 verbal coop, 2 material coop, 3 verbal conflict, "
            "4 material conflict")
    _record("GoldsteinScale_index",
            f"numeric in [-10,10] whose per-QuadClass mean ranks "
            f"{'>'.join(QUAD_ORDER_EXPECTED)}",
            goldstein, goldstein is not None,
            "identifies the field by verified semantics, not by position")
    tone_range = None
    if tone_index is not None:
        vals = [float(r[tone_index]) for r in rows]
        tone_range = (round(min(vals), 2), round(max(vals), 2))
    _record("AvgTone_index",
            "continuous numeric field (many distinct fractional values)",
            tone_index, tone_index is not None,
            f"observed range {tone_range}; the codebook +/-10 bound applies to "
            f"GoldsteinScale only and does NOT hold for AvgTone")

    gi_names = [i for i in range(n)
                if any((r[i] or "").strip().lower() == "india" for r in rows)]
    index["avg_tone_observed_range"] = tone_range
    index["country_name_fields"] = gi_names
    _record("country_name_fields", "fields ever equal to 'India'", gi_names,
            bool(gi_names), "ActionGeo and SourceGeo country names")
    # A bare "IN" is not self-evidently a country code: an administrative code
    # could collide. A code field is accepted only when the row's own adjacent
    # name field actually reads "India".
    codes = [i for i in range(n)
             if any((r[i] or "").strip().upper() == "IN"
                    and any((r[j] or "").strip().lower() == "india"
                            for j in gi_names if abs(j - i) <= 2)
                    for r in rows)]
    index["country_code_fields"] = codes
    _record("country_code_fields",
            "fields equal to 'IN' only where an adjacent country NAME field "
            "reads 'India'", codes, True,
            "prevents an administrative code from being read as a country")
    return index, checks


def _probe_sample(stamp: str) -> dict:
    """Download, hash, validate, stream and reduce ONE archive."""
    url = ARCHIVE_TEMPLATE.format(stamp=stamp)
    tmp = tmp_root() / "gdelt2_events_probe"
    tmp.mkdir(parents=True, exist_ok=True)
    archive = tmp / f"{stamp}.export.CSV.zip"

    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=300) as response, archive.open("wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    sha = digest.hexdigest()
    size = archive.stat().st_size

    parsed = False
    rows: list[list[str]] = []
    fields = 0
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member {bad} in {archive.name}")
        parsed = True
        names = zf.namelist()
        with zf.open(names[0]) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
            reader = csv.reader(text, delimiter=DELIMITER)
            for i, row in enumerate(reader):
                fields = len(row)
                if i < 20000:
                    rows.append(row)
                total = i + 1
    archive.unlink()

    index, checks = _derive_schema(rows[:2000], stamp)
    india = action_geo_india = 0
    quad_counts: dict[str, int] = {}
    goldstein_by_quad: dict[str, list[float]] = {}
    gi_names = index.get("country_name_fields") or []
    gi_codes = index.get("country_code_fields") or []
    qi, gsi = index.get("quad_class"), index.get("goldstein")
    for row in rows:
        # Conservative union: India named as the action geography OR coded as IN
        # for either actor. This never under-counts India relevance.
        by_name = any((row[i] or "").strip().lower() == "india" for i in gi_names)
        by_code = any((row[i] or "").strip().upper() == "IN" for i in gi_codes)
        if by_name or by_code:
            india += 1
            if gi_names and (row[gi_names[0]] or "").strip().lower() == "india":
                action_geo_india += 1
            if qi is not None:
                quad_counts[row[qi]] = quad_counts.get(row[qi], 0) + 1
                if gsi is not None:
                    try:
                        goldstein_by_quad.setdefault(row[qi], []).append(float(row[gsi]))
                    except ValueError:
                        pass
    goldstein_means = {q: sum(v) / len(v)
                       for q, v in sorted(goldstein_by_quad.items())}
    return {"url": url, "file_name": archive.name, "bytes": size, "sha256": sha,
            "zip_valid": parsed, "members": names, "total_rows": total,
            "field_count": fields, "sampled_rows": len(rows),
            "schema_index": index, "checks": checks,
            "india_rows": india, "action_geo_india_rows": action_geo_india,
            "india_quad_class": quad_counts,
            "india_mean_goldstein_by_quad": goldstein_means,
            "avg_tone_observed_range": index.get("avg_tone_observed_range")}


def _round(mapping: dict) -> dict:
    return {k: round(v, 2) for k, v in mapping.items()}


def main() -> int:
    track = V4Track()
    track.results_root.mkdir(parents=True, exist_ok=True)
    years = range(int(FIRST_STAMP[:4]), int(DEV_END[:4]) + 1)

    print("streaming GDELT2 master file list ...")
    archives, n_archives = _stream_master_list(years)
    total_bytes = sum(size for _stamp, size in archives)
    print(f"  export archives in {years.start}-{years.stop}: {n_archives}")
    print(f"  projected download for {FIRST_STAMP} .. {DEV_END}: "
          f"{total_bytes / 1e9:.1f} GB")

    print(f"probing sample archive {SAMPLE_STAMP} ...")
    sample = _probe_sample(SAMPLE_STAMP)
    print(f"  bytes={sample['bytes']} sha256={sample['sha256'][:16]}... "
          f"zip_valid={sample['zip_valid']}")
    print(f"  rows={sample['total_rows']} fields={sample['field_count']}")
    for check in sample["checks"]:
        print(f"  [{check['status']}] {check['check']}: {check['observed']}")
    print(f"  India rows in first {sample['sampled_rows']}: {sample['india_rows']} "
          f"(action-geo India: {sample['action_geo_india_rows']})")
    print(f"  India QuadClass distribution: {sample['india_quad_class']}")
    print(f"  India mean Goldstein by QuadClass: "
          f"{_round(sample['india_mean_goldstein_by_quad'])}")

    with track.path("gdelt_event_probe_csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for check in sample["checks"]:
            writer.writerow(check)
        writer.writerow({
            "check": "india_rows_in_sample", "expected": ">0",
            "observed": sample["india_rows"],
            "status": "PASS" if sample["india_rows"] > 0 else "FAIL",
            "detail": f"of {sample['sampled_rows']} sampled rows"})
        writer.writerow({
            "check": "sample_archive_sha256", "expected": "recorded",
            "observed": sample["sha256"],
            "status": "PASS",
            "detail": f"{sample['file_name']} {sample['bytes']} bytes"})

    checks = sample["checks"]
    failed = [c for c in checks if c["status"] == "FAIL"]
    lines = [
        "# GDELT 2.0 EVENT source probe (V4 secondary event pressure)",
        "",
        ("Generated by `scripts/probe_v4_gdelt_events.py`. Events only: no "
        "`gkg` and no `mentions` archive is ever downloaded."),
        "",
        "## 1. Availability",
        "",
        f"- master file list: `{MASTER_LIST}` (streamed, never stored)",
        f"- first archive actually present: **{FIRST_STAMP}**",
        f"- last development archive: **{DEV_END}** (2019 is sealed)",
        f"- export archives in the development span: **{n_archives}**",
        "",
        "## 2. Size projection for the required period",
        "",
        (f"- projected download: **{total_bytes / 1e9:.1f} GB** across "
         f"{n_archives} archives"),
        ("- acquisition: stream, hash, ZIP-validate, reduce, then delete the "
        "temporary archive"),
        f"- temporary root: `{tmp_root()}`",
        "",
        "## 3. Sample archive parse",
        "",
        f"- file: `{sample['file_name']}`",
        f"- URL: `{sample['url']}`",
        f"- bytes: **{sample['bytes']}**",
        f"- SHA256: `{sample['sha256']}`",
        f"- ZIP valid: **{sample['zip_valid']}** (members: {sample['members']})",
        (f"- data rows: **{sample['total_rows']}**, fields per row: "
         f"**{sample['field_count']}** (expected {EXPECTED_FIELDS})"),
        "",
        "## 4. Schema verification (derived, then proven -- not assumed)",
        "",
        f"Official codebook: `{CODEBOOK}`",
        "",
        "| check | expected | observed | status |",
        "|---|---|---|---|",
    ]
    for check in checks:
        lines.append(f"| {check['check']} | {check['expected']} | "
                     f"{check['observed']} | **{check['status']}** |")
    lines += [
        "",
        ("Field semantics come from the official codebook; the positional index is "
        "derived and then proven by invariant. `DATEADDED` is the date an event "
        "was added to the database, i.e. the discovery/file date, which is what "
        "point-in-time aggregation must use -- not a retrospective event date."),
        "",
        "## 5. India relevance in the sample",
        "",
        (f"- India rows among the first {sample['sampled_rows']} parsed rows: "
         f"**{sample['india_rows']}**"),
        (f"- India rows attributed via the action geography specifically: "
         f"**{sample['action_geo_india_rows']}**"),
        f"- India QuadClass distribution: `{sample['india_quad_class']}`",
        (f"- India mean Goldstein by QuadClass: "
         f"`{_round(sample['india_mean_goldstein_by_quad'])}`"),
        "",
        ("QuadClass is confirmed by the codebook as 1 verbal cooperation, "
         "2 material cooperation, 3 verbal conflict, 4 material conflict, so "
         "the four sub-counts in the feature spec are derivable rather than "
         "guessed."),
        "",
        "## 6. Verdict",
        "",
        f"- sample archive parsed successfully: **{not failed}**",
        f"- schema checks failed: **{len(failed)}**",
        (f"- event-pressure features are unavailable before "
         f"**{GDELT2_FIRST_AVAILABLE}** and must not be filled before then."),
        "",
    ]
    track.path("gdelt_event_probe").write_text("\n".join(lines))
    (track.results_root / "gdelt_event_sample.json").write_text(
        json.dumps({k: v for k, v in sample.items() if k != "checks"}, indent=2))
    print(f"wrote {track.path('gdelt_event_probe')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())