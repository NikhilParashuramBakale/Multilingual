from __future__ import annotations
import csv, json
from pathlib import Path
from datetime import datetime, timezone

def build_report(recording: str, reference: str, results: list[dict], reviewed_code_switch: str = "", reviewed_clinical: str = "", language: str | None = None, code_switch_language: str | None = None, english_translation: dict | None = None) -> dict:
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "recording": recording, "reference": reference,
            "language": language, "code_switch_language": code_switch_language,
            "english_translation": english_translation,
            "reviewed_code_switch_segment": reviewed_code_switch, "reviewed_clinical_terms": reviewed_clinical, "results": results}

def write_reports(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True); stem = datetime.now().strftime("comparison_%Y%m%d_%H%M%S")
    json_path, csv_path = output_dir / f"{stem}.json", output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["recording", "language", "code_switch_language", "model", "device", "detected_language", "elapsed_seconds", "memory_usage_mb", "wer", "cer", "text"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in report["results"]:         writer.writerow({"recording": report["recording"], "language": report.get("language"),
                         "code_switch_language": report.get("code_switch_language"),
                         **{k:r.get(k) for k in fields if k not in {"recording", "language", "code_switch_language"}}})
    return json_path, csv_path

def aggregate_reports(reports: list[dict]) -> list[dict]:
    """Macro-average only recordings with a verified reference/metric."""
    grouped: dict[str, list[dict]] = {}
    for report in reports:
        for row in report.get("results", []): grouped.setdefault(row["model"], []).append(row)
    summary = []
    for model, rows in grouped.items():
        scored = [r for r in rows if r.get("wer") is not None]
        item = {"model": model, "recordings_run": len(rows), "recordings_scored": len(scored),
                "mean_wer": round(sum(r["wer"] for r in scored)/len(scored), 4) if scored else None,
                "mean_cer": round(sum(r["cer"] for r in scored)/len(scored), 4) if scored else None}
        if any("language" in r for r in rows):
            item["language"] = rows[0].get("language")
            item["mean_latency_seconds"] = round(sum(r.get("elapsed_seconds", 0) for r in rows) / len(rows), 4)
            item["mean_memory_usage_mb"] = round(sum(r.get("memory_usage_mb") or 0 for r in rows) / len(rows), 2)
        summary.append(item)
    return summary
