from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AfroGenRecord:
    file_name: str
    image_path: str
    prompt: str
    gender_presentation: str = "unknown"
    age_group: str = "adult"
    skin_tone: str = "medium"
    hairstyle: str = "curly"
    expression: str = "neutral"
    accessory: str = "none"
    lighting: str = "studio"
    background: str = "plain"
    split: str = "train"
    source: str = "local"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _default_prompt(row: dict) -> str:
    parts = [
        "A portrait of",
        row.get("age_group", "adult"),
        row.get("gender_presentation", "person"),
        "with",
        row.get("skin_tone", "medium"),
        "skin,",
        row.get("hairstyle", "curly"),
        "hair,",
        row.get("expression", "neutral"),
        "expression,",
        row.get("lighting", "studio"),
        "lighting,",
        row.get("background", "plain"),
        "background",
    ]
    accessory = row.get("accessory", "none")
    if accessory and accessory != "none":
        parts.extend(["and", accessory])
    return " ".join(parts)


def _read_metadata(metadata_path: Path) -> dict[str, dict]:
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: dict[str, dict] = {}
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            if not file_name:
                continue
            tags = row.get("tags", "").strip()
            row["tags"] = [tag.strip() for tag in tags.split("|") if tag.strip()]
            rows[file_name] = row
        return rows


def _infer_split(file_name: str) -> str:
    lowered = file_name.lower()
    if "val" in lowered or "valid" in lowered:
        return "validation"
    if "test" in lowered:
        return "test"
    return "train"


def build_manifest(
    project_root: Path,
    raw_dir: Path,
    processed_manifest: Path,
    allowed_extensions: tuple[str, ...],
) -> tuple[Path, list[AfroGenRecord]]:
    full_raw_dir = project_root / raw_dir
    manifest_path = project_root / processed_manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    full_raw_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = _read_metadata(full_raw_dir / "metadata.csv")
    image_paths = sorted(
        path for path in full_raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_extensions
    )

    records: list[AfroGenRecord] = []
    for path in image_paths:
        meta = metadata_rows.get(path.name, {})
        record_data = {
            "file_name": path.name,
            "image_path": str(path.relative_to(project_root)),
            "gender_presentation": meta.get("gender_presentation", "unknown") or "unknown",
            "age_group": meta.get("age_group", "adult") or "adult",
            "skin_tone": meta.get("skin_tone", "medium") or "medium",
            "hairstyle": meta.get("hairstyle", "curly") or "curly",
            "expression": meta.get("expression", "neutral") or "neutral",
            "accessory": meta.get("accessory", "none") or "none",
            "lighting": meta.get("lighting", "studio") or "studio",
            "background": meta.get("background", "plain") or "plain",
            "split": meta.get("split", _infer_split(path.name)) or "train",
            "source": meta.get("source", "local") or "local",
            "tags": meta.get("tags", []),
        }
        prompt = (meta.get("prompt") or "").strip() or _default_prompt(record_data)
        record = AfroGenRecord(prompt=prompt, **record_data)
        records.append(record)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")

    return manifest_path, records


def load_manifest(manifest_path: Path) -> list[AfroGenRecord]:
    if not manifest_path.exists():
        return []
    records: list[AfroGenRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(AfroGenRecord(**json.loads(line)))
    return records
