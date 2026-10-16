from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data_provider" / "metadata"

def attachment_dir(name):
    local = ROOT / name
    return local if local.exists() else ROOT.parent / name

def artifact_path(relative):
    value = Path(relative)
    if value.is_absolute():
        return value
    local = ROOT / value
    if local.exists() or not value.parts or value.parts[0] not in {"data_release"}:
        return local
    return ROOT.parent / value
