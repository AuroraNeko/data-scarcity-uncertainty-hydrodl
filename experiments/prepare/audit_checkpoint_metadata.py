"""Remove machine-local paths from scarcity sensitivity checkpoints.

This rewrites checkpoint metadata only. A stable digest over the
model-state tensors is computed before and after rewriting to prove that the
learned parameters are unchanged; compact result JSON hashes are then updated.
Checkpoints that already store repository-relative paths pass through the
same audit without a rewrite.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
GROUPS = [
    "scarcity_replications",
    "scarcity_equal_updates",
    "scarcity_standard_lstm",
]
CANONICAL_SUBSET_PATH = "results/tables/scarcity_replications/replication_subsets.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def repository_path(value: str) -> Path:
    parts = value.replace("\\", "/").split("/")
    if (
        not value
        or any(part in {"", ".", ".."} or ":" in part for part in parts)
    ):
        raise ValueError(f"Invalid repository-relative path: {value!r}")
    path = ROOT.joinpath(*parts)
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {value!r}") from exc
    return path


def contains_workspace_path(value: object) -> bool:
    root = str(ROOT).replace("\\", "/").lower()
    if isinstance(value, str):
        return root in value.replace("\\", "/").lower()
    if isinstance(value, dict):
        return any(
            contains_workspace_path(key) or contains_workspace_path(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_workspace_path(item) for item in value)
    return False


def contains_absolute_path(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        return bool(
            re.match(r"^[A-Za-z]:/", normalized)
            or normalized.startswith("//")
            or normalized.startswith("/")
        )
    if isinstance(value, dict):
        return any(
            contains_absolute_path(key) or contains_absolute_path(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_absolute_path(item) for item in value)
    return False


def sanitize_result(path: Path, migrate: bool = False) -> bool:
    result = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_path = repository_path(result["checkpoint"])
    if sha256(checkpoint_path) != result.get('checkpoint_sha256'):
        raise ValueError(f'Checkpoint hash changed before metadata audit: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_digest = tensor_state_sha256(checkpoint["model_state_dict"])
    if state_digest != result.get('checkpoint_tensor_sha256'):
        raise ValueError(f'Trusted tensor digest missing or changed: {checkpoint_path}')

    args = checkpoint.get("args")
    rewritten = False
    if isinstance(args, dict) and args.get("basin_list_file"):
        value = str(args["basin_list_file"])
        normalized = value.replace("\\", "/")
        if not normalized.lower().endswith(CANONICAL_SUBSET_PATH.lower()):
            raise ValueError(f"Unexpected basin-list path in {checkpoint_path}: {value!r}")
        if value != CANONICAL_SUBSET_PATH:
            if not migrate:
                raise ValueError('Legacy path requires explicit --migrate-paths after identity verification')
            args["basin_list_file"] = CANONICAL_SUBSET_PATH
            rewritten = True

    if contains_workspace_path(checkpoint):
        raise ValueError(f"Machine-local workspace path remains in {checkpoint_path}")
    if contains_absolute_path(checkpoint):
        raise ValueError(f"Absolute filesystem path remains in {checkpoint_path}")

    if rewritten:
        temp = checkpoint_path.with_suffix(".sanitized.tmp.pt")
        torch.save(checkpoint, temp)
        reloaded = torch.load(temp, map_location="cpu", weights_only=False)
        if tensor_state_sha256(reloaded["model_state_dict"]) != state_digest:
            temp.unlink(missing_ok=True)
            raise ValueError(f"Tensor state changed while sanitizing {checkpoint_path}")
        temp.replace(checkpoint_path)

    if not rewritten:
        return False
    previous_hash = result['checkpoint_sha256']
    result["checkpoint_sha256"] = sha256(checkpoint_path)
    result["checkpoint_tensor_sha256"] = state_digest
    result["checkpoint_metadata_sanitized"] = True
    result['metadata_migration'] = {'previous_checkpoint_sha256': previous_hash,
                                  'tensor_sha256': state_digest, 'operation': 'relative_subset_path'}
    temp_json = path.with_suffix(path.suffix + ".tmp")
    temp_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temp_json.replace(path)
    return rewritten


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--migrate-paths', action='store_true')
    args = parser.parse_args()
    paths = []
    for group in GROUPS:
        directory = ROOT / "results" / "tables" / group
        if directory.exists():
            # Per-fit results carry a modelseed suffix; aggregate summaries
            # such as scarcity_replication_summary.json have no checkpoint
            # and must stay out of the checkpoint/result audit.
            paths.extend(sorted(directory.glob("scarcity_*_modelseed*.json")))
    if not paths:
        raise FileNotFoundError("No v3 scarcity sensitivity result files found")

    rewritten = sum(sanitize_result(path, args.migrate_paths) for path in paths)
    print(f"Audited {len(paths)} checkpoint/result pairs; rewrote {rewritten} checkpoints")
    print("Model-state tensor digests were unchanged")


if __name__ == "__main__":
    main()
