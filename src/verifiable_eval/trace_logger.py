"""Append-only execution trace with hash chain integrity."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .models import EntryType, TraceEntry

_GENESIS_HASH = "0" * 64  # sentinel for first entry


class ExecutionTraceLogger:
    """Append-only log where each entry's hash includes the previous entry's hash."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._entries: list[TraceEntry] = []
        self._path = Path(path) if path else None
        self._last_hash = _GENESIS_HASH

        # Resume from existing file
        if self._path and self._path.exists():
            self._load()

    # -- appending -------------------------------------------------------

    def append(
        self,
        entry_type: EntryType,
        data: dict[str, object],
        timestamp: str | None = None,
    ) -> TraceEntry:
        """Append a new entry to the trace and return it."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        entry_id = len(self._entries)

        entry = TraceEntry(
            entry_id=entry_id,
            timestamp=ts,
            entry_type=entry_type,
            data=data,
            previous_hash=self._last_hash,
        )
        entry.entry_hash = self._compute_hash(entry)
        self._last_hash = entry.entry_hash
        self._entries.append(entry)

        # Append to file immediately
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.model_dump(mode="json"), sort_keys=True) + "\n")

        return entry

    # -- verification ----------------------------------------------------

    def verify_chain(self) -> tuple[bool, str]:
        """Walk the chain and verify every entry's hash. Returns (ok, detail)."""
        if not self._entries:
            return True, "Empty trace"

        prev_hash = _GENESIS_HASH
        for entry in self._entries:
            if entry.previous_hash != prev_hash:
                return False, (
                    f"Entry {entry.entry_id}: previous_hash mismatch "
                    f"(expected {prev_hash[:16]}..., got {entry.previous_hash[:16]}...)"
                )
            expected = self._compute_hash(entry)
            if entry.entry_hash != expected:
                return False, (
                    f"Entry {entry.entry_id}: entry_hash mismatch "
                    f"(expected {expected[:16]}..., got {entry.entry_hash[:16]}...)"
                )
            prev_hash = entry.entry_hash

        return True, f"Chain intact ({len(self._entries)} entries)"

    # -- access ----------------------------------------------------------

    @property
    def entries(self) -> list[TraceEntry]:
        return list(self._entries)

    @property
    def last_hash(self) -> str:
        return self._last_hash

    @property
    def count(self) -> int:
        return len(self._entries)

    def judge_entries(self) -> list[TraceEntry]:
        """Return only judge_decision entries."""
        return [e for e in self._entries if e.entry_type == EntryType.JUDGE_DECISION]

    # -- serialisation ---------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the full trace to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry.model_dump(mode="json"), sort_keys=True) + "\n")

    @classmethod
    def from_file(cls, path: str | Path) -> ExecutionTraceLogger:
        """Load a trace from a JSONL file."""
        instance = cls.__new__(cls)
        instance._entries = []
        instance._path = Path(path)
        instance._last_hash = _GENESIS_HASH
        instance._load()
        return instance

    # -- internal --------------------------------------------------------

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            entry = TraceEntry.model_validate(json.loads(line))
            self._entries.append(entry)
            self._last_hash = entry.entry_hash

    @staticmethod
    def _compute_hash(entry: TraceEntry) -> str:
        """Hash the entry content + previous_hash. Excludes entry_hash itself."""
        payload = json.dumps(
            {
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp,
                "entry_type": entry.entry_type.value,
                "data": entry.data,
                "previous_hash": entry.previous_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
