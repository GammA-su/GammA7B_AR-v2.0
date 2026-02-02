from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class NormalizedDocument:
    text: str
    source: str
    domain: str
    doc_id: str
    license_tag: Optional[str] = None
    created_at: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Always include optional keys for a stable schema.
        if payload.get("meta") is None:
            payload["meta"] = None
        return payload

