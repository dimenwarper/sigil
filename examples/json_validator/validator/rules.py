"""Baseline JSON payload validator with redundant passes over the data."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


REQUIRED_FIELDS = ["id", "name", "active", "roles"]


def validate_payload(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing:{field}")
    for field in REQUIRED_FIELDS:
        if field in payload:
            value = payload[field]
            if field == "id" and not isinstance(value, int):
                errors.append("id_type")
            elif field == "name" and not (isinstance(value, str) and value.strip()):
                errors.append("name_type")
            elif field == "active" and not isinstance(value, bool):
                errors.append("active_type")
            elif field == "roles":
                if not isinstance(value, list) or not value:
                    errors.append("roles_type")
                else:
                    for item in value:
                        if not isinstance(item, str) or not item:
                            errors.append("roles_member")
                            break
    unknown = sorted(set(payload.keys()) - set(REQUIRED_FIELDS))
    for extra in unknown:
        errors.append(f"unknown:{extra}")
    return len(errors) == 0, errors
