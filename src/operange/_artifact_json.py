"""Artifact transport mechanics, with no model or result-family dependencies.

Schemas choose their own identity body and semantic validation. This module
only encodes JSON, reads it strictly, and checks envelopes and content identities.
Loading or matching an identity does not verify the evidence in an artifact.
"""

from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
import hashlib
import json
import math


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if is_dataclass(value):
        return {item.name: plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    return value


def dumps(value, *, indent=None, sort_keys=False, separators=None):
    """Encode finite JSON, retaining each existing schema's presentation options."""
    return json.dumps(
        value,
        indent=indent,
        sort_keys=sort_keys,
        separators=separators,
        allow_nan=False,
    )


def canonical_json(value):
    """The historical encoding used for Operange's SHA-256 content identities."""
    return dumps(plain(value), sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(document):
    """Reject duplicate keys and nonfinite numbers anywhere in a JSON document."""
    if not isinstance(document, (str, bytes, bytearray)):
        raise ValueError("expected a JSON document")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            return constant(value)
        return result

    return json.loads(
        document, object_pairs_hook=pairs, parse_constant=constant, parse_float=number
    )


def require_fields(data, required, *, optional=()):
    if not isinstance(data, Mapping) or not (
        set(required) <= set(data) <= set(required) | set(optional)
    ):
        raise ValueError("invalid artifact envelope fields")


def require_schema(data, schema):
    if not isinstance(data, Mapping):
        raise ValueError("artifact must be a JSON object")
    actual = data.get("schema_version")
    if type(actual) is not type(schema) or actual != schema:
        raise ValueError(f"unsupported artifact schema; expected {schema}")


def check_identity(expected, actual):
    if not isinstance(expected, str) or expected != actual:
        raise ValueError("artifact identity/digest mismatch (integrity check failed)")


def record_to_json(record, schema):
    """Encode the flat record envelope used by numerical study reports."""
    return dumps(
        {
            "schema_version": schema,
            "result_id": record.ref.artifact_id,
            **record.to_dict(),
        },
        indent=2,
    )


def record_from_dict(cls, data, schema, *, decode=None):
    """Load a flat record envelope; its constructor owns all semantic checks.

    A family may decode nested artifacts before construction. Optional dataclass
    fields retain their historical defaults, including intentionally omitted fields.
    The caller's mapping is never mutated.
    """
    require_schema(data, schema)
    required, optional = ["schema_version", "result_id"], []
    for item in fields(cls):
        target = (
            required
            if item.default is MISSING and item.default_factory is MISSING
            else optional
        )
        target.append(item.name)
    require_fields(data, required, optional=optional)
    body = {
        key: value
        for key, value in data.items()
        if key not in ("schema_version", "result_id")
    }
    try:
        result = cls(**(body if decode is None else decode(body)))
    except (TypeError, KeyError, IndexError) as exc:
        raise ValueError(f"invalid {cls.__name__} fields") from exc
    check_identity(data["result_id"], result.ref.artifact_id)
    return result
