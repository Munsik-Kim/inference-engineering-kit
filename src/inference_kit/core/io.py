"""Immutable JSON, canonical identity and strict path primitives from modelpack."""
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any
def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def loads(blob: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(blob, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def read(path: Path) -> Any:
    return loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    require(not path.exists(), "Refusing to overwrite " + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(blob)


def safe_name(name: str) -> str:
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and ".." not in p.parts and "\\" not in name
            and ":" not in name and str(p) == name, "Unsafe artifact path")
    return name


def fields(value, required, optional=()):
    require(isinstance(value, dict), "Expected JSON object")
    require(set(required) <= value.keys(), "Missing fields: " + str(sorted(set(required) - value.keys())))
    require(value.keys() <= set(required) | set(optional), "Unknown fields: " + str(sorted(value.keys() - set(required) - set(optional))))
    return value

