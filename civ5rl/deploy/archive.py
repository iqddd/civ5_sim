"""Deployment-facing imports for the local packet codec."""

from __future__ import annotations

from civ5rl.deploy.remote import _read as _read
from civ5rl.deploy.remote import _sha as _sha
from civ5rl.deploy.remote import _write as _write
from civ5rl.engine.runtime import archive as _codec

JSON_MIN_BYTES = _codec.JSON_MIN_BYTES
JSON_MIN_FILES = _codec.JSON_MIN_FILES
PPMD = _codec.PPMD

needs_archive = _codec.needs_archive
pack = _codec.pack
seven_zip = _codec.seven_zip
unpack = _codec.unpack

_extract = _codec.extract
_inventory = _codec.inventory
_name = _codec.member_name
