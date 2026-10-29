"""Write a self-contained local HTML inspection artifact (no server needed)."""

import json
from pathlib import Path


def write_viewer(maps, destination):
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    payload = json.dumps(maps, ensure_ascii=False).replace("<", "\\u003c")
    Path(destination).write_text(
        template.replace("/*MAP_DATA*/ []", payload), encoding="utf-8"
    )
