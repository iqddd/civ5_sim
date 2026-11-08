"""Keep the stable contract links in native code and behavioral tests usable."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ContractLinksTests(unittest.TestCase):
    def test_native_contract_references_resolve(self):
        contract = ROOT / "docs/combat-ai/combat-ai-contract.md"
        anchors = set(re.findall(r'<a id="([a-z-]+)"></a>', contract.read_text(encoding="utf-8")))
        sources = sorted((ROOT / "native/combat_ai").glob("*.cpp"))
        sources += sorted((ROOT / "native/combat_ai").glob("*.h"))
        sources.append(ROOT / "tests/native/combat_ai_planner_test.cpp")
        referenced = set()
        for source in sources:
            for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
                if "// Contract: docs/combat-ai/combat-ai-contract.md#" not in line:
                    continue
                for fragment in re.findall(r"#([a-z][a-z-]*)", line):
                    with self.subTest(source=str(source.relative_to(ROOT)), line=number, anchor=fragment):
                        self.assertIn(fragment, anchors)
                    referenced.add(fragment)
        self.assertEqual(referenced, anchors, "Every stable contract anchor should have a code or test reference")
