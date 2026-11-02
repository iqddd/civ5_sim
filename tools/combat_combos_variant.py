"""Generate the blocker-driven combos experiment from frozen port sources.

The mechanism follows CvTacticalAI.cpp:8174,8490,8685 (pinned CP Only).
Candidate values still use the port's existing plot, danger and attack models.
"""

from pathlib import Path

FRAGMENTS = Path(__file__).with_name("experiments")


def change(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence: {old[:100]!r}; got {text.count(old)}")
    return text.replace(old, new)


def transform(directory):
    native = Path(directory) / "native/combat_ai"
    header = native / "model.h"
    text = header.read_text(encoding="utf-8")
    text = change(text, "  virtual std::vector<Command> actions(const Branch&) const = 0;", """  // CP MOVEFLAG_IGNORE_STACKING_SELF: proposals may end on a friendly.
  virtual std::vector<ReachableTile> move_intents(const Branch&, int32_t unit) const = 0;
  // Private hypothetical vacancy for evaluating the moving unit's proposal.
  // No physical command, contact update or visibility reveal is performed.
  virtual std::unique_ptr<Branch> assume_move(const Branch&, int32_t unit,
                                             int32_t tile, int32_t moves) const = 0;
  virtual std::vector<Command> actions(const Branch&) const = 0;""")
    header.write_text(text, encoding="utf-8")
    model = native / "model.cpp"
    text = model.read_text(encoding="utf-8")
    marker = "  std::vector<Command> actions(const Branch& base) const override {"
    text = change(text, marker, (FRAGMENTS / "combat_combos_model.inc").read_text(encoding="utf-8") + marker)
    model.write_text(text, encoding="utf-8")
    planner = native / "planner.cpp"
    text = planner.read_text(encoding="utf-8")
    begin = text.index("  std::vector<Candidate> candidates(")
    end = text.index("  std::vector<int32_t> key(", begin)
    text = text[:begin] + (FRAGMENTS / "combat_combos_candidates.inc").read_text(encoding="utf-8") + text[end:]
    planner.write_text(text, encoding="utf-8")
