"""Half-open decision ranges within an episode."""

from typing import NamedTuple


class Span(NamedTuple):
    episode: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start
