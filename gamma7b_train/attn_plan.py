from __future__ import annotations
from dataclasses import dataclass
from typing import Set

@dataclass(frozen=True)
class AttnPlan:
    swa_window: int
    global_layers: Set[int]
    sink_tokens: int

    def is_global(self, layer_idx: int) -> bool:
        return layer_idx in self.global_layers
