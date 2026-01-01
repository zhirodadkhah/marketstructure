from dataclasses import dataclass
@dataclass(frozen=True)
class MTFConfig:
    htf_bar_size: int = 24  # e.g., 24 → 1D from 1H
    min_confluence_score: float = 0.6  # For signal filtering
    alignment_method: str = "backward"  # unused in current impl