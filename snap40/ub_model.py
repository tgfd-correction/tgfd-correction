from dataclasses import dataclass, field
from collections import Counter
from typing import Dict
@dataclass
class UniformBlock:
    """
    Represents a Uniform Block (UB) for the TGFD (person, hours) → pay.
    Attributes:
      - subject: the full RDF URI of the person (e.g., "<Employee
      - hours: the hours value as a string
      - values_by_snapshot: Dictionary mapping snapshot index (int) to a Counter of pay values,
          e.g. { "0": Counter({"744": 1}) }
      - values_by_snapshot_injected: A deep copy of values_by_snapshot for error injection.
    """
    subject: str
    hours: str
    values_by_snapshot: Dict[int, Counter[str, int]]
    values_by_snapshot_injected: Dict[int, Counter[str, int]] = field(default_factory=dict)
    def __post_init__(self):
        if not self.values_by_snapshot_injected:
            self.values_by_snapshot_injected = {
                snap: Counter(vals) for snap, vals in self.values_by_snapshot.items()
            }
    def total_matches(self) -> int:
        """Return the total number of TGFD matches (sum over snapshots)."""
        return sum(sum(counter.values()) for counter in self.values_by_snapshot.values())
    def to_dict(self) -> Dict:
        """Return a dictionary representation suitable for JSON serialization."""
        return {
            "subject": self.subject,
            "hours": self.hours,
            "values_by_snapshot": {str(snap): dict(counter) for snap, counter in self.values_by_snapshot.items()},
            "values_by_snapshot_injected": {str(snap): dict(counter) for snap, counter in
                                            self.values_by_snapshot_injected.items()}
        }
    def ideal_distribution(self):
        """
        Compute the ideal distribution of matches across snapshots for this UB.
        Returns a tuple: (distribution_dict, total_count).
        """
        total = self.total_matches()  
        if total == 0:
            return {}, 0
        freq_by_snapshot = {snap: sum(counter.values())
                            for snap, counter in self.values_by_snapshot.items()}
        distribution = {snap: count / total for snap, count in freq_by_snapshot.items()}
        return distribution, total
