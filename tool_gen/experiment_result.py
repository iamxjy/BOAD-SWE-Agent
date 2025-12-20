from pathlib import Path
from typing import Any, Dict, Optional

class ExperimentResult:
    """Results from running an experiment."""
    def __init__(self, experiment_dir: str, p2p_success: int = 0, p2p_failure: int = 0,
                 f2p_success: int = 0, f2p_failure: int = 0, resolved: int = 0, unresolved: int = 0,
                 config_path: Optional[Path] = None, total_cost: float = 0.0):
        self.experiment_dir = experiment_dir
        self.p2p_success = p2p_success
        self.p2p_failure = p2p_failure
        self.f2p_success = f2p_success
        self.f2p_failure = f2p_failure
        self.resolved = resolved
        self.unresolved = unresolved
        self.config_path = config_path
        self.total_cost = total_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_dir": self.experiment_dir,
            "p2p_success": self.p2p_success,
            "p2p_failure": self.p2p_failure,
            "f2p_success": self.f2p_success,
            "f2p_failure": self.f2p_failure,
            "resolved": self.resolved,
            "unresolved": self.unresolved,
            "config_path": str(self.config_path) if self.config_path else None,
            "total_cost": self.total_cost
        } 