from dataclasses import dataclass


@dataclass(frozen=True)
class PassageFeatures:
    clearance_left: float = 5.0
    clearance_right: float = 5.0
    passage_width: float = 10.0
    heading_error: float = 0.0
    lateral_offset: float = 0.0
    current_vx: float = 0.0
    current_wz: float = 0.0
    stuck_score: float = 0.0
    collision_flag: float = 0.0
    previous_action_vx: float = 0.0
    previous_action_wz: float = 0.0

    @property
    def min_clearance(self) -> float:
        return min(self.clearance_left, self.clearance_right)
