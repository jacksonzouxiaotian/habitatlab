"""Strict result completeness and dry-run plan sequencing, not a robot driver."""
from dataclasses import dataclass, field
import math

def aggregate(expected_keys, rows):
    """Do not silently drop failed/missing trials or select the best retry.

    Caller supplies one designated attempt per expected task. An error trial must
    be resolved explicitly; null measurements are never converted to zero scores.
    """
    expected = set(expected_keys)
    if len(expected) != len(expected_keys):
        raise ValueError('duplicate_expected_keys')
    seen = set()
    for row in rows:
        key = (row['scene_id'], str(row['episode_id']))
        if key in seen:
            raise ValueError('duplicate_result')
        if key not in expected:
            raise ValueError('unexpected_result')
        seen.add(key)
        if row.get('metric_source') != 'actual_formal_evaluator':
            raise ValueError('not_formal_metrics')
        if row.get('termination_reason') != 'model_stop' and row.get('success') == 1:
            raise ValueError('nonmodel_termination_cannot_count_as_model_success')
        for name in ('success', 'spl'):
            value = row.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('invalid_metric:' + name)
    if seen != expected:
        raise ValueError('missing_results')
    if not rows:
        raise ValueError('empty_results')
    return {'n':len(rows), 'sr':sum(r['success'] for r in rows)/len(rows), 'spl':sum(r['spl'] for r in rows)/len(rows)}

@dataclass
class DryRunPlanGate:
    """No ROS imports, no sockets, no /cmd_vel writes. Local monotonic times only.

    Actions are already parsed native primitives; this gate does not alter units
    or synthesize an action. A model's native executor chooses its execution horizon.
    """
    session_id: str
    latest_observation: int = -1
    plan_id: str = ''
    actions: list = field(default_factory=list)
    consumed: int = 0
    expires_at: float = 0.
    seen_plans: set = field(default_factory=set)

    def reset(self, session_id):
        self.session_id = session_id
        self.latest_observation = -1
        self.plan_id = ''
        self.actions.clear()
        self.consumed = 0
        self.expires_at = 0.
        self.seen_plans.clear()

    def accept(self, message, now, observation_capture_local, max_age):
        if message['session_id'] != self.session_id:
            return 'wrong_session'
        if message['plan_id'] in self.seen_plans:
            return 'duplicate_plan'
        if message['observation_id'] <= self.latest_observation:
            return 'out_of_order'
        if not 0 <= now - observation_capture_local <= max_age:
            return 'stale_observation'
        if message['mode'] not in ('act','reason','invalid'):
            return 'invalid_mode'
        self.latest_observation = message['observation_id']
        self.plan_id = message['plan_id']
        self.seen_plans.add(self.plan_id)
        self.actions = list(message['actions']) if message['mode'] == 'act' else []
        self.consumed = 0
        self.expires_at = observation_capture_local + max_age
        return message['mode']

    def consume(self, now):
        if now >= self.expires_at:
            self.actions.clear()
            return None
        if self.consumed == len(self.actions):
            return None
        action = self.actions[self.consumed]
        self.consumed += 1
        return action
