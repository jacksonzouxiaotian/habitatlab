"""Test Habitat's real Success class when the diagnostic runtime has Habitat."""
from types import SimpleNamespace
import unittest

try:
    from habitat.tasks.nav.nav import Success
except ImportError:
    Success = None

@unittest.skipIf(Success is None, 'Run with navila-eval Python for actual Habitat 0.1.7 metrics')
class FormalSuccessBoundary(unittest.TestCase):
    def evaluate(self, distance, stop):
        metric = Success(sim=None, config=SimpleNamespace(SUCCESS_DISTANCE=3.0))
        task = SimpleNamespace(is_stop_called=stop, measurements=SimpleNamespace(measures={
            'distance_to_goal':SimpleNamespace(get_metric=lambda:distance)}))
        metric.update_metric(episode=None, task=task)
        return metric.get_metric()

    def test_strict_distance_boundary(self):
        self.assertEqual(self.evaluate(2.999, True), 1.)
        self.assertEqual(self.evaluate(3.0, True), 0.)
        self.assertEqual(self.evaluate(3.001, True), 0.)

    def test_reach_without_stop_not_success(self):
        self.assertEqual(self.evaluate(.1,False), 0.)

    def test_infinite_distance_failure(self):
        self.assertEqual(self.evaluate(float('inf'),True), 0.)

if __name__ == '__main__': unittest.main(verbosity=2)
