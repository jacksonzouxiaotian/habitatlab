import unittest
from result_contract import aggregate, DryRunPlanGate

class ResultContracts(unittest.TestCase):
    def row(self, ep='1', **kw):
        return dict(scene_id='scene', episode_id=ep, success=1, spl=.5,
                    metric_source='actual_formal_evaluator', termination_reason='model_stop', **kw)

    def test_aggregate_all_tasks(self):
        a, b = self.row(), self.row('2')
        b.update(success=0, spl=0, termination_reason='budget_exceeded')
        self.assertEqual(aggregate([('scene','1'),('scene','2')], [a,b]), {'n':2,'sr':.5,'spl':.25})

    def test_missing_and_duplicates_rejected(self):
        for rows in ([], [self.row(), self.row()]):
            with self.assertRaises(ValueError): aggregate([('scene','1')], rows)

    def test_forced_stop_not_model_success(self):
        r = self.row(); r['termination_reason'] = 'inference_budget_exceeded'
        with self.assertRaises(ValueError): aggregate([('scene','1')], [r])

    def test_counterfactual_not_formal(self):
        r = self.row(); r['metric_source'] = 'offline_counterfactual'
        with self.assertRaises(ValueError): aggregate([('scene','1')], [r])

    def test_failed_unmeasured_trial_not_zero(self):
        r = self.row(); r['success'] = None
        with self.assertRaises(ValueError): aggregate([('scene','1')], [r])

    def message(self, obs=1, plan='p1', mode='act', session='s1'):
        return dict(session_id=session, observation_id=obs, plan_id=plan,
                    mode=mode, actions=['forward_25cm','left_30deg'])

    def test_no_duplicate_consumption(self):
        g=DryRunPlanGate('s1'); m=self.message()
        self.assertEqual(g.accept(m,10,9,2),'act')
        self.assertEqual(g.consume(10),'forward_25cm')
        self.assertEqual(g.accept(m,10,9,2),'duplicate_plan')
        self.assertEqual(g.consume(10),'left_30deg')
        self.assertIsNone(g.consume(10))

    def test_out_of_order_rejected(self):
        g=DryRunPlanGate('s1'); g.accept(self.message(2),10,9,2)
        self.assertEqual(g.accept(self.message(1,'p0'),10,9,2),'out_of_order')

    def test_stale_and_wrong_session(self):
        g=DryRunPlanGate('s1')
        self.assertEqual(g.accept(self.message(),10,1,2),'stale_observation')
        self.assertEqual(g.accept(self.message(session='old'),10,9,2),'wrong_session')

    def test_reason_and_invalid_never_move_or_stop(self):
        for mode in ('reason','invalid'):
            g=DryRunPlanGate('s1'); g.accept(self.message(mode=mode),10,9,2)
            self.assertIsNone(g.consume(10))

    def test_expiry_watchdog_and_reset(self):
        g=DryRunPlanGate('s1'); g.accept(self.message(),10,9,2)
        self.assertIsNone(g.consume(11))
        g.reset('s2'); self.assertIsNone(g.consume(12))
        self.assertEqual(g.accept(self.message(),12,11,2),'wrong_session')

if __name__ == '__main__': unittest.main(verbosity=2)
