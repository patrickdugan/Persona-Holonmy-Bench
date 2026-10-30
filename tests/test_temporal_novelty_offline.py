import unittest
from persona_holonomy.temporal_novelty_offline import History, fit, evaluate, synthetic

class TemporalOfflineTests(unittest.TestCase):
    def test_synthetic_families_and_replay(self):
        for kind in ("stationary","drifting","order_sensitive","event_boundary"):
            hs=synthetic(kind)
            train=[h for h in hs if h.split=="train"]
            test=[h for h in hs if h.split=="test"]
            a=evaluate(fit(train),test,shuffled_control=fit(train,shuffled=True))
            self.assertEqual(a,evaluate(fit(train),test,shuffled_control=fit(train,shuffled=True)))
            self.assertEqual(a["independent_test_units"],8)
            self.assertEqual(len(a["per_history"]),8)
            self.assertTrue(all(r["steps"]==11 for r in a["per_history"]))
    def test_predictable_and_controls(self):
        hs=[History(i,"train" if i<4 else "test",
                    tuple((float((i%3)*2**t),) for t in range(8))) for i in range(8)]
        tr=hs[:4]; result=evaluate(fit(tr),hs[4:],shuffled_control=fit(tr,shuffled=True))
        self.assertLess(result["mean_mse"],1e-12)
        self.assertLess(result["mean_mse"],result["mean_static_mse"])
        self.assertLess(result["mean_mse"],result["mean_shuffle_mse"])
    def test_planted_novel_event(self):
        hs=[History(i,"train" if i<4 else "test",
                    tuple((float(t if t<5 else t+20),) for t in range(10)),
                    (5,)) for i in range(8)]
        # Fit on clean non-event training histories (event-specific evaluation only)
        tr=[History(i,"train",tuple((float(t),) for t in range(10))) for i in range(4)]
        a=evaluate(fit(tr),hs[4:])
        self.assertGreater(a["per_history"][0]["event_residual_norm"],10)
    def test_guards(self):
        train=History(1,"train",((0.,),(1.,),(2.,)))
        held=History(2,"test",((0.,),(1.,),(2.,)))
        with self.assertRaisesRegex(ValueError,"held-out-fit"):
            fit([train,held])
        with self.assertRaisesRegex(ValueError,"unit leakage"):
            fit([train,train])
        with self.assertRaisesRegex(ValueError,"overlaps fit"):
            evaluate(fit([train]),[History(1,"test",held.states)])
        with self.assertRaises(ValueError):
            evaluate(fit([train]),[History(3,"train",held.states)])
        with self.assertRaises(ValueError):
            fit([History(4,"train",((0.,),(float("nan"),)))])
    def test_no_future_state_in_predictor(self):
        tr=[History(i,"train",((0.,),(1.,),(2.,))) for i in range(3)]
        m=fit(tr)
        a=History(9,"test",((0.,),(1.,),(2.,)))
        b=History(9,"test",((0.,),(1.,),(999.,)))
        self.assertEqual(evaluate(m,[a])["per_history"][0]["residuals"][0],
                         evaluate(m,[b])["per_history"][0]["residuals"][0])

if __name__=="__main__":
    unittest.main()
