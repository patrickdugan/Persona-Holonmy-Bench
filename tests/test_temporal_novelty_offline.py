"""Self-contained CPU tests. No model, activation loading, or external services."""
import unittest

from persona_holonomy.temporal_novelty_offline import History, evaluate, fit, synthetic


class TemporalOfflineTests(unittest.TestCase):
    def test_families_determinism_and_unit_counts(self):
        for kind in ("stationary", "drifting", "order_sensitive", "event_boundary"):
            histories = synthetic(kind)
            train = [h for h in histories if h.split == "train"]
            test = [h for h in histories if h.split == "test"]
            a = evaluate(fit(train), test, shuffled_control=fit(train, shuffled=True))
            self.assertEqual(a, evaluate(fit(train), test, shuffled_control=fit(train, shuffled=True)))
            self.assertEqual(a["independent_test_units"], 8)
            self.assertEqual(len(a["per_history"]), 8)
            self.assertTrue(all(r["steps"] == 11 for r in a["per_seed"]))
            self.assertEqual(a["paired_group_bootstrap"]["time_shuffle"]["status"],
                             "descriptive_only")

    def test_exact_linear_predictability(self):
        histories = [History(i, "train" if i < 4 else "test",
                             tuple((float((i % 3) * 2**t),) for t in range(8)))
                     for i in range(8)]
        train = histories[:4]
        report = evaluate(fit(train), histories[4:], shuffled_control=fit(train, shuffled=True))
        self.assertLess(report["mean_mse"], 1e-12)
        self.assertLess(report["mean_mse"], report["mean_static_mse"])
        self.assertLess(report["mean_mse"], report["mean_shuffle_mse"])

    def test_order_sensitive_vs_time_shuffle(self):
        histories = synthetic("order_sensitive")
        train = [h for h in histories if h.split == "train"]
        result = evaluate(fit(train), [h for h in histories if h.split == "test"],
                          shuffled_control=fit(train, shuffled=True))
        self.assertLess(result["mean_mse"], result["mean_shuffle_mse"])

    def test_planted_novel_event_localizes(self):
        histories = synthetic("event_boundary")
        train = [h for h in histories if h.split == "train"]
        self.assertTrue(all(not h.events for h in train))
        result = evaluate(fit(train), [h for h in histories if h.split == "test"])
        self.assertTrue(all(row["event_residual_norm"] >
                            row["ordinary_residual_norm"] * 3
                            for row in result["per_seed"]))

    def test_static_mean_difference_is_train_fitted(self):
        histories = [History(i, "train" if i < 4 else "test",
                             tuple((float(i + t / 2),) for t in range(8)))
                     for i in range(8)]
        train = histories[:4]
        trained = fit(train)
        self.assertAlmostEqual(trained.mean_difference[0], .5)
        result = evaluate(trained, histories[4:])
        self.assertLess(result["mean_static_mean_difference_mse"], 1e-14)

    def test_repeated_probes_count_once_per_seed(self):
        train = [History(i, "train", ((0.,), (1.,), (2.,))) for i in range(4)]
        test = [History(i, "test", ((0.,), (1.,), (2.,)), probe_id="alpha")
                for i in range(4, 6)]
        test += [History(i, "test", ((0.,), (1.,), (2.,)), probe_id="beta")
                 for i in range(4, 6)]
        report = evaluate(fit(train), test)
        self.assertEqual(report["independent_test_units"], 2)
        self.assertEqual(len(report["per_history"]), 4)
        self.assertTrue(all(r["probe_count"] == 2 for r in report["per_seed"]))
        self.assertEqual(report["paired_group_bootstrap"]["static_mean"]["status"],
                         "insufficient_independent_units")
        self.assertIsNone(report["paired_group_bootstrap"]["static_mean"]["bootstrap_95_interval"])

    def test_fitting_equal_seed_weight_when_repeated(self):
        a = History(1, "train", ((1.,), (1.,), (1.,)))
        b = History(2, "train", ((9.,), (9.,), (9.,)))
        a_repeat = [History(1, "train", ((1.,), (1.,), (1.,)), probe_id=f"p{i}")
                    for i in range(5)]
        self.assertAlmostEqual(fit([a, b]).mean[0], fit([a, b] + a_repeat).mean[0])

    def test_guards_and_leakage(self):
        train = History(1, "train", ((0.,), (1.,), (2.,)))
        held = History(2, "test", ((0.,), (1.,), (2.,)))
        with self.assertRaisesRegex(ValueError, "held-out-fit"):
            fit([train, held])
        with self.assertRaisesRegex(ValueError, "duplicate seed/probe"):
            fit([train, train])
        with self.assertRaisesRegex(ValueError, "group leakage"):
            evaluate(fit([train]), [held, History(2, "train", held.states, probe_id="a")])
        with self.assertRaisesRegex(ValueError, "overlaps fit"):
            evaluate(fit([train]), [History(1, "test", held.states)])
        with self.assertRaises(ValueError):
            fit([History(4, "train", ((0.,), (float("nan"),)))])
        with self.assertRaises(ValueError):
            fit([History(4, "train", ((0.,), (1.,)), (1, 1))])

    def test_control_training_fingerprint_binding(self):
        train = [History(i, "train", ((0.,), (1.,), (2.,))) for i in range(4)]
        altered = [History(i, "train", ((0.,), (3.,), (2.,))) for i in range(4)]
        held = [History(8, "test", ((0.,), (1.,), (2.,)))]
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            evaluate(fit(train), held, shuffled_control=fit(altered, shuffled=True))

    def test_future_state_prefix_invariance(self):
        train = [History(i, "train", ((0.,), (1.,), (2.,))) for i in range(3)]
        model = fit(train)
        a = History(9, "test", ((0.,), (1.,), (2.,)))
        b = History(9, "test", ((0.,), (1.,), (999.,)))
        self.assertEqual(evaluate(model, [a])["per_history"][0]["residuals"][0],
                         evaluate(model, [b])["per_history"][0]["residuals"][0])


if __name__ == "__main__":
    unittest.main()
