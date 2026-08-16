import unittest

from threes_rl.promotion_label_report import classify_label, summarize_rows


class PromotionLabelReportTests(unittest.TestCase):
    def test_classifies_action_sensitive_promotion_rates(self):
        item = {
            "id": "sample",
            "base_action": "left",
            "comparison_action": "up",
            "top_two_actions": ["left", "up"],
            "label": {
                "stable": False,
                "horizon_results": [
                    {
                        "horizon": 40,
                        "promotion_rate": {"left": 0.0, "up": 0.25},
                        "promotion_winner": "up",
                        "mean_delta": {"left": 10.0, "up": 20.0},
                    }
                ],
                "promotion_rate_gain_vs_base_at_max_horizon": 0.25,
                "oracle_regret_at_max_horizon": 10.0,
            },
        }

        row = classify_label(item, rate_gap=0.125)
        summary = summarize_rows([row])

        self.assertEqual(row["bucket"], "action_sensitive")
        self.assertEqual(summary["action_sensitive_labels"], 1)
        self.assertEqual(summary["positive_promotion_gains"], 1)

    def test_classifies_inevitable_and_unreachable(self):
        inevitable = {
            "label": {
                "horizon_results": [
                    {"horizon": 40, "promotion_rate": {"left": 1.0, "up": 1.0}}
                ]
            }
        }
        unreachable = {
            "label": {
                "horizon_results": [
                    {"horizon": 40, "promotion_rate": {"left": 0.0, "up": 0.0}}
                ]
            }
        }

        self.assertEqual(classify_label(inevitable, rate_gap=0.125)["bucket"], "inevitable")
        self.assertEqual(
            classify_label(unreachable, rate_gap=0.125)["bucket"],
            "unreachable_by_horizon",
        )


if __name__ == "__main__":
    unittest.main()
