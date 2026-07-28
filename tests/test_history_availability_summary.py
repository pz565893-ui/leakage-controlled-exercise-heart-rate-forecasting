import unittest

import numpy as np
import pandas as pd

from src.summarize_history_availability import summarize_regime


class HistoryAvailabilitySummaryTests(unittest.TestCase):
    def test_repeated_origins_do_not_duplicate_sessions(self):
        session_ids = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
        prior_counts = np.array([0, 2, 10], dtype=np.int32)
        session_users = pd.Series([7, 7, 9], index=[0, 1, 2])
        result = summarize_regime("test", session_ids, prior_counts, session_users)
        self.assertEqual(result["sessions"], 3)
        self.assertEqual(result["users"], 2)
        self.assertEqual(result["sessions_with_history"], 2)
        self.assertEqual(result["sessions_prior_0"], 1)
        self.assertEqual(result["sessions_prior_1_4"], 1)
        self.assertEqual(result["sessions_prior_10_plus"], 1)
        self.assertEqual(result["users_with_any_history"], 2)
        self.assertEqual(result["users_with_history_in_all_test_sessions"], 1)

    def test_no_history_is_preserved_as_absence(self):
        session_ids = np.array([0, 1], dtype=np.int32)
        prior_counts = np.array([0, 0], dtype=np.int32)
        session_users = pd.Series([3, 4], index=[0, 1])
        result = summarize_regime("test", session_ids, prior_counts, session_users)
        self.assertEqual(result["sessions_without_history"], 2)
        self.assertEqual(result["users_with_any_history"], 0)
        self.assertEqual(result["prior_count_session_median"], 0.0)


if __name__ == "__main__":
    unittest.main()

