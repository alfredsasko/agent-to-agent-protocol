"""Tests for the local doctor MCP tool."""

import unittest

from mcpserver import list_doctors


class DoctorToolTest(unittest.TestCase):
    def test_requires_a_location(self) -> None:
        self.assertEqual(
            list_doctors(),
            [{"error": "Please provide a state or a city."}],
        )

    def test_search_is_case_insensitive(self) -> None:
        results = list_doctors(state="ma", city="BOSTON")
        self.assertGreater(len(results), 0)
        self.assertTrue(
            all(
                doctor["address"]["state"] == "MA"
                and doctor["address"]["city"] == "Boston"
                for doctor in results
            )
        )


if __name__ == "__main__":
    unittest.main()
