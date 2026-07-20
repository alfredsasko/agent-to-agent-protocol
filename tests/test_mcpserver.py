"""Tests for the local doctor MCP tool."""

from mcpserver import list_doctors


def test_requires_a_location() -> None:
    assert list_doctors() == [{"error": "Please provide a state or a city."}]


def test_search_is_case_insensitive() -> None:
    results = list_doctors(state="ma", city="BOSTON")
    assert results
    assert all(
        doctor["address"]["state"] == "MA"
        and doctor["address"]["city"] == "Boston"
        for doctor in results
    )
