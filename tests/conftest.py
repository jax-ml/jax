import pytest


@pytest.hookimpl(optionalhook=True)
def pytest_json_modifyreport(json_report):
    """Get rid of skipped tests in reporting. We only care about xfails."""
    if "summary" in json_report and "total" in json_report["summary"]:
        json_report["summary"]["unskipped_total"] = json_report["summary"]["total"] - json_report["summary"].get("skipped", 0)
