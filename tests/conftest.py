import os
import pytest


INCLUDE_SKIPS = os.getenv("ROCM_TEST_INCLUDE_SKIPS", default=False)

@pytest.hookimpl(optionalhook=True)
def pytest_json_modifyreport(json_report):
    """Get rid of skipped tests in reporting. We only care about xfails."""
    if (not INCLUDE_SKIPS
            and "summary" in json_report
            and "total" in json_report["summary"]):
        json_report["summary"]["unskipped_total"] = json_report["summary"]["total"] - json_report["summary"].get("skipped", 0)
        del json_report["summary"]["total"]
