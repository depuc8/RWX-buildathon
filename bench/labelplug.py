"""pytest plugin: record every failing nodeid (subtests and collection errors
included) to $LABEL_OUT."""

import json
import os

failed = set()
errors = set()


def pytest_runtest_logreport(report):
    if report.failed:
        failed.add(report.nodeid)


def pytest_collectreport(report):
    if report.failed:
        errors.add(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["LABEL_OUT"], "w") as fh:
        json.dump({"failed": sorted(failed), "errors": sorted(errors), "exit": int(exitstatus)}, fh)
