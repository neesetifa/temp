"""Trusted pytest launcher.

This file is executed with the image-built Python copy under ``-I -S``.
That prevents solve-phase ``sitecustomize.py``, user-site, PYTHONPATH and the
working directory from running code before the verifier.  Only the trusted
runtime site-packages and /tests are added after interpreter startup.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

LOG = pathlib.Path('/logs/verifier')
CTRF = LOG / 'ctrf.json'
TRUSTED_SITE = pathlib.Path('/opt/oe-runtime/usr/local/lib/python3.11/site-packages')


def _install_trusted_import_paths() -> None:
    if not (sys.flags.isolated and sys.flags.no_site):
        raise RuntimeError('trusted pytest launcher requires python -I -S')
    if not TRUSTED_SITE.is_dir():
        raise RuntimeError(f'trusted site-packages missing: {TRUSTED_SITE}')
    # -I removes the script/cwd and environment-controlled paths. Add back only
    # the hidden tests and the image-built package snapshot.
    sys.path.insert(0, str(TRUSTED_SITE))
    sys.path.insert(0, '/tests')
    os.environ.pop('PYTHONPATH', None)
    os.environ.pop('PYTHONHOME', None)
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'


class CtrfPlugin:
    def __init__(self) -> None:
        self.started_ms = int(time.time() * 1000)
        self.tests_collected = 0
        self.collection_errors = 0
        self.states: dict[str, str] = {}
        self.durations_ms: dict[str, float] = {}

    def pytest_collection_finish(self, session) -> None:
        # Count the *selected* tests, after -m/-k deselection has been applied.
        # pytest_collection_modifyitems runs before marker deselection, so
        # len(items) there incorrectly counts image_build_only tests that will
        # not run during per-submission grading.  session.items is the finalized
        # post-deselection collection at this hook.
        self.tests_collected = len(session.items)

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        # At session finish pytest has populated testscollected with the final
        # post-deselection count.  Assign it exactly (rather than taking max):
        # the latter would preserve a stale pre-deselection count.
        self.tests_collected = int(getattr(session, 'testscollected', 0) or 0)

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.collection_errors += 1

    def pytest_runtest_logreport(self, report) -> None:
        nodeid = str(report.nodeid)
        self.durations_ms[nodeid] = self.durations_ms.get(nodeid, 0.0) + float(report.duration) * 1000.0
        previous = self.states.get(nodeid)
        if report.failed:
            self.states[nodeid] = 'failed'
        elif report.skipped and previous != 'failed':
            self.states[nodeid] = 'skipped'
        elif report.when == 'call' and report.passed and previous not in ('failed', 'skipped'):
            self.states[nodeid] = 'passed'

    def write(self, exit_code: int) -> None:
        stop_ms = int(time.time() * 1000)
        passed = sum(v == 'passed' for v in self.states.values())
        failed = sum(v == 'failed' for v in self.states.values())
        skipped = sum(v == 'skipped' for v in self.states.values())
        accounted = passed + failed + skipped
        # A collected test without a final call outcome, a collection error, or
        # an abnormal pytest termination is an explicit non-pass condition.
        other = max(0, self.tests_collected - accounted) + self.collection_errors
        if exit_code != 0 and failed == 0 and other == 0:
            other = 1

        test_rows = []
        for nodeid in sorted(self.states):
            status = self.states[nodeid]
            test_rows.append({
                'name': nodeid,
                'status': status,
                'duration': int(round(self.durations_ms.get(nodeid, 0.0))),
            })

        ctrf = {
            'results': {
                'tool': {'name': 'pytest'},
                'summary': {
                    'tests': self.tests_collected,
                    'passed': passed,
                    'failed': failed,
                    'pending': 0,
                    'skipped': skipped,
                    'other': other,
                    'start': self.started_ms,
                    'stop': stop_ms,
                },
                'tests': test_rows,
            }
        }
        LOG.mkdir(parents=True, exist_ok=True)
        CTRF.write_text(json.dumps(ctrf, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main() -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    CTRF.unlink(missing_ok=True)
    plugin = CtrfPlugin()
    try:
        _install_trusted_import_paths()
        import pytest
        code = int(pytest.main([
            '-q',
            '-p', 'no:cacheprovider',
            '-m', 'not image_build_only',
            '/tests/test_artifact_security.py',
            '/tests/test_security.py',
            '/tests/test_grading_security.py',
            '/tests/test_smoke.py',
            '/tests/test_schema.py',
            '/tests/test_main.py',
        ], plugins=[plugin]))
    except BaseException:
        code = 2
        plugin.collection_errors += 1
    try:
        plugin.write(code)
    except BaseException:
        return 2
    return code


if __name__ == '__main__':
    raise SystemExit(main())
