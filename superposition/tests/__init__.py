"""
Test suite for superposition analysis package.

All tests use unittest framework for consistency.
"""

import unittest
from pathlib import Path


def load_tests(loader, standard_tests, pattern):
    """Load all test suites (unittest discovery protocol)."""
    suite = unittest.TestSuite()

    # Find all test modules
    test_dir = Path(__file__).parent
    for test_file in test_dir.glob('test_*.py'):
        if test_file.name != '__init__.py':
            module_name = f'superposition.tests.{test_file.stem}'
            suite.addTests(loader.loadTestsFromName(module_name))

    return suite


def run_all_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    suite = load_tests()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)