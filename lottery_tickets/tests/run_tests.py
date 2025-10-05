#!/usr/bin/env python
"""
Test runner for lottery tickets test suite.
============================================
Run all tests or specific test modules.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --module utils     # Run only utils tests
    python run_tests.py --verbose          # Verbose output
    python run_tests.py --failfast        # Stop on first failure
"""

import sys
import unittest
import argparse
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def discover_tests(pattern='test*.py'):
    """Discover all test cases."""
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    return loader.discover(start_dir, pattern=pattern)


def run_specific_module(module_name):
    """Run tests from a specific module."""
    loader = unittest.TestLoader()

    module_map = {
        'ggn': 'test_ggn_verification',
        'importance': 'test_importance_scoring',
        'pruning': 'test_magnitude_pruning',
        'memory': 'test_memory_fixes',
        'utils': 'test_utils'
    }

    if module_name in module_map:
        module_file = module_map[module_name]
    else:
        module_file = f'test_{module_name}'

    try:
        # Import the module
        test_module = __import__(module_file)
        return loader.loadTestsFromModule(test_module)
    except ImportError as e:
        print(f"Error importing module {module_file}: {e}")
        return unittest.TestSuite()


def run_tests(args):
    """Run the test suite."""
    # Configure runner
    runner_kwargs = {
        'verbosity': 2 if args.verbose else 1,
        'failfast': args.failfast
    }

    runner = unittest.TextTestRunner(**runner_kwargs)

    # Get test suite
    if args.module:
        suite = run_specific_module(args.module)
        if suite.countTestCases() == 0:
            print(f"No tests found for module: {args.module}")
            return False
    else:
        suite = discover_tests()

    # Run tests
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")

        if result.failures:
            print("\nFailed tests:")
            for test, traceback in result.failures:
                print(f"  - {test}")

        if result.errors:
            print("\nTests with errors:")
            for test, traceback in result.errors:
                print(f"  - {test}")

        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run lottery tickets test suite'
    )

    parser.add_argument(
        '--module', '-m',
        help='Run tests from specific module (ggn, importance, pruning, memory, utils)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose test output'
    )

    parser.add_argument(
        '--failfast', '-f',
        action='store_true',
        help='Stop on first test failure'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available test modules'
    )

    args = parser.parse_args()

    if args.list:
        print("Available test modules:")
        print("  - ggn: GGN theoretical verification tests")
        print("  - importance: Importance scoring tests")
        print("  - pruning: Magnitude pruning tests")
        print("  - memory: Memory leak fixes tests (ICML 2026)")
        print("  - utils: Utility function tests")
        return

    # Run tests
    success = run_tests(args)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()