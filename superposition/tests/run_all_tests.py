#!/usr/bin/env python3
"""
Run all superposition tests.

Usage:
    python run_all_tests.py              # Run all tests
    python run_all_tests.py -v          # Run with minimal verbosity
    python run_all_tests.py test_base   # Run specific test module
"""

import unittest
import sys
import argparse
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Run superposition tests')
    parser.add_argument('test', nargs='?', default='discover',
                       help='Specific test module to run (e.g., test_base)')
    parser.add_argument('-v', '--verbosity', type=int, default=2, choices=[0, 1, 2],
                       help='Test output verbosity (0=quiet, 1=normal, 2=verbose)')
    parser.add_argument('--failfast', action='store_true',
                       help='Stop on first failure')

    args = parser.parse_args()

    # Create test loader
    loader = unittest.TestLoader()

    # Load tests
    if args.test == 'discover':
        # Discover all tests
        suite = loader.discover('superposition.tests', pattern='test_*.py')
        print(f"Running all superposition tests...")
    else:
        # Load specific test module
        test_module = args.test if args.test.startswith('test_') else f'test_{args.test}'
        try:
            suite = loader.loadTestsFromName(f'superposition.tests.{test_module}')
            print(f"Running tests from {test_module}...")
        except Exception as e:
            print(f"Error loading test module '{test_module}': {e}")
            print("\nAvailable test modules:")
            test_dir = Path(__file__).parent
            for test_file in sorted(test_dir.glob('test_*.py')):
                print(f"  - {test_file.stem}")
            return 1

    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=args.verbosity,
        failfast=args.failfast
    )

    result = runner.run(suite)

    # Return exit code based on success
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())