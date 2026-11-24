"""Print how many tests pass."""

import io
import unittest

suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
print("tests passing:", result.testsRun - len(result.failures) - len(result.errors))
