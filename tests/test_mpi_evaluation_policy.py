"""CPU-only checks for the existing MPI selection policy dispatch."""
import ast
from pathlib import Path
import unittest


class EvaluationPolicyTests(unittest.TestCase):
    def test_declared_policy_and_override(self):
        source = Path(__file__).resolve().parents[1] / 'train_3dhp.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in {'_effective_evaluation_policy', '_should_evaluate_epoch'}]
        namespace = {'PER_EPOCH_TEST_MONITORING': False}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
        self.assertFalse(namespace['_should_evaluate_epoch']('final_epoch', False))
        self.assertTrue(namespace['_should_evaluate_epoch']('final_epoch', True))
        self.assertTrue(namespace['_should_evaluate_epoch']('legacy_test_best', False))
        self.assertEqual(namespace['_effective_evaluation_policy']('legacy_test_best'), 'legacy_test_best')
        namespace['PER_EPOCH_TEST_MONITORING'] = True
        self.assertTrue(namespace['_should_evaluate_epoch']('final_epoch', False))


if __name__ == '__main__':
    unittest.main()
