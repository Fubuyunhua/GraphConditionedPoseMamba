"""Portable recipe and launcher checks: no datasets, CUDA or training."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_registry_and_recipes(self):
        rows = json.loads((ROOT / 'configs/gcs_pose/registry.json').read_text())
        self.assertEqual(len(rows), 6)
        for row in rows:
            c = yaml.safe_load((ROOT / row['config']).read_text())
            self.assertNotIn('_base_', c)
            self.assertEqual((c['epochs'], c['batch_size'], c['clip_len']), (80, 4, 243))
            self.assertEqual(c['gt_2d'], row['protocol'] == 'gt2d')
            self.assertFalse(c['finetune'] or c['resume'] or c['pretrained'])
            self.assertEqual(c['graph_injection_mode'], 'control')
            self.assertEqual(c['dim_feat'], int(row['model'].split('d')[0][1:]))
            self.assertEqual(c['depth'], int(row['model'].split('d')[1]))

    def test_help_does_not_import_training(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce_gcspose.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--run', result.stdout)


if __name__ == '__main__':
    unittest.main()
