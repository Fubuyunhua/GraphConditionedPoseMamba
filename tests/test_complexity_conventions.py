"""Arithmetic tests for the read-only complexity audit, no model execution."""
import importlib.util
from pathlib import Path
import unittest

p=Path(__file__).resolve().parents[1]/'scripts/audit_complexity_conventions.py'
s=importlib.util.spec_from_file_location('complexity_audit',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class CostTests(unittest.TestCase):
    def test_broadcast_batch(self):
        self.assertEqual(m.matrix_macs((17,17),(243,17,32),(243,17,32)),2247264)
    def test_mm(self):
        self.assertEqual(m.matrix_macs((2,3),(3,4),(2,4)),24)
    def test_vector(self):
        self.assertEqual(m.matrix_macs((3,),(3,),()),3)
    def test_scan_units(self):
        c=m.scan_cost(1,2,3,4)
        self.assertEqual(c['mac_equivalent'],114)
        self.assertEqual(c['old_mixed_count'],222)
        self.assertEqual(2*c['mac_equivalent'],c['parallel_core_flops']+c['skip_flops'])


if __name__=='__main__':unittest.main()
