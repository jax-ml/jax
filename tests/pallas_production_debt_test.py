# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../jax/_src/pallas/production_debt.py",
)
spec = importlib.util.spec_from_file_location("jax_pallas_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["jax_pallas_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtPallasGate = production_debt_mod.ProductionDebtPallasGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtPallasGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtPallasGate(
            never_equate_intent_to_approval=True,
            max_acceptable_pdi=12.0,
        )

    def test_clean_pallas_kernel_passes_readiness(self) -> None:
        report = self.gate.evaluate_pallas_kernel(
            kernel_id="jax_pallas_tpu_v5p_matmul_kernel",
            allocated_scratchpad_bytes=16000000000,
            utilized_scratchpad_bytes=16800000000,
            dispatch_latency_ms=2.4,
            sharded_mesh_stalls=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.pdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_pallas_kernel_fails_debt(self) -> None:
        report = self.gate.evaluate_pallas_kernel(
            kernel_id="uncalibrated_pallas_kernel",
            allocated_scratchpad_bytes=16000000000,
            utilized_scratchpad_bytes=45000000000,  # 2.81x scratchpad memory sprawl
            dispatch_latency_ms=35.0,  # High dispatch latency
            sharded_mesh_stalls=3,  # 3 sharded mesh stalls
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.pdi_score, 50.0)
        self.assertIn("HIGH_SCRATCHPAD_MEMORY_SPRAWL_2.81X", report.critical_smells)
        self.assertIn("HIGH_PALLAS_DISPATCH_LATENCY_35.0MS", report.critical_smells)
        self.assertIn("DETECTED_3_SHARDED_MESH_STALLS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_PALLAS_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_pallas_kernel("kernel-1")
        self.gate.evaluate_pallas_kernel("kernel-2")
        self.gate.evaluate_pallas_kernel("kernel-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
