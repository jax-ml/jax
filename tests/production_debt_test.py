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
    "../jax/_src/production_debt.py",
)
spec = importlib.util.spec_from_file_location("jax_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["jax_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtXLAGate = production_debt_mod.ProductionDebtXLAGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtXLAGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtXLAGate(
            never_equate_intent_to_approval=True,
            max_acceptable_jdi=12.0,
        )

    def test_clean_xla_pipeline_passes_readiness(self) -> None:
        report = self.gate.evaluate_xla_pipeline(
            pipeline_id="tpu_v5p_gemini_training_step",
            allocated_sharded_bytes=32000000000,
            peak_device_buffer_bytes=33500000000,
            jit_trace_latency_seconds=0.32,
            recompilation_cycles=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.jdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_xla_pipeline_fails_debt(self) -> None:
        report = self.gate.evaluate_xla_pipeline(
            pipeline_id="uncalibrated_xla_recompilation_run",
            allocated_sharded_bytes=32000000000,
            peak_device_buffer_bytes=84000000000,  # High sharded buffer sprawl (2.6x)
            jit_trace_latency_seconds=3.8,  # High trace latency
            recompilation_cycles=4,  # 4 recompilation cycles
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.jdi_score, 50.0)
        self.assertIn("HIGH_SHARDED_BUFFER_SPRAWL_2.62X", report.critical_smells)
        self.assertIn("HIGH_JIT_TRACE_LATENCY_3.80S", report.critical_smells)
        self.assertIn("DETECTED_4_XLA_RECOMPILATION_CYCLES", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_STATE_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_xla_pipeline("pipeline-1")
        self.gate.evaluate_xla_pipeline("pipeline-2")
        self.gate.evaluate_xla_pipeline("pipeline-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
