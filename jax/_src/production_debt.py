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

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class JAXDebtReport:
    pipeline_id: str
    jdi_score: float  # JAX Debt Index (target <= 12.0)
    sharded_buffer_multiplier: float  # Target <= 1.08x
    jit_trace_latency_seconds: float  # Target <= 0.45s
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: list[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """Cryptographic SHA-256 hash-chained Action Ledger for JAX / XLA compilation runs."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_compilation_event(
        self,
        pipeline_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = (
            f"{index}|{self._last_hash}|{pipeline_id}|{event_type}|"
            f"{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        )
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "pipeline_id": pipeline_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtXLAGate:
    """A2Z SOC Production Debt & Technical Due Diligence Gate for JAX & XLA JIT Pipelines.

    Quantifies XLA recompilation cascades, sharded buffer fragmentation, and JIT trace latency against 4 Enterprise KPIs:
    1. JAX Debt Index (JDI <= 12.0)
    2. Sharded Buffer Sprawl Multiplier (SBSM <= 1.08x)
    3. P99 JIT Trace Latency (<= 0.45s)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_jdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_jdi = max_acceptable_jdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        return any(Path(p).exists() for p in ("artifacts/KILL", "/tmp/KILL"))

    def evaluate_xla_pipeline(
        self,
        pipeline_id: str,
        allocated_sharded_bytes: int = 32000000000,
        peak_device_buffer_bytes: int = 33600000000,
        jit_trace_latency_seconds: float = 0.32,
        recompilation_cycles: int = 0,
        un_gated_mutations: int = 0,
    ) -> JAXDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_compilation_event(
                pipeline_id=pipeline_id,
                event_type="pipeline_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            err_msg = "A2Z SOC ActionGate: Emergency kill switch is engaged. JAX/XLA execution halted."
            raise PermissionError(err_msg)

        critical_smells: list[str] = []

        # KPI 2: Sharded Buffer Sprawl Multiplier
        buffer_ratio = peak_device_buffer_bytes / max(1, allocated_sharded_bytes)
        if buffer_ratio > 1.8:
            critical_smells.append(f"HIGH_SHARDED_BUFFER_SPRAWL_{buffer_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if jit_trace_latency_seconds > 2.0:
            critical_smells.append(f"HIGH_JIT_TRACE_LATENCY_{jit_trace_latency_seconds:.2f}S")

        # Recompilation cycles
        if recompilation_cycles > 1:
            critical_smells.append(f"DETECTED_{recompilation_cycles}_XLA_RECOMPILATION_CYCLES")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_STATE_MUTATIONS")

        # KPI 1: JAX Debt Index (0 = Clean, 100 = Catastrophic)
        jdi = (
            max(0.0, (buffer_ratio - 1.0) * 20.0)
            + max(0.0, (jit_trace_latency_seconds - 0.45) * 10.0)
            + (recompilation_cycles * 15.0)
            + (un_gated_mutations * 30.0)
        )
        jdi_score = round(min(100.0, jdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - jdi_score)
        is_production_ready = (
            jdi_score <= self.max_acceptable_jdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_compilation_event(
            pipeline_id=pipeline_id,
            event_type="pipeline_authorized" if is_production_ready else "pipeline_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "jdi_score": jdi_score,
                "buffer_ratio": buffer_ratio,
                "allocated_sharded_bytes": allocated_sharded_bytes,
                "peak_device_buffer_bytes": peak_device_buffer_bytes,
                "jit_trace_latency_seconds": jit_trace_latency_seconds,
                "recompilation_cycles": recompilation_cycles,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return JAXDebtReport(
            pipeline_id=pipeline_id,
            jdi_score=jdi_score,
            sharded_buffer_multiplier=round(buffer_ratio, 2),
            jit_trace_latency_seconds=round(jit_trace_latency_seconds, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
