# Skill: Debugging and Fixing JAX Repro Tests

This skill describes the workflow for fixing tests in `tests/repro_test.py` when they fail due to issues in the test itself or missing support in the repro machinery.

## Workflow

### Phase 1: Verify the Test Itself
Before trying to fix the repro machinery, ensure the test is correct and passes in standard JAX execution.

1.  **Disable Repro Machinery**:
    *   Modify `jax/_src/config.py` to set the default for `jax_repro_dir` to `""`.
    *   Comment out the `repro_is_enabled()` check in the `setUp` method of the test class (usually `ReproTest` in `tests/repro_test.py`).
2.  **Bypass `collect_and_check`**:
    *   In the test function, replace the call to `self.collect_and_check(f, x)` with a direct call `f(x)`.
3.  **Run the Test**:
    *   Run the test using `pytest`.
    *   If it fails, the issue is in the test itself or the JAX API usage, not the repro tracking. Fix the test first.

### Phase 2: Verify Tracking without Replay
Check if the test passes when tracing is enabled but we don't try to replay the generated repro yet.

1.  **Reenable Repro Machinery**:
    *   Revert changes to `jax_repro_dir` in `config.py` and `setUp` in `repro_test.py`.
2.  **Keep Direct Call**:
    *   Keep the direct call `f(x)` instead of `self.collect_and_check`.
3.  **Run the Test**:
    *   If it fails, look at the error messages. If it complains about calling directly into tracing or missing higher-order primitive interception, you may need to annotate new functions with `@traceback_util.api_boundary`.

### Phase 3: Verify Full Repro Cycle
Restore the full check to ensure repros are generated and can be executed correctly.

1.  **Restore `collect_and_check`**:
    *   Replace `f(x)` with `self.collect_and_check(f, x)`.
2.  **Diagnose Failures**:
    *   If the test fails, look for "Undefined" values in the emitted repro file (found in `../repros/` or as configured) or in the logs.
    *   "Undefined" values usually mean the emitter does not know how to serialize a specific type.

## Recipes for Emitters

### Recipe 1: Emitters for Enums
If an enum type is not supported, use `emit_enum` in `jax/_src/repro/emitter.py`:
```python
from path.to.module import MyEnum
_operand_emitter_by_type[MyEnum] = emit_enum("module.MyEnum")
```
