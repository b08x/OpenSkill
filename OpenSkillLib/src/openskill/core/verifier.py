"""
verifier.py — EvoSkills: Surrogate Verifier & Sandbox Execution
================================================================
FIXES:
  1. diagnostic never empty again — captures stdout + stderr + returncode
  2. Isolated sandbox with clean sys.path to avoid import side-effects
  3. Timeout with clear message
  4. Robust wrapping: skill code + tests in separate namespace
  5. Automatic main function detection for asserts
"""

from __future__ import annotations

import sys
import os
import textwrap
import tempfile
import subprocess
import structlog
from typing import Tuple

from openskill.llm.base import BaseLLMProvider, LLMMessage

log = structlog.get_logger()

VERIFIER_SYSTEM_PROMPT = (
    "You are an independent, isolated Surrogate Verifier for an AI system.\n"
    "Your ONLY job is to write deterministic Python `assert` tests to verify if a "
    "solution correctly solves a given task.\n"
    "Do NOT write the solution. ONLY write the test cases.\n"
    "IMPORTANT: Assume the solution function is already defined in the namespace.\n"
    "           Do NOT import it. Do NOT use 'from X import Y'.\n"
    "           Write ONLY assert statements and helper variables.\n"
    "Output ONLY valid Python code wrapped in ```python ... ``` blocks.\n"
    "Example output:\n"
    "```python\n"
    "assert fibonacci(0) == 0\n"
    "assert fibonacci(1) == 1\n"
    "assert fibonacci(10) == 55\n"
    "```"
)

VERIFIER_TEST_TEMPLATE = (
    "TASK:\n{task}\n\n"
    "Write Python assert statements to test the solution.\n"
    "RULES:\n"
    "- Do NOT import anything\n"
    "- Do NOT define functions\n"
    "- ONLY write assert statements\n"
    "- Test at least 5 cases including edge cases (0, 1, small, medium values)\n"
    "- The main function name is likely: {func_hint}\n"
    "Example:\n"
    "assert {func_hint}(0) == 0\n"
    "assert {func_hint}(1) == 1\n"
)

# Template of the full script that runs in the subprocess
SANDBOX_TEMPLATE = '''\
import sys
import os

# Isolates sandbox from main project
_project_paths = [p for p in sys.path if "OpenSkill" in p or "openskill" in p.lower()]
for _p in _project_paths:
    try:
        sys.path.remove(_p)
    except ValueError:
        pass

# ── SKILL CODE ────────────────────────────────────────────────
{skill_code}

# ── TEST CODE ─────────────────────────────────────────────────
_test_passed = 0
_test_failed = 0
_errors = []

{indented_tests}

if _test_failed == 0:
    print(f"SANDBOX_OK: {{_test_passed}} tests passed")
else:
    print(f"SANDBOX_FAIL: {{_test_failed}} failed, {{_test_passed}} passed")
    for e in _errors:
        print(f"  ERROR: {{e}}")
    sys.exit(1)
'''

# Wrapper for each assert — captures failures individually
ASSERT_WRAPPER = '''\
try:
    {assert_line}
    _test_passed += 1
except Exception as _e:
    _test_failed += 1
    _errors.append(f"{assert_line!r} → {{_e}}")
'''


def _extract_func_hint(skill_code: str) -> str:
    """Attempts to guess the main function name in the code."""
    import re
    # Searches for 'def name(' in code
    matches = re.findall(r'def\s+(\w+)\s*\(', skill_code)
    if not matches:
        return "solution"
    # Prefers non-helper names (not starting with _)
    public = [m for m in matches if not m.startswith("_")]
    if public:
        # Prefers task-related names
        for name in public:
            if any(kw in name.lower() for kw in ["fibonacci", "fib", "solution", "calc", "compute"]):
                return name
        return public[0]
    return matches[0]


def _wrap_asserts(test_code: str) -> str:
    """Wraps each assert line in try/except for individual diagnostics."""
    lines = []
    for line in test_code.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("assert ") or stripped.startswith("assert("):
            wrapped = ASSERT_WRAPPER.format(assert_line=stripped)
            lines.append(textwrap.indent(wrapped, ""))
        elif stripped and not stripped.startswith("#"):
            # Lines that are not asserts (e.g., auxiliary variables) — kept directly
            lines.append(line)
    return "\n".join(lines) if lines else test_code


class SurrogateVerifier:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    async def generate_tests(self, task: str, skill_code: str = "") -> str:
        """
        Generates the test script (verifier test suite V).
        Uses skill code to infer the main function name.
        """
        func_hint = _extract_func_hint(skill_code) if skill_code else "solution"

        messages = [
            LLMMessage(role="system", content=VERIFIER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=VERIFIER_TEST_TEMPLATE.format(
                task=task,
                func_hint=func_hint,
            )),
        ]

        resp = await self.llm.generate(messages, max_tokens=800, temperature=0.1)
        test_code = self._extract_python_code(resp.content)

        log.debug("verifier.tests_generated", func_hint=func_hint, lines=len(test_code.split("\n")))
        return test_code

    def evaluate_in_sandbox(self, skill_code: str, test_code: str) -> Tuple[bool, str]:
        """
        Executes skill_code + test_code in an isolated subprocess.

        Returns:
            (success: bool, diagnostic: str)
            diagnostic is ALWAYS non-empty on failure.
        """
        if not skill_code or not skill_code.strip():
            return False, "DIAGNOSTIC: Skill code is empty — no code was generated."

        if not test_code or not test_code.strip():
            return False, "DIAGNOSTIC: Test code is empty — no test was generated."

        # Wraps asserts in try/except for granular diagnostics
        wrapped_tests = _wrap_asserts(test_code)

        # Assembles the complete script
        script = SANDBOX_TEMPLATE.format(
            skill_code=skill_code,
            indented_tests=wrapped_tests,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=15,
                # Runs in temp directory to avoid accidental project imports
                cwd=tempfile.gettempdir(),
                env={**os.environ, "PYTHONPATH": ""},  # clears PYTHONPATH
            )

            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()

            log.debug(
                "verifier.sandbox_result",
                returncode=result.returncode,
                stdout=stdout[:200],
                stderr=stderr[:200],
            )

            if result.returncode == 0 and "SANDBOX_OK" in stdout:
                return True, "Passed"

            # Assembles detailed diagnostics — NEVER empty
            parts = []
            if stdout:
                parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                parts.append(f"STDERR:\n{stderr}")
            if not parts:
                parts.append(
                    f"DIAGNOSTIC: Process returned code {result.returncode} with no output.\n"
                    "Possible causes:\n"
                    "  1. SyntaxError in generated code\n"
                    "  2. Uninstalled module import\n"
                    "  3. Indentation error\n"
                    f"Executed script:\n{script[:500]}..."
                )

            diagnostic = "\n".join(parts)
            return False, diagnostic

        except subprocess.TimeoutExpired:
            return False, (
                "DIAGNOSTIC: Timeout (15s) — code likely has an infinite loop.\n"
                "Check if the recursion stop condition is correct."
            )
        except Exception as e:
            return False, f"DIAGNOSTIC: Error executing sandbox: {type(e).__name__}: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _extract_python_code(self, text: str) -> str:
        """Extracts only the code block from markdown."""
        import re
        m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback: returns full text if no markdown block exists
        return text.strip()
