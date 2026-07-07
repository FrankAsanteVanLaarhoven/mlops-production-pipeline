"""Deterministic function-level code reviewer.

Statically audits every module, class, and function under the given paths and
enforces reviewable-code rules that lint alone does not cover:

    R1  public modules, classes, and top-level functions carry a docstring
    R2  every function parameter and return is type-annotated
    R3  cyclomatic complexity stays at or below --max-complexity
    R4  function bodies stay at or below --max-length lines
    R5  no bare except clauses
    R6  no mutable default arguments
    R7  no work-in-progress markers left in source
    R8  nesting depth stays at or below --max-depth

Each function is fingerprinted (SHA-256 of its source segment) and compared to
the previous run's report, so every build shows exactly which functions were
added, changed, or removed — and every changed function is re-audited. The
report is written as JSON for CI artifacts and trend tracking.

Exit status: 0 when the audit is clean, 1 when any rule is violated.

Usage:
    python tools/code_review.py [--paths src tools] [--report artifacts/review/report.json]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Built from fragments so this file does not trip its own R7 rule.
_MARKER_WORDS = ("TO" + "DO", "FIX" + "ME", "X" + "XX")
MARKER_PATTERN = re.compile(r"\b(" + "|".join(_MARKER_WORDS) + r")\b")

# Nodes that open a new decision path (McCabe-style complexity).
_BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
    ast.comprehension,
)
_NESTING_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
_MUTABLE_DEFAULTS = (ast.List, ast.Dict, ast.Set)
_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass
class Finding:
    """A single rule violation, anchored to a file, line, and symbol."""

    rule: str
    symbol: str
    file: str
    line: int
    message: str

    def render(self) -> str:
        """Format the finding for terminal output."""
        return f"  {self.rule}  {self.file}:{self.line}  {self.symbol} — {self.message}"


@dataclass
class FunctionRecord:
    """Audit record for one function: identity, fingerprint, and metrics."""

    symbol: str
    file: str
    line: int
    fingerprint: str
    complexity: int
    length: int
    findings: list[Finding] = field(default_factory=list)


def _complexity(node: _FunctionNode) -> int:
    """McCabe-style cyclomatic complexity: 1 + decision points."""
    score = 1
    for child in ast.walk(node):
        if isinstance(child, _BRANCH_NODES):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += len(child.values) - 1
    return score


def _nesting_depth(node: _FunctionNode) -> int:
    """Deepest chain of nested control-flow statements inside the function."""

    def depth(n: ast.AST, current: int) -> int:
        deepest = current
        for child in ast.iter_child_nodes(n):
            bump = 1 if isinstance(child, _NESTING_NODES) else 0
            deepest = max(deepest, depth(child, current + bump))
        return deepest

    return depth(node, 0)


def _is_public(name: str) -> bool:
    """Dunder methods count as public API surface; single-underscore names do not."""
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def _unannotated_args(args: ast.arguments) -> list[str]:
    """Argument names lacking a type annotation (self/cls receivers exempt)."""
    positional = args.posonlyargs + args.args
    missing = [
        arg.arg
        for i, arg in enumerate(positional)
        if arg.annotation is None and not (i == 0 and arg.arg in ("self", "cls"))
    ]
    missing += [arg.arg for arg in args.kwonlyargs if arg.annotation is None]
    missing += [
        f"*{arg.arg}" for arg in (args.vararg, args.kwarg) if arg and arg.annotation is None
    ]
    return missing


def _annotation_findings(node: _FunctionNode) -> str | None:
    """R2: message describing missing annotations, or None when fully annotated."""
    missing = _unannotated_args(node.args)
    if node.returns is None and node.name != "__init__":
        missing.append("return")
    return f"unannotated: {', '.join(missing)}" if missing else None


def _body_findings(node: _FunctionNode) -> list[tuple[str, str, int]]:
    """R5/R6: (rule, message, line) tuples found inside the function body."""
    found: list[tuple[str, str, int]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.ExceptHandler) and child.type is None:
            found.append(("R5", "bare except clause", child.lineno))
    defaults = node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
    for default in defaults:
        if isinstance(default, _MUTABLE_DEFAULTS):
            found.append(("R6", "mutable default argument", default.lineno))
    return found


def _audit_function(
    node: _FunctionNode,
    qualname: str,
    path: Path,
    source_lines: list[str],
    limits: argparse.Namespace,
) -> FunctionRecord:
    """Apply every function-level rule to one function and record its fingerprint."""
    segment = "\n".join(source_lines[node.lineno - 1 : node.end_lineno])
    record = FunctionRecord(
        symbol=qualname,
        file=str(path),
        line=node.lineno,
        fingerprint=hashlib.sha256(segment.encode()).hexdigest()[:16],
        complexity=_complexity(node),
        length=(node.end_lineno or node.lineno) - node.lineno + 1,
    )
    checks: list[tuple[str, str, int]] = []

    # Nested helpers are implementation detail: exempt from R1, audited otherwise.
    is_nested = "<locals>" in qualname
    if _is_public(node.name) and not is_nested and ast.get_docstring(node) is None:
        checks.append(("R1", "public function has no docstring", node.lineno))
    if message := _annotation_findings(node):
        checks.append(("R2", message, node.lineno))
    if record.complexity > limits.max_complexity:
        message = f"cyclomatic complexity {record.complexity} > {limits.max_complexity}"
        checks.append(("R3", message, node.lineno))
    if record.length > limits.max_length:
        message = f"function spans {record.length} lines > {limits.max_length}"
        checks.append(("R4", message, node.lineno))
    if (depth := _nesting_depth(node)) > limits.max_depth:
        checks.append(("R8", f"nesting depth {depth} > {limits.max_depth}", node.lineno))
    checks.extend(_body_findings(node))

    record.findings = [Finding(rule, qualname, str(path), line, msg) for rule, msg, line in checks]
    return record


def _module_findings(path: Path, tree: ast.Module, source: str) -> list[Finding]:
    """R1/R7 findings that attach to the module rather than a function."""
    findings: list[Finding] = []
    if ast.get_docstring(tree) is None:
        findings.append(Finding("R1", path.stem, str(path), 1, "module has no docstring"))
    for match in MARKER_PATTERN.finditer(source):
        line = source[: match.start()].count("\n") + 1
        findings.append(
            Finding("R7", path.stem, str(path), line, f"{match.group(1)} marker in source")
        )
    return findings


def _audit_module(
    path: Path, limits: argparse.Namespace
) -> tuple[list[FunctionRecord], list[Finding]]:
    """Audit one Python file; return its function records and module-level findings."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    source_lines = source.splitlines()
    findings = _module_findings(path, tree, source)
    records: list[FunctionRecord] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                if _is_public(child.name) and ast.get_docstring(child) is None:
                    findings.append(
                        Finding(
                            "R1",
                            f"{prefix}{child.name}",
                            str(path),
                            child.lineno,
                            "public class has no docstring",
                        )
                    )
                visit(child, f"{prefix}{child.name}.")
            elif isinstance(child, _FunctionNode):
                records.append(
                    _audit_function(child, f"{prefix}{child.name}", path, source_lines, limits)
                )
                visit(child, f"{prefix}{child.name}.<locals>.")

    visit(tree, f"{path.stem}::")
    return records, findings


def _diff_against_previous(
    records: list[FunctionRecord], report_path: Path
) -> dict[str, list[str]]:
    """Compare current fingerprints with the previous report to track function churn."""
    previous: dict[str, str] = {}
    if report_path.exists():
        try:
            payload = json.loads(report_path.read_text())
            previous = {f["symbol"]: f["fingerprint"] for f in payload.get("functions", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            previous = {}
    current = {r.symbol: r.fingerprint for r in records}
    return {
        "added": sorted(s for s in current if s not in previous),
        "changed": sorted(s for s in current if s in previous and current[s] != previous[s]),
        "removed": sorted(s for s in previous if s not in current),
    }


def _write_report(
    report_path: Path,
    paths: list[str],
    files: list[Path],
    records: list[FunctionRecord],
    findings: list[Finding],
    churn: dict[str, list[str]],
) -> None:
    """Persist the machine-readable audit report for CI artifacts and trend tracking."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "paths": paths,
                "files_audited": len(files),
                "functions_audited": len(records),
                "churn_since_last_run": churn,
                "violations": [f.__dict__ for f in findings],
                "functions": [
                    {
                        "symbol": r.symbol,
                        "file": r.file,
                        "line": r.line,
                        "fingerprint": r.fingerprint,
                        "complexity": r.complexity,
                        "length": r.length,
                        "violations": len(r.findings),
                    }
                    for r in records
                ],
            },
            indent=2,
        )
    )


def run_review(paths: list[str], report_path: Path, limits: argparse.Namespace) -> int:
    """Audit all Python files under `paths`; write the JSON report; return exit status."""
    files = sorted({f for p in paths for f in Path(p).rglob("*.py")})
    records: list[FunctionRecord] = []
    findings: list[Finding] = []
    for file in files:
        file_records, file_findings = _audit_module(file, limits)
        records.extend(file_records)
        findings.extend(file_findings)
        findings.extend(f for r in file_records for f in r.findings)

    churn = _diff_against_previous(records, report_path)
    _write_report(report_path, paths, files, records, findings, churn)

    print(
        f"[code-review] {len(files)} files, {len(records)} functions audited "
        f"(+{len(churn['added'])} added, ~{len(churn['changed'])} changed, "
        f"-{len(churn['removed'])} removed since last run)"
    )
    for symbol in churn["changed"]:
        print(f"[code-review]   changed: {symbol}")
    if findings:
        print(f"[code-review] {len(findings)} violation(s):")
        for finding in findings:
            print(finding.render())
        print(f"[code-review] FAIL — report: {report_path}")
        return 1
    print(f"[code-review] PASS — report: {report_path}")
    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", nargs="+", default=["src", "tools"])
    parser.add_argument("--report", default="artifacts/review/report.json")
    parser.add_argument("--max-complexity", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=60)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()
    sys.exit(run_review(args.paths, Path(args.report), args))


if __name__ == "__main__":
    main()
