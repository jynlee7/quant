"""AST scans for the rules import-linter can't express (ARCHITECTURE §3 and §13).

- K5: the approved-order types are constructed only in `quant.risk.engine`, and `TrialTicket`
  only in `quant.research.ledger`.
- K6: `datetime.now`, `time.time` and every other read of the wall clock appear only in
  `quant.core.clock`.
- I2: nothing outside `quant.eval` calls a smoother or Viterbi (`predict`, `predict_proba`,
  `decode`, `score_samples`, `quant.eval.smoother`), and only `quant.hmm.train` imports hmmlearn.
- K4, the parts import-linter can't see: only the adapter imports `alpaca.trading`, and no
  sibling inside `quant.execution` imports the adapter.

Each `*_violations` function takes one module's source and returns (line, message) pairs, one per
offending line. The scans resolve import aliases, so `from time import time as t; t()` is caught.
They catch the ways code is normally written, not deliberate evasion through `getattr` chains or
`importlib`; the runtime tokens (I4, A2) and the truncation tests (I1) stand behind them.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator

from quant_check.findings import Finding
from quant_check.repo import Repo

Violation = tuple[int, str]
Check = Callable[..., list[Violation]]

# K5: each guarded type, and the one module allowed to construct it.
CONSTRUCTED_ONLY_IN = {
    "RiskApprovedOrder": "quant.risk.engine",
    "RiskApprovedCancel": "quant.risk.engine",
    "RiskApprovedFlatten": "quant.risk.engine",
    "RiskApprovedStop": "quant.risk.engine",
    "TrialTicket": "quant.research.ledger",
}
# Ways to get an instance without calling the class: object protocol and pydantic.
_ALTERNATE_CONSTRUCTORS = frozenset(
    {
        "__call__",
        "__init__",
        "__new__",
        "construct",
        "model_construct",
        "model_copy",
        "model_validate",
        "model_validate_json",
        "model_validate_strings",
        "parse_obj",
        "parse_raw",
    }
)
# Ways to copy an existing instance, possibly with changes. Flagged when the copied name is
# annotated with a guarded type. The runtime token and one-time approval ids (B9) back this up.
_COPY_METHODS = _ALTERNATE_CONSTRUCTORS | {"__copy__", "__deepcopy__", "__replace__", "copy"}
_COPY_FUNCTIONS = frozenset({"copy.copy", "copy.deepcopy", "copy.replace", "dataclasses.replace"})

# K6: reads of the wall clock, by the dotted name they resolve to.
CLOCK_MODULE = "quant.core.clock"
_CLOCK_READS = frozenset(
    {
        "datetime.date.today",
        "datetime.datetime.now",
        "datetime.datetime.today",
        "datetime.datetime.utcnow",
        "pandas.Timestamp.now",
        "pandas.Timestamp.today",
        "pandas.Timestamp.utcnow",
        "os.times",
        "time.asctime",
        "time.clock_gettime",
        "time.clock_gettime_ns",
        "time.ctime",
        "time.gmtime",
        "time.localtime",
        "time.monotonic",
        "time.monotonic_ns",
        "time.perf_counter",
        "time.perf_counter_ns",
        "time.process_time",
        "time.process_time_ns",
        "time.strftime",
        "time.thread_time",
        "time.thread_time_ns",
        "time.time",
        "time.time_ns",
        "uuid.uuid1",
        "uuid.uuid6",
        "uuid.uuid7",
    }
)
# An event loop's `time()` is a clock too.
_LOOP_GETTERS = frozenset(
    {
        "asyncio.events.get_event_loop",
        "asyncio.events.get_running_loop",
        "asyncio.get_event_loop",
        "asyncio.get_running_loop",
        "asyncio.new_event_loop",
    }
)
# Constructors that read the clock when handed the string "now" or "today".
_CLOCK_LITERAL_CONSTRUCTORS = frozenset(
    {"numpy.datetime64", "pandas.Timestamp", "pandas.to_datetime"}
)
_CLOCK_LITERALS = frozenset({"now", "today", "utcnow"})

# I2: smoothed posteriors and Viterbi paths stay out of everything but eval/.
EVAL_PACKAGE = "quant.eval"
SMOOTHER_MODULE = "quant.eval.smoother"
TRAIN_MODULE = "quant.hmm.train"
_SMOOTHER_METHODS = frozenset({"predict", "predict_proba", "score_samples"})

# K4: alpaca-py's trading client belongs to the adapter alone, and inside quant.execution no
# sibling may import the adapter (import-linter treats imports within a source package as fine).
EXECUTION_PACKAGE = "quant.execution"
TRADING_ADAPTER = "quant.execution.alpaca_broker"
TRADING_SDK = "alpaca.trading"


def _within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _package_of(module: str, is_package: bool) -> str:
    return module if is_package else module.rpartition(".")[0]


def _from_base(node: ast.ImportFrom, package: str) -> str | None:
    """The absolute module a `from ... import` statement reads from."""
    if node.level == 0:
        return node.module
    parts = package.split(".") if package else []
    if node.level - 1 > len(parts):
        return None
    base = parts[: len(parts) - (node.level - 1)]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base) or None


def _imported_names(tree: ast.Module, package: str) -> Iterator[tuple[int, str]]:
    """(line, dotted name) for every module and name an import statement brings in."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            base = _from_base(node, package)
            if base is None:
                continue
            yield node.lineno, base
            for alias in node.names:
                if alias.name != "*":
                    yield node.lineno, f"{base}.{alias.name}"


class _Scope:
    """Resolves the names a module binds through imports to the dotted names they stand for."""

    def __init__(self, tree: ast.Module, package: str) -> None:
        self._bound: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self._bound[alias.asname] = alias.name
                    else:
                        top = alias.name.partition(".")[0]
                        self._bound[top] = top
            elif isinstance(node, ast.ImportFrom):
                base = _from_base(node, package)
                if base is None:
                    continue
                for alias in node.names:
                    if alias.name != "*":
                        self._bound[alias.asname or alias.name] = f"{base}.{alias.name}"

    def qualify(self, node: ast.expr) -> str | None:
        """The dotted name of a Name or Attribute chain, its root resolved through imports."""
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        return ".".join([self._bound.get(node.id, node.id), *reversed(parts)])


class _Hits:
    """At most one violation per line, so one offending expression is reported once."""

    def __init__(self) -> None:
        self._by_line: dict[int, str] = {}

    def add(self, line: int, message: str) -> None:
        self._by_line.setdefault(line, message)

    def sorted(self) -> list[Violation]:
        return sorted(self._by_line.items())


def _parse(source: str, module: str, is_package: bool) -> tuple[ast.Module, _Scope, str]:
    tree = ast.parse(source)
    package = _package_of(module, is_package)
    return tree, _Scope(tree, package), package


def _is_str_in(node: ast.expr, values: frozenset[str]) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in values


def _annotation_names(annotation: ast.expr) -> Iterator[ast.expr]:
    """The names inside an annotation, including quoted ones like `"RiskApprovedOrder"`."""
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name | ast.Attribute):
            yield node
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                yield from _annotation_names(ast.parse(node.value, mode="eval").body)
            except SyntaxError:
                continue


def k5_violations(source: str, module: str, *, is_package: bool = False) -> list[Violation]:
    tree, scope, _ = _parse(source, module, is_package)
    hits = _Hits()

    def guarded(expr: ast.expr) -> str | None:
        name = scope.qualify(expr)
        if name is None:
            return None
        last = name.rpartition(".")[2]
        return last if CONSTRUCTED_ONLY_IN.get(last, module) != module else None

    # Names annotated with a guarded type: parameters and annotated assignments.
    instances: dict[str, str] = {}
    for node in ast.walk(tree):
        target, annotation = None, None
        if isinstance(node, ast.arg) and node.annotation is not None:
            target, annotation = node.arg, node.annotation
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, annotation = node.target.id, node.annotation
        if target and annotation:
            for name in _annotation_names(annotation):
                if kind := guarded(name):
                    instances[target] = kind

    def instance(expr: ast.expr) -> str | None:
        return instances.get(expr.id) if isinstance(expr, ast.Name) else None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            copied = None
            if isinstance(func, ast.Attribute) and func.attr in _COPY_METHODS:
                copied = instance(func.value)
            if copied is None and scope.qualify(func) in _COPY_FUNCTIONS:
                copied = next(
                    (k for arg in node.args if (k := instance(arg) or guarded(arg))), None
                )
            if copied:
                home = CONSTRUCTED_ONLY_IN[copied]
                hits.add(node.lineno, f"copies a {copied} outside {home}")
                continue
            target = guarded(func)
            if (
                target is None
                and isinstance(func, ast.Attribute)
                and func.attr in _ALTERNATE_CONSTRUCTORS
            ):
                target = guarded(func.value) or next(
                    (g for arg in node.args if (g := guarded(arg))), None
                )
            if target:
                home = CONSTRUCTED_ONLY_IN[target]
                hits.add(node.lineno, f"{target} is constructed outside {home}")
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                if target := guarded(base):
                    home = CONSTRUCTED_ONLY_IN[target]
                    hits.add(node.lineno, f"{target} is subclassed outside {home}")
    return hits.sorted()


def k6_violations(source: str, module: str, *, is_package: bool = False) -> list[Violation]:
    if module == CLOCK_MODULE:
        return []
    tree, scope, _ = _parse(source, module, is_package)
    hits = _Hits()
    advice = "take the time from the Clock port in quant.core.clock"

    def is_loop(expr: ast.expr) -> bool:
        if isinstance(expr, ast.Call):
            return scope.qualify(expr.func) in _LOOP_GETTERS
        return isinstance(expr, ast.Name) and expr.id in loops

    loops: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and is_loop(node.value):
            loops.update(t.id for t in node.targets if isinstance(t, ast.Name))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "time"
            and is_loop(node.func.value)
        ):
            hits.add(node.lineno, f"an event loop's time() reads the clock; {advice}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name | ast.Attribute) and isinstance(node.ctx, ast.Load):
            name = scope.qualify(node)
            if name in _CLOCK_READS:
                hits.add(node.lineno, f"{name} reads the wall clock; {advice}")
        elif isinstance(node, ast.Call):
            name = scope.qualify(node.func)
            args = [*node.args, *(kw.value for kw in node.keywords)]
            if name in _CLOCK_LITERAL_CONSTRUCTORS and any(
                _is_str_in(a, _CLOCK_LITERALS) for a in args
            ):
                hits.add(node.lineno, f"{name}('now') reads the wall clock; {advice}")
    return hits.sorted()


def _looks_like_viterbi(call: ast.Call) -> bool:
    """`bytes.decode` takes an encoding string. hmmlearn's `decode` takes an array, and maybe
    `lengths` and `algorithm`. A non-string positional argument means the second."""
    if any(kw.arg in ("lengths", "algorithm") for kw in call.keywords):
        return True
    return any(not (isinstance(a, ast.Constant) and isinstance(a.value, str)) for a in call.args)


def i2_violations(source: str, module: str, *, is_package: bool = False) -> list[Violation]:
    tree, scope, package = _parse(source, module, is_package)
    hits = _Hits()
    for line, imported in _imported_names(tree, package):
        if module != TRAIN_MODULE and _within(imported, "hmmlearn"):
            hits.add(line, f"imports hmmlearn; only {TRAIN_MODULE} may, and it calls only fit")
    if _within(module, EVAL_PACKAGE):
        return hits.sorted()  # eval/ is where smoothing is allowed to run
    for line, imported in _imported_names(tree, package):
        if _within(imported, SMOOTHER_MODULE):
            hits.add(line, f"imports {SMOOTHER_MODULE}; smoothing runs only in eval/")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in _SMOOTHER_METHODS:
                hits.add(
                    node.lineno,
                    f".{node.attr} gives smoothed posteriors or a Viterbi path; "
                    "decisions use our forward filter",
                )
            elif (name := scope.qualify(node)) and _within(name, SMOOTHER_MODULE):
                hits.add(node.lineno, f"uses {SMOOTHER_MODULE}; smoothing runs only in eval/")
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "decode"
                and _looks_like_viterbi(node)
            ):
                hits.add(
                    node.lineno, ".decode(X) is Viterbi decoding; decisions use the forward filter"
                )
            elif (
                scope.qualify(func) == "getattr"
                and len(node.args) >= 2
                and _is_str_in(node.args[1], _SMOOTHER_METHODS | {"decode"})
            ):
                hits.add(node.lineno, "getattr reaches a smoother or Viterbi method")
    return hits.sorted()


def k4_violations(source: str, module: str, *, is_package: bool = False) -> list[Violation]:
    if module == TRADING_ADAPTER:
        return []
    tree, scope, package = _parse(source, module, is_package)
    hits = _Hits()
    message = f"uses {TRADING_SDK}; only {TRADING_ADAPTER} talks to the broker"
    sibling = f"imports {TRADING_ADAPTER}; inside execution only the engine's wiring may (K4)"
    for line, imported in _imported_names(tree, package):
        if _within(imported, TRADING_SDK):
            hits.add(line, message)
        elif _within(module, EXECUTION_PACKAGE) and _within(imported, TRADING_ADAPTER):
            hits.add(line, sibling)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = scope.qualify(node)
            if name and _within(name, TRADING_SDK):
                hits.add(node.lineno, message)
    return hits.sorted()


def scan(repo: Repo, rule: str, check: Check) -> list[Finding]:
    """Run one per-module check over every module under src/quant."""
    findings = []
    for module, path in repo.module_files():
        where = repo.relative(path)
        try:
            violations = check(path.read_text(), module, is_package=path.name == "__init__.py")
        except SyntaxError as exc:
            findings.append(Finding(rule, f"{where}:{exc.lineno or 0}", f"can't parse: {exc.msg}"))
            continue
        findings.extend(Finding(rule, f"{where}:{line}", message) for line, message in violations)
    return findings


def scan_k5(repo: Repo) -> list[Finding]:
    return scan(repo, "K5", k5_violations)


def scan_k6(repo: Repo) -> list[Finding]:
    return scan(repo, "K6", k6_violations)


def scan_i2(repo: Repo) -> list[Finding]:
    return scan(repo, "I2", i2_violations)
