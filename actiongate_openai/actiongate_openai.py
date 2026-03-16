"""ActionGate OpenAI Integration — gate agent tool calls before execution.

Sits between the model's tool_call and your function. Evaluates rate limits,
budgets, and policies before the function runs. If any gate blocks, the
function never executes.

Works with both Chat Completions API and Responses API.

Example (minimal — rate limit only):

    from actiongate_openai import ToolGate
    from actiongate import Engine, Gate, Policy

    ag = Engine()
    tg = ToolGate(actiongate=ag)

    tg.register("search_web", search_fn,
                gate=Gate("tools", "search_web", "agent:1"),
                policy=Policy(max_calls=10, window=60))

    # In your tool execution loop:
    result = tg.call("search_web", {"query": "latest news"})

Example (full four-gate pipeline):

    from actiongate import Engine as AG, Gate, Policy
    from budgetgate import Engine as BG, Ledger, Budget
    from rulegate import Engine as RG, Rule, Ruleset, Context
    from auditgate import Engine as AuditEngine, Trail, AuditPolicy

    tg = ToolGate(
        actiongate=AG(),
        budgetgate=BG(),
        rulegate=RG(),
        auditgate=AuditEngine(recorded_by="agent:1"),
    )

    def no_pii(ctx: Context) -> bool:
        return "ssn" not in str(ctx.kwargs).lower()

    tg.register(
        "search_web", search_fn,
        gate=Gate("tools", "search_web", "agent:1"),
        policy=Policy(max_calls=10, window=60),
        ledger=Ledger("openai", "search", "agent:1"),
        budget=Budget(max_spend=Decimal("5.00")),
        cost=Decimal("0.01"),
        rule=Rule("tools", "search_web", "agent:1"),
        ruleset=Ruleset(predicates=(no_pii,)),
        trail=Trail("tools", "search_web", "agent:1"),
    )

    result = tg.call("search_web", {"query": "latest news"})
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from actiongate import Engine as ActionEngine, Gate, Policy
    from budgetgate import Engine as BudgetEngine, Ledger, Budget
    from rulegate import Engine as RuleEngine, Rule, Ruleset
    from auditgate import Engine as AuditEngine, Trail, AuditPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """A registered tool with its gate configurations."""

    fn: Callable[..., Any]
    # ActionGate (rate limiting)
    gate: Gate | None = None
    policy: Policy | None = None
    # BudgetGate (spend control)
    ledger: Ledger | None = None
    budget: Budget | None = None
    cost: Decimal | None = None
    # RuleGate (policy enforcement)
    rule: Rule | None = None
    ruleset: Ruleset | None = None
    # AuditGate (audit logging)
    trail: Trail | None = None
    audit_policy: AuditPolicy | None = None


@dataclass(frozen=True, slots=True)
class GateResult:
    """Result of running a tool call through the gate pipeline."""

    allowed: bool
    tool_name: str
    result: Any | None = None
    blocked_by: str | None = None
    message: str | None = None
    decisions: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.allowed

    def __repr__(self) -> str:
        if self.allowed:
            return f"GateResult(allowed=True, tool={self.tool_name!r})"
        return (
            f"GateResult(allowed=False, tool={self.tool_name!r}, "
            f"blocked_by={self.blocked_by!r})"
        )


def _blocked(
    tool_name: str,
    blocked_by: str,
    message: str,
    decisions: dict[str, Any],
) -> GateResult:
    """Construct a blocked result. Single path for all block returns."""
    return GateResult(
        allowed=False,
        tool_name=tool_name,
        blocked_by=blocked_by,
        message=message,
        decisions=decisions,
    )


class ToolGate:
    """Gates OpenAI tool calls through ActionGate primitives before execution.

    Each gate is optional. Pass only the engines you need:
    - actiongate only: rate limiting
    - actiongate + budgetgate: rate limiting + spend control
    - all four: full execution control pipeline

    ToolGate does NOT modify the OpenAI API flow. It sits between
    the model's tool_call request and your tool execution:

        OpenAI tool_call
              ↓
        ToolGate.call()
              ↓
        ActionGate → BudgetGate → RuleGate  (any block → GateResult(allowed=False))
              ↓
        your function executes
              ↓
        AuditGate logs outcome
              ↓
        GateResult(allowed=True, result=...)
    """

    __slots__ = ("_tools", "_ag", "_bg", "_rg", "_audit")

    def __init__(
        self,
        actiongate: ActionEngine | None = None,
        budgetgate: BudgetEngine | None = None,
        rulegate: RuleEngine | None = None,
        auditgate: AuditEngine | None = None,
    ) -> None:
        self._tools: dict[str, ToolRegistration] = {}
        self._ag = actiongate
        self._bg = budgetgate
        self._rg = rulegate
        self._audit = auditgate

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        gate: Gate | None = None,
        policy: Policy | None = None,
        ledger: Ledger | None = None,
        budget: Budget | None = None,
        cost: Decimal | None = None,
        rule: Rule | None = None,
        ruleset: Ruleset | None = None,
        trail: Trail | None = None,
        audit_policy: AuditPolicy | None = None,
    ) -> None:
        """Register a tool function with its gate configurations.

        Raises:
            ValueError: If gate is provided without policy, or ledger
                without budget/cost, or rule without ruleset.
        """
        if gate is not None and policy is None:
            raise ValueError(f"Tool {name!r}: gate requires policy")
        if ledger is not None and (budget is None or cost is None):
            raise ValueError(f"Tool {name!r}: ledger requires budget and cost")
        if rule is not None and ruleset is None:
            raise ValueError(f"Tool {name!r}: rule requires ruleset")

        self._tools[name] = ToolRegistration(
            fn=fn,
            gate=gate, policy=policy,
            ledger=ledger, budget=budget, cost=cost,
            rule=rule, ruleset=ruleset,
            trail=trail, audit_policy=audit_policy,
        )

    def call(self, name: str, arguments: dict[str, Any] | str) -> GateResult:
        """Execute a tool call through the gate pipeline.

        Args:
            name: Tool function name (must match a registered tool).
            arguments: Tool arguments as dict, or JSON string from OpenAI.

        Returns:
            GateResult with allowed=True and result, or
            allowed=False with block info.
        """
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                return _blocked(name, "parse", f"Invalid JSON arguments: {e}", {})

        reg = self._tools.get(name)
        if reg is None:
            return _blocked(name, "registry", f"Tool not registered: {name}", {})

        decisions: dict[str, Any] = {}

        # ── ActionGate: rate limiting ──
        if self._ag is not None and reg.gate is not None:
            decision = self._ag.check(reg.gate, reg.policy)
            decisions["actiongate"] = decision.to_dict()
            if decision.blocked:
                self._audit_decision(reg, "BLOCK", "actiongate", decisions)
                return _blocked(name, "actiongate", decision.message, decisions)

        # ── BudgetGate: spend control ──
        if self._bg is not None and reg.ledger is not None:
            decision = self._bg.check(reg.ledger, reg.cost, reg.budget)
            decisions["budgetgate"] = decision.to_dict()
            if decision.blocked:
                self._audit_decision(reg, "BLOCK", "budgetgate", decisions)
                return _blocked(name, "budgetgate", decision.message, decisions)

        # ── RuleGate: policy enforcement ──
        if self._rg is not None and reg.rule is not None:
            decision = self._rg.check(reg.rule, reg.ruleset, kwargs=arguments)
            decisions["rulegate"] = decision.to_dict()
            if decision.blocked:
                self._audit_decision(reg, "BLOCK", "rulegate", decisions)
                return _blocked(name, "rulegate", decision.message, decisions)

        # ── Execute tool ──
        try:
            result = reg.fn(**arguments)
        except Exception as e:
            self._audit_decision(
                reg, "ERROR", "execution", decisions, reason=str(e),
            )
            raise

        # ── AuditGate: log success ──
        self._audit_decision(reg, "ALLOW", "pipeline", decisions)

        return GateResult(
            allowed=True,
            tool_name=name,
            result=result,
            decisions=decisions,
        )

    def call_from_openai(self, tool_call: Any) -> GateResult:
        """Execute a tool call directly from an OpenAI API response.

        Works with both Chat Completions and Responses API tool calls.

        Chat Completions API:
            tool_call.function.name, tool_call.function.arguments

        Responses API:
            tool_call.name, tool_call.arguments (when type='function_call')
        """
        # Chat Completions API format
        if hasattr(tool_call, "function"):
            name = tool_call.function.name
            args = tool_call.function.arguments
        # Responses API format
        elif hasattr(tool_call, "name") and hasattr(tool_call, "arguments"):
            name = tool_call.name
            args = tool_call.arguments
        else:
            raise ValueError(f"Unrecognized tool_call format: {type(tool_call)}")

        return self.call(name, args)

    def process_tool_calls(self, tool_calls: list[Any]) -> list[GateResult]:
        """Process multiple tool calls from a single model response.

        Returns a list of GateResults in the same order as tool_calls.
        """
        return [self.call_from_openai(tc) for tc in tool_calls]

    def _audit_decision(
        self,
        reg: ToolRegistration,
        verdict_str: str,
        gate_type: str,
        decisions: dict[str, Any],
        reason: str | None = None,
    ) -> None:
        """Log to AuditGate if configured. Never raises."""
        if self._audit is None or reg.trail is None:
            return

        from auditgate import Verdict, Severity

        verdict_map = {
            "ALLOW": Verdict.ALLOW,
            "BLOCK": Verdict.BLOCK,
            "ERROR": Verdict.ERROR,
        }
        severity_map = {
            "ALLOW": Severity.INFO,
            "BLOCK": Severity.WARN,
            "ERROR": Severity.ERROR,
        }

        try:
            self._audit.record(
                reg.trail,
                verdict=verdict_map.get(verdict_str, Verdict.ERROR),
                severity=severity_map.get(verdict_str, Severity.ERROR),
                gate_type=gate_type,
                gate_identity=str(reg.trail),
                reason=reason or f"{verdict_str} by {gate_type}",
                detail=decisions,
                policy=reg.audit_policy,
            )
        except Exception:
            logger.debug("Audit write failed for %s/%s", reg.trail, gate_type,
                         exc_info=True)
