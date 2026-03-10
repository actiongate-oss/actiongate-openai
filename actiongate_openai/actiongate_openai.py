"""ActionGate OpenAI Integration — gate agent tool calls before execution.

Wraps OpenAI function/tool calls through ActionGate primitives.
Works with both Chat Completions API and Responses API.

Example (minimal — rate limit only):

    from actiongate_openai import GatedToolRunner
    from actiongate import Engine, Gate, Policy

    ag = Engine()
    runner = GatedToolRunner(actiongate=ag)

    runner.register("search_web", search_fn,
                    gate=Gate("tools", "search_web", "agent:1"),
                    policy=Policy(max_calls=10, window=60))

    # In your tool execution loop:
    result = runner.call("search_web", {"query": "latest news"})

Example (full four-gate pipeline):

    from actiongate import Engine as AG, Gate, Policy
    from budgetgate import Engine as BG, Ledger, Budget
    from rulegate import Engine as RG, Rule, Ruleset, Context
    from auditgate import Engine as AuditEngine, Trail, AuditPolicy, Verdict, Severity

    ag = AG()
    bg = BG()
    rg = RG()
    audit = AuditEngine(recorded_by="agent:1")

    runner = GatedToolRunner(
        actiongate=ag,
        budgetgate=bg,
        rulegate=rg,
        auditgate=audit,
    )

    def no_pii(ctx: Context) -> bool:
        return "ssn" not in str(ctx.kwargs).lower()

    runner.register(
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

    result = runner.call("search_web", {"query": "latest news"})
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from actiongate import Engine as ActionEngine, Gate, Policy
    from budgetgate import Engine as BudgetEngine, Ledger, Budget
    from rulegate import Engine as RuleEngine, Rule, Ruleset
    from auditgate import Engine as AuditEngine, Trail, AuditPolicy


@dataclass
class ToolRegistration:
    """A registered tool with its gate configurations."""
    fn: Callable[..., Any]
    # ActionGate (rate limiting)
    gate: Any | None = None
    policy: Any | None = None
    # BudgetGate (spend control)
    ledger: Any | None = None
    budget: Any | None = None
    cost: Decimal | None = None
    # RuleGate (policy enforcement)
    rule: Any | None = None
    ruleset: Any | None = None
    # AuditGate (logging)
    trail: Any | None = None
    audit_policy: Any | None = None


@dataclass
class GateResult:
    """Result of running a tool through the gate pipeline."""
    allowed: bool
    tool_name: str
    result: Any | None = None
    blocked_by: str | None = None
    message: str | None = None
    decisions: dict[str, Any] = field(default_factory=dict)


class GatedToolRunner:
    """Runs OpenAI tool calls through ActionGate primitives before execution.

    Each gate is optional. Pass only the engines you need:
    - actiongate only: rate limiting
    - actiongate + budgetgate: rate limiting + spend control
    - all four: full governance pipeline

    The runner does NOT modify the OpenAI API flow. It sits between
    the model's tool_call request and your tool execution:

        Model → tool_call → GatedToolRunner.call() → your function
                                   ↓ (if blocked)
                              GateResult(allowed=False)
    """

    __slots__ = ("_tools", "_ag", "_bg", "_rg", "_audit")

    def __init__(
        self,
        actiongate: "ActionEngine | None" = None,
        budgetgate: "BudgetEngine | None" = None,
        rulegate: "RuleEngine | None" = None,
        auditgate: "AuditEngine | None" = None,
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
        gate: "Gate | None" = None,
        policy: "Policy | None" = None,
        ledger: "Ledger | None" = None,
        budget: "Budget | None" = None,
        cost: Decimal | None = None,
        rule: "Rule | None" = None,
        ruleset: "Ruleset | None" = None,
        trail: "Trail | None" = None,
        audit_policy: "AuditPolicy | None" = None,
    ) -> None:
        """Register a tool function with its gate configurations."""
        self._tools[name] = ToolRegistration(
            fn=fn, gate=gate, policy=policy,
            ledger=ledger, budget=budget, cost=cost,
            rule=rule, ruleset=ruleset,
            trail=trail, audit_policy=audit_policy,
        )

    def call(self, name: str, arguments: dict[str, Any] | str) -> GateResult:
        """Execute a tool call through the gate pipeline.

        Args:
            name: Tool function name (must match a registered tool).
            arguments: Tool arguments (dict or JSON string from OpenAI).

        Returns:
            GateResult with allowed=True and result, or allowed=False with block info.
        """
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        reg = self._tools.get(name)
        if reg is None:
            return GateResult(
                allowed=False, tool_name=name,
                blocked_by="registry",
                message=f"Tool not registered: {name}",
            )

        decisions: dict[str, Any] = {}

        # ── ActionGate: rate limiting ──
        if self._ag and reg.gate:
            from actiongate import Status
            decision = self._ag.check(reg.gate, reg.policy)
            decisions["actiongate"] = decision.to_dict()
            if decision.blocked:
                self._audit_decision(reg, "BLOCK", "actiongate", decisions)
                return GateResult(
                    allowed=False, tool_name=name,
                    blocked_by="actiongate",
                    message=decision.message,
                    decisions=decisions,
                )

        # ── BudgetGate: spend control ──
        if self._bg and reg.ledger and reg.cost is not None:
            decision = self._bg.check(reg.ledger, reg.cost, reg.budget)
            decisions["budgetgate"] = decision.to_dict()
            if decision.blocked:
                self._audit_decision(reg, "BLOCK", "budgetgate", decisions)
                return GateResult(
                    allowed=False, tool_name=name,
                    blocked_by="budgetgate",
                    message=decision.message,
                    decisions=decisions,
                )

        # ── RuleGate: policy enforcement ──
        if self._rg and reg.rule and reg.ruleset:
            decision = self._rg.check(
                reg.rule, reg.ruleset,
                kwargs=arguments,
            )
            decisions["rulegate"] = decision.to_dict()
            if decision.blocked:
                self._audit_decision(reg, "BLOCK", "rulegate", decisions)
                return GateResult(
                    allowed=False, tool_name=name,
                    blocked_by="rulegate",
                    message=decision.message,
                    decisions=decisions,
                )

        # ── Execute tool ──
        try:
            result = reg.fn(**arguments)
        except Exception as e:
            self._audit_decision(reg, "ERROR", "execution", decisions, reason=str(e))
            raise

        # ── AuditGate: log success ──
        self._audit_decision(reg, "ALLOW", "pipeline", decisions)

        return GateResult(
            allowed=True, tool_name=name,
            result=result, decisions=decisions,
        )

    def call_from_openai(
        self,
        tool_call: Any,
    ) -> GateResult:
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

    def process_tool_calls(
        self,
        tool_calls: list[Any],
    ) -> list[GateResult]:
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
        """Log to AuditGate if configured."""
        if self._audit is None or reg.trail is None:
            return

        from auditgate import Verdict, Severity

        verdict_map = {"ALLOW": Verdict.ALLOW, "BLOCK": Verdict.BLOCK, "ERROR": Verdict.ERROR}
        severity_map = {"ALLOW": Severity.INFO, "BLOCK": Severity.WARN, "ERROR": Severity.ERROR}

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
            pass  # Audit is fire-and-forget
