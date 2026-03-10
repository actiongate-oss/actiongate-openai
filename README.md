# actiongate-openai

Gate OpenAI function/tool calls through [ActionGate](https://github.com/actiongate-oss/actiongate) primitives before execution.

```
pip install actiongate-openai
```

## What it does

Sits between the model's `tool_call` and your function. Checks rate limits, budgets, and policies before the function runs. If any gate blocks, the function never executes.

```
Model → tool_call → GatedToolRunner.call() → your function
                           ↓ (if blocked)
                      GateResult(allowed=False)
```

## Quick start

```python
from actiongate import Engine, Gate, Policy
from actiongate_openai import GatedToolRunner

engine = Engine()
runner = GatedToolRunner(actiongate=engine)

runner.register(
    "search_web", search_fn,
    gate=Gate("tools", "search_web", "agent:1"),
    policy=Policy(max_calls=10, window=60),
)

# In your tool execution loop:
for tool_call in response.choices[0].message.tool_calls:
    result = runner.call_from_openai(tool_call)
    if result.allowed:
        # send result.result back to model
    else:
        # send result.message as error
```

Works with both Chat Completions API and Responses API tool calls.

## Add more gates

Each gate is optional. Install what you need:

```
pip install actiongate-openai[budget]   # + spend control
pip install actiongate-openai[all]      # + spend, policy, audit
```

```python
from actiongate import Engine as AG, Gate, Policy
from budgetgate import Engine as BG, Ledger, Budget
from rulegate import Engine as RG, Rule, Ruleset, Context
from auditgate import Engine as Audit, Trail
from actiongate_openai import GatedToolRunner
from decimal import Decimal

def no_pii(ctx: Context) -> bool:
    return "ssn" not in str(ctx.kwargs).lower()

runner = GatedToolRunner(
    actiongate=AG(),
    budgetgate=BG(),
    rulegate=RG(),
    auditgate=Audit(recorded_by="agent:1"),
)

runner.register(
    "search_web", search_fn,
    # Rate limit: 10 calls per minute
    gate=Gate("tools", "search_web", "agent:1"),
    policy=Policy(max_calls=10, window=60),
    # Budget: $5 per hour
    ledger=Ledger("openai", "search", "agent:1"),
    budget=Budget(max_spend=Decimal("5.00"), window=3600),
    cost=Decimal("0.01"),
    # Policy: no PII in queries
    rule=Rule("tools", "search_web", "agent:1"),
    ruleset=Ruleset(predicates=(no_pii,)),
    # Audit: log every decision
    trail=Trail("tools", "search_web", "agent:1"),
)
```

Gates evaluate in order: ActionGate → BudgetGate → RuleGate → execute → AuditGate. First block stops the pipeline.

## GateResult

Every call returns a `GateResult`:

```python
result = runner.call("search_web", {"query": "latest news"})
result.allowed      # bool
result.result       # function return value (None if blocked)
result.blocked_by   # "actiongate" | "budgetgate" | "rulegate" | None
result.message      # human-readable block reason
result.decisions    # dict of gate decisions for inspection
```

## License

Apache-2.0. ActionGate and BudgetGate are Apache-2.0. RuleGate and AuditGate are BSL-1.1.
