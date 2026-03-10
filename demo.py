"""Demo: ActionGate OpenAI integration — full four-gate pipeline.

Simulates an agent making tool calls through rate limiting,
spend control, policy enforcement, and audit logging.

Run:
    pip install actiongate-openai[all]
    python demo.py

No OpenAI API key needed — uses simulated tool calls.
"""

from decimal import Decimal
from dataclasses import dataclass

from actiongate import Engine as AG, Gate, Policy
from budgetgate import Engine as BG, Ledger, Budget
from rulegate import Engine as RG, Rule, Ruleset, Context
from auditgate import Engine as Audit, Trail, Verdict, Severity
from actiongate_openai import GatedToolRunner


# ── Simulated tools ──

def search_web(query: str) -> str:
    return f"Results for: {query}"

def send_email(to: str, body: str) -> str:
    return f"Email sent to {to}"

def query_database(sql: str) -> str:
    return f"3 rows returned"


# ── Policy predicates ──

def no_pii(ctx: Context) -> bool:
    """Block queries containing PII markers."""
    text = str(ctx.kwargs).lower()
    return not any(w in text for w in ("ssn", "password", "credit card"))

def no_drop_statements(ctx: Context) -> bool:
    """Block destructive SQL."""
    sql = str(ctx.kwargs.get("sql", "")).lower()
    return "drop" not in sql and "delete" not in sql


# ── Build the pipeline ──

runner = GatedToolRunner(
    actiongate=AG(),
    budgetgate=BG(),
    rulegate=RG(),
    auditgate=Audit(recorded_by="demo-agent"),
)

# Search: 3 calls/min, $0.01 each, no PII allowed
runner.register(
    "search_web", search_web,
    gate=Gate("tools", "search_web", "agent:demo"),
    policy=Policy(max_calls=5, window=60),
    ledger=Ledger("api", "search", "agent:demo"),
    budget=Budget(max_spend=Decimal("0.05"), window=3600),
    cost=Decimal("0.01"),
    rule=Rule("tools", "search_web"),
    ruleset=Ruleset(predicates=(no_pii,)),
    trail=Trail("tools", "search_web", "agent:demo"),
)

# Email: 2 calls/min, $0.005 each
runner.register(
    "send_email", send_email,
    gate=Gate("tools", "send_email", "agent:demo"),
    policy=Policy(max_calls=2, window=60),
    ledger=Ledger("api", "email", "agent:demo"),
    budget=Budget(max_spend=Decimal("0.10"), window=3600),
    cost=Decimal("0.005"),
    trail=Trail("tools", "send_email", "agent:demo"),
)

# Database: no destructive SQL
runner.register(
    "query_database", query_database,
    gate=Gate("tools", "query_database", "agent:demo"),
    policy=Policy(max_calls=10, window=60),
    rule=Rule("tools", "query_database"),
    ruleset=Ruleset(predicates=(no_drop_statements,)),
    trail=Trail("tools", "query_database", "agent:demo"),
)


# ── Simulate OpenAI tool calls ──

@dataclass
class Function:
    name: str
    arguments: str

@dataclass
class ToolCall:
    function: Function


print("=" * 60)
print("ActionGate OpenAI Demo — Four-Gate Pipeline")
print("=" * 60)

scenarios = [
    ("Normal search",        ToolCall(Function("search_web", '{"query": "AI news"}'))),
    ("PII search (policy)",  ToolCall(Function("search_web", '{"query": "find user SSN"}'))),
    ("Second search",        ToolCall(Function("search_web", '{"query": "Python tips"}'))),
    ("Third search",         ToolCall(Function("search_web", '{"query": "weather"}'))),
    ("Fourth search",        ToolCall(Function("search_web", '{"query": "recipes"}'))),
    ("Fifth search (rate)",  ToolCall(Function("search_web", '{"query": "one too many"}'))),
    ("Send email",           ToolCall(Function("send_email", '{"to": "boss@co.com", "body": "report"}'))),
    ("Safe SQL query",       ToolCall(Function("query_database", '{"sql": "SELECT * FROM users"}'))),
    ("DROP table (policy)",  ToolCall(Function("query_database", '{"sql": "DROP TABLE users"}'))),
    ("Unknown tool",         ToolCall(Function("hack_mainframe", '{}'))),
]

for label, tc in scenarios:
    result = runner.call_from_openai(tc)
    status = "ALLOW" if result.allowed else "BLOCK"
    detail = result.result if result.allowed else f"[{result.blocked_by}] {result.message}"
    print(f"\n  {label}")
    print(f"    {status}: {detail}")

print(f"\n{'=' * 60}")
print("Demo complete. Four gates evaluated on every tool call.")
print("=" * 60)
