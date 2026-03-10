"""ActionGate OpenAI demo — no API key needed.

    pip install actiongate-openai[all]
    python demo.py
"""

from decimal import Decimal
from actiongate import Engine as AG, Gate, Policy
from budgetgate import Engine as BG, Ledger, Budget
from rulegate import Engine as RG, Rule, Ruleset, Context
from auditgate import Engine as Audit, Trail
from actiongate_openai import GatedToolRunner

# ── Tools ──

def search_web(query: str) -> str:
    return f"Results for: {query}"

def send_email(to: str, body: str) -> str:
    return f"Sent to {to}"

def query_db(sql: str) -> str:
    return "3 rows"

# ── Policies ──

def no_pii(ctx: Context) -> bool:
    return "ssn" not in str(ctx.kwargs).lower()

def no_drop(ctx: Context) -> bool:
    return "drop" not in str(ctx.kwargs.get("sql", "")).lower()

# ── Pipeline ──

runner = GatedToolRunner(
    actiongate=AG(), budgetgate=BG(), rulegate=RG(),
    auditgate=Audit(recorded_by="demo"),
)

runner.register("search_web", search_web,
    gate=Gate("tools", "search", "agent:1"),
    policy=Policy(max_calls=5, window=60),
    ledger=Ledger("api", "search", "agent:1"),
    budget=Budget(max_spend=Decimal("0.05"), window=3600),
    cost=Decimal("0.01"),
    rule=Rule("tools", "search"),
    ruleset=Ruleset(predicates=(no_pii,)),
    trail=Trail("tools", "search", "agent:1"))

runner.register("send_email", send_email,
    gate=Gate("tools", "email", "agent:1"),
    policy=Policy(max_calls=2, window=60),
    trail=Trail("tools", "email", "agent:1"))

runner.register("query_db", query_db,
    gate=Gate("tools", "db", "agent:1"),
    policy=Policy(max_calls=10, window=60),
    rule=Rule("tools", "db"),
    ruleset=Ruleset(predicates=(no_drop,)),
    trail=Trail("tools", "db", "agent:1"))

# ── Run ──

calls = [
    ("Search",           "search_web", {"query": "AI news"}),
    ("PII blocked",      "search_web", {"query": "find SSN for user"}),
    ("Search again",     "search_web", {"query": "Python tips"}),
    ("Search again",     "search_web", {"query": "weather"}),
    ("Search again",     "search_web", {"query": "recipes"}),
    ("Rate limited",     "search_web", {"query": "one too many"}),
    ("Email",            "send_email", {"to": "boss@co.com", "body": "done"}),
    ("Safe SQL",         "query_db",   {"sql": "SELECT * FROM users"}),
    ("DROP blocked",     "query_db",   {"sql": "DROP TABLE users"}),
    ("Unknown tool",     "hack_it",    {}),
]

print("ActionGate OpenAI Demo")
print("─" * 50)
for label, tool, args in calls:
    r = runner.call(tool, args)
    status = "ALLOW" if r.allowed else f"BLOCK ({r.blocked_by})"
    detail = r.result if r.allowed else r.message
    print(f"  {label:20s} {status:25s} {detail}")
