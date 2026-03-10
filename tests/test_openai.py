"""Tests for actiongate-openai integration."""

from __future__ import annotations

import sys
import time
import traceback
from decimal import Decimal
from dataclasses import dataclass

sys.path.insert(0, ".")

passed = 0
failed = 0
errors: list[str] = []

def test(name):
    def decorator(fn):
        global passed, failed
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            errors.append(f"  FAIL  {name}: {e}\n{traceback.format_exc()}")
            print(f"  FAIL  {name}: {e}")
        return fn
    return decorator

_clk = 1.0
def fake_clock():
    return _clk


print("── ActionGate only ──")

@test("allowed call returns result")
def _():
    from actiongate import Engine, Gate, Policy
    from actiongate_openai import GatedToolRunner

    ag = Engine(clock=fake_clock)
    runner = GatedToolRunner(actiongate=ag)
    runner.register("add", lambda a, b: a + b,
                    gate=Gate("test", "add"),
                    policy=Policy(max_calls=5, window=60))
    r = runner.call("add", {"a": 2, "b": 3})
    assert r.allowed and r.result == 5


@test("rate limited call is blocked")
def _():
    global _clk
    _clk = 1.0
    from actiongate import Engine, Gate, Policy
    from actiongate_openai import GatedToolRunner

    ag = Engine(clock=fake_clock)
    runner = GatedToolRunner(actiongate=ag)
    runner.register("ping", lambda: "pong",
                    gate=Gate("test", "ping"),
                    policy=Policy(max_calls=1, window=60))
    r1 = runner.call("ping", {})
    assert r1.allowed

    _clk = 2.0
    r2 = runner.call("ping", {})
    assert not r2.allowed
    assert r2.blocked_by == "actiongate"


@test("unregistered tool is blocked")
def _():
    from actiongate_openai import GatedToolRunner
    runner = GatedToolRunner()
    r = runner.call("nonexistent", {})
    assert not r.allowed
    assert r.blocked_by == "registry"


@test("json string arguments are parsed")
def _():
    from actiongate_openai import GatedToolRunner
    runner = GatedToolRunner()
    runner.register("echo", lambda msg: msg)
    r = runner.call("echo", '{"msg": "hello"}')
    assert r.allowed and r.result == "hello"


print("\n── With BudgetGate ──")

@test("budget exceeded blocks call")
def _():
    global _clk
    _clk = 1.0
    from actiongate import Engine as AG, Gate, Policy
    from budgetgate import Engine as BG, Ledger, Budget
    from actiongate_openai import GatedToolRunner

    ag = AG(clock=fake_clock)
    bg = BG(clock=fake_clock)
    runner = GatedToolRunner(actiongate=ag, budgetgate=bg)
    runner.register("expensive", lambda: "done",
                    gate=Gate("test", "expensive"),
                    policy=Policy(max_calls=100, window=60),
                    ledger=Ledger("test", "api", "user:1"),
                    budget=Budget(max_spend=Decimal("1.00"), window=60),
                    cost=Decimal("0.60"))

    r1 = runner.call("expensive", {})
    assert r1.allowed

    _clk = 2.0
    r2 = runner.call("expensive", {})
    assert not r2.allowed
    assert r2.blocked_by == "budgetgate"


print("\n── With RuleGate ──")

@test("policy violation blocks call")
def _():
    global _clk
    _clk = 1.0
    from actiongate import Engine as AG, Gate, Policy
    from rulegate import Engine as RG, Rule, Ruleset, Context
    from actiongate_openai import GatedToolRunner

    def no_bad_words(ctx: Context) -> bool:
        return "bad" not in str(ctx.kwargs.get("query", ""))

    ag = AG(clock=fake_clock)
    rg = RG(clock=fake_clock)
    runner = GatedToolRunner(actiongate=ag, rulegate=rg)
    runner.register("search", lambda query: f"results for {query}",
                    gate=Gate("test", "search"),
                    policy=Policy(max_calls=100, window=60),
                    rule=Rule("test", "search"),
                    ruleset=Ruleset(predicates=(no_bad_words,)))

    r1 = runner.call("search", {"query": "good stuff"})
    assert r1.allowed

    r2 = runner.call("search", {"query": "bad stuff"})
    assert not r2.allowed
    assert r2.blocked_by == "rulegate"


print("\n── OpenAI format ──")

@test("call_from_openai with Chat Completions format")
def _():
    from actiongate_openai import GatedToolRunner

    @dataclass
    class Function:
        name: str
        arguments: str
    @dataclass
    class ToolCall:
        function: Function

    runner = GatedToolRunner()
    runner.register("greet", lambda name: f"hello {name}")

    tc = ToolCall(function=Function(name="greet", arguments='{"name": "world"}'))
    r = runner.call_from_openai(tc)
    assert r.allowed and r.result == "hello world"


@test("call_from_openai with Responses API format")
def _():
    from actiongate_openai import GatedToolRunner

    @dataclass
    class ResponseToolCall:
        name: str
        arguments: str
        type: str = "function_call"

    runner = GatedToolRunner()
    runner.register("greet", lambda name: f"hello {name}")

    tc = ResponseToolCall(name="greet", arguments='{"name": "world"}')
    r = runner.call_from_openai(tc)
    assert r.allowed and r.result == "hello world"


@test("process_tool_calls handles multiple calls")
def _():
    from actiongate_openai import GatedToolRunner

    @dataclass
    class Function:
        name: str
        arguments: str
    @dataclass
    class ToolCall:
        function: Function

    runner = GatedToolRunner()
    runner.register("add", lambda a, b: a + b)
    runner.register("mul", lambda a, b: a * b)

    calls = [
        ToolCall(function=Function(name="add", arguments='{"a": 1, "b": 2}')),
        ToolCall(function=Function(name="mul", arguments='{"a": 3, "b": 4}')),
    ]
    results = runner.process_tool_calls(calls)
    assert len(results) == 2
    assert results[0].result == 3
    assert results[1].result == 12


@test("decisions dict contains gate outputs")
def _():
    global _clk
    _clk = 1.0
    from actiongate import Engine, Gate, Policy
    from actiongate_openai import GatedToolRunner

    ag = Engine(clock=fake_clock)
    runner = GatedToolRunner(actiongate=ag)
    runner.register("ping", lambda: "pong",
                    gate=Gate("test", "ping"),
                    policy=Policy(max_calls=5, window=60))
    r = runner.call("ping", {})
    assert "actiongate" in r.decisions
    assert r.decisions["actiongate"]["status"] == "ALLOW"


print(f"\n{'═' * 50}")
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print("\nFailures:")
    for e in errors:
        print(e)
print(f"{'═' * 50}")
sys.exit(1 if failed else 0)
