# Operange

**Find which operating conditions violate your process requirements—and test
the changes that restore them.**

Operange is a Python library for process robustness and operational flexibility.
Declare your model, uncertain conditions, requirements and allowed controls;
get a **pass, fail or inconclusive** verdict with physical results and evidence
of what was checked.

- **Find the limits.** Identify failing conditions, binding constraints and,
  for supported models, the distance to failure.
- **Compare engineering changes.** Re-audit equipment and operating changes
  against the original requirements and uncertainty envelope.
- **Check the operating rule.** Assess permitted adjustments or replay a saved
  controller with explicit observation timing and traceable failures.

## Why this matters

A dryer and an evaporator share a **19 MW steam header**:

| Condition | Demand | Fits? |
| --- | --- | --- |
| Nominal production | 16 MW | Yes |
| Either user at its individual peak | 18 MW | Yes |
| Both users at their peaks | 20 MW | No |

Operange finds the combined overload and returns a failing condition you can
inspect and replay. You can then test a larger header or a scheduling restriction,
while keeping explicit whether the change preserves the original load envelope.

**[Work through the shared-steam tutorial →](docs/index.md)**

## Install

```bash
python -m pip install operange
```

Requires **Python 3.10+**, NumPy and SciPy. See
[installation and development](docs/installation.md)
for checkout and wheel installation.

## Explore

- [Engineering workflow](docs/overview.md) — what
  to declare, how to interpret a verdict, and what makes a comparison meaningful.
- [Worked examples](docs/examples.md) — shared
  utilities, adjustable boilers, storage replay, startup loads and heat integration.
- [API and capabilities](docs/api.md) — supported
  models, uncertainty sets, queries and portable results.
- [Methods and scope](docs/methods.md) — the
  relationship to flexibility analysis, robust optimization and downstream economics.

**Status: alpha.** Supports caller-declared affine and linear models plus bounded
engineering reference models. Broader nonlinear flowsheets and general dynamic
simulation remain outside the current scope.
