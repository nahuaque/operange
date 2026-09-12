# Audit combined uncertainty domains

Use `Product` to combine disjoint input groups and `Union` to declare alternative
operating modes over the same inputs. Both now provide linear support whenever
every factor provides it and every coordinate has an explicit nominal and scale.
Affine process audits and frozen affine-controller audits consume those bounds
automatically.

Use `Intersection(..., backend="cvxpy")` when several convex restrictions must
hold together. This requires the optional `operange[cvxpy]` extra. Intersections
without an explicit backend retain membership-only behavior.

## Combine operating modes and auxiliary demand

Suppose the dryer and evaporator run in one of two permitted modes:

| Mode | Dryer demand | Evaporator demand | Maximum combined demand |
| --- | --- | --- | --- |
| Evaporator intensive | 8–10 MW | 6–8 MW | 18 MW |
| Dryer intensive | 10–12 MW | 4–6 MW | 18 MW |

An auxiliary user adds 0.5–1.5 MW in either mode. A union describes the two modes;
a product permits every pairing of a mode realization and auxiliary demand.
The maximum total is 19.5 MW, so a 20 MW header passes. The unrestricted box
permits both process users to peak together, reaching 21.5 MW with auxiliaries,
and fails the same limit.

```{literalinclude} ../examples/uncertainty_compositions.py
:language: python
:start-at: def operating_modes_example():
:end-before: def convex_examples():
```

Different mode nominals require an explicit common `space`. Factor bounds stay
in their original physical coordinates; the common space chooses the coordinates
used to express the support objective.

```python
model, combined, unrestricted = operating_modes_example()
audit = model.as_claim(combined).audit_result()
failure = model.as_claim(unrestricted).audit_result()
witness = failure.payload.witness.realizations[0]
replay = model.as_claim(unrestricted).evaluate_result(witness)
```

These are different declared operating assumptions. The restricted modes need an
engineering justification; their passing audit does not increase installed
capacity. A product permits every cross-factor combination without asserting
probabilistic independence. Union membership and witnesses retain the original
branches, including gaps between modes.

Run this example with the default dependencies:

```bash
uv run python -m examples.uncertainty_compositions
```

## Apply shared convex restrictions

The same example includes three optional intersection workflows:

| Domain | Checked maximum | Requirement |
| --- | --- | --- |
| Load ellipsoid intersected with physical bounds | About 18.732 MW total steam | At most 18.75 MW |
| Load box with a shared scheduling inequality | 18 MW total steam | At most 18.25 MW |
| Binary feed simplex with a linear composition specification | 0.75 mole fraction A | At most 0.8 |

```bash
uv run --extra cvxpy python -m examples.uncertainty_compositions --convex
```

```{literalinclude} ../examples/uncertainty_compositions.py
:language: python
:start-at: def convex_examples():
:end-before: def run_example(
```

Constraints and support weights use normalized coordinates. In the scheduling
case the nominal loads are 10 and 6 MW, both with a 2 MW scale. The inequality
`z_dryer + z_evaporator <= 1` therefore limits their sum to 18 MW. In the feed
case, A has nominal 0.5 and scale 1; `z_a <= 0.25` means physical A is at most
0.75. The simplex also enforces nonnegativity and a total mole fraction of one.

The [convex-backend guide](convex-backends.md) explains supported intersection
factors, exact witness checks and independently verified support decompositions.
For affine inequalities over a box alone, a single `PolytopeSet` already provides
support without CVXPY; intersections become useful when combining it with other
restrictions such as a simplex or an ellipsoid.

## How product and union bounds are checked

Writing `h_D(w)` for the maximum normalized linear objective over domain `D`:

```text
h_Product(w) = sum_i h_factor_i(w_i)
h_Union(w)   = max_i h_factor_i(w)
```

No joint solver is needed for these identities. Factors may themselves be
products, unions, support-enabled intersections, or finite scenario sets with
explicit normalization. A product joins factor witnesses over disjoint names.
A union selects the actual factor witness with the highest objective value and
retains the largest upper bound across every branch. The branch with the largest
upper bound need not have the best witness when factor bounds are loose.

When normalizations differ, each local coefficient is
`w[name] * local_scale / joint_scale`, with a constant shift from the two
nominals. The implementation computes these transforms as exact rationals of
the declared floats. It bounds any coefficient rounding or underflow with
additional signed coordinate support queries, then rounds the aggregate bounds
outward. Exported evidence includes all factor supports, exact transforms,
corrections and the factors supplying the witness.

Every factor must advertise linear support. An unsupported factor leaves the
composition unsupported; a missing, unresolved or inconsistent support result
leaves the query unresolved. Partial factor coverage cannot establish a passing
audit. Conservative factor bounds can produce a `bounded` result even if their
gap is too wide to establish an optimum.

Compositions keep their existing manifest format and content-derived identities.
Frozen controller replay reconstructs the domain and checks its witnesses.
Loading a declaration does not import CVXPY; executing a nested intersection
support query requires the extra. Native unions can contain convex intersections,
but unions are still unsupported inside the convex intersection compiler.
These support operations do not add continuous audits for freely adjustable
recourse or failure-distance searches over composed domains.
