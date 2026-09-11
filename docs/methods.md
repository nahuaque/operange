# Methods, scope and design lineage

Operange binds process models, uncertainty and operating permissions to explicit
engineering evidence. This guide explains how that workflow relates to established
methods, downstream economic studies and the project’s origins.

## Robustness, flexibility and operator adjustments

Here, **robustness** means satisfying the declared requirements throughout the
chosen uncertainty envelope, under the stated operating permissions.
**Operational flexibility** concerns the range of conditions the design can
accommodate by adjusting operation. This follows the established process-design
meaning of flexibility; Swaney and Grossmann formalized an index measuring a
feasible steady-state parameter region with adjustable controls.
([Swaney and Grossmann, 1985](https://doi.org/10.1002/aic.690310412))

The operating permissions matter as much as the equipment:

| Operating assumption | Engineering meaning | Current example |
| --- | --- | --- |
| Fixed operation | The declared control settings remain fixed while uncertain inputs vary. | Affine utility model |
| Adjustment after observation, also called recourse | Operators can choose a permitted setting after observing the current disturbance. | Static heat-recovery model |
| Causal operation | Decisions can use information already available, with earlier decisions shared between futures that have not yet become distinguishable. | Two-period thermal storage |

For storage, preparing for a demand surge may require charging, while preparing
for incoming surplus heat may require leaving space. Each future can be feasible
when planned with advance knowledge, yet there may be no preparation decision
that accommodates both before the event is known. The storage reference model
can expose that information conflict and compare it with perfect foresight.

Feasible steady-state adjustment alone does not establish that a real controller
can reach the operating point, remain stable, or satisfy constraints during a
transient. Those questions require the corresponding dynamic model and evidence.
The current storage model addresses only its declared two-period horizon.

## How this relates to familiar methods

The contribution is a reusable way to declare the engineering question, run
supported methods and preserve their evidence. It builds on established
sensitivity, flexibility and robust-optimization ideas.

| Familiar approach | What it tells you | How `operange` relates |
| --- | --- | --- |
| Nominal simulation and design margins | Whether the design works at specified conditions, with an allowance for variation. | Makes the variation envelope explicit and checks which combinations exhaust the available margin. The supplied physical model still determines the answer. |
| Parameter sweeps and one-at-a-time sensitivity | How responses change at selected points or along selected directions. | Returns named local derivatives and separately audits requirements over a supported domain. A local derivative or a sweep alone does not establish a full-domain pass. |
| Monte Carlo uncertainty analysis | Output distributions and estimated violation frequencies under assumed input distributions. | The current audits ask whether admissible conditions can violate a requirement, without assigning their probability. A set-based robustness margin is not a failure probability; no failures in a finite sample alone is not an exhaustive proof. See [Monte Carlo uncertainty analysis](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=929780). |
| Global sensitivity analysis, such as Sobol indices | How inputs and their interactions contribute to output variance under the analysis distribution. | Local derivatives and constraint-failure evidence answer different questions. The package does not currently compute Sobol indices; a tool such as [SALib](https://salib.readthedocs.io/en/latest/user_guide/basics.html) can complement it. |
| Classical flexibility analysis | Whether operating adjustments can maintain feasibility over a parameter region, and how large that region can be. | This is a direct mathematical foundation. The package adds explicit model, uncertainty and operating declarations, typed results and repeatable engineering-change audits; it does not claim to invent a new flexibility index. |
| Robust optimization | Find decisions that satisfy constraints across an uncertainty set, potentially with adjustable controls. | Uses the same concepts, with an initial focus on auditing supplied designs. General robust design optimization is not part of the supported API. Existing tools such as [PyROS](https://pyomo.readthedocs.io/en/stable/explanation/solvers/pyros/index.html) solve broader nonlinear two-stage problems; there is currently no PyROS integration. |
| Model predictive control, including robust MPC | Choose control moves repeatedly using a dynamic model, measurements and a prediction horizon. | Makes operating permissions and information timing part of an audit. It does not implement an online controller or replace a controller framework such as [do-mpc](https://www.do-mpc.com/en/latest/theory_mpc.html). |

Different tools can therefore be useful in the same study: sensitivity analysis
to understand responses, probabilistic analysis to estimate frequency, and a
robustness audit to establish the limits of a stated engineering commitment.
A narrower uncertainty envelope can make a robustness claim easier to satisfy;
its physical justification remains part of the study.

## Where techno-economic analysis fits

For a TEA, this package supplies the engineering side: physical outputs,
constraint evidence, sensitivities and the consequences of engineering changes.
Your downstream model converts those outputs into energy bills, installed costs,
annual cash flows or other economic measures. Pricing, investment selection,
financial attribution and presentation belong to that downstream code.

Hidden-variable recomposition can still be useful in the **economics reporting**
part of a TEA, using `updatesupport` and, where its financial interfaces fit,
[`updatesupport-finance`](https://github.com/nahuaque/updatesupport/blob/08b3652bb6f8e9c7437b4329632c6e6583e9b96a/packages/updatesupport-finance/README.md).
For example, an annual unit-cost summary might hide a different mix of operating
hours, load levels or tariff periods within the same reported categories. With
retained detailed rows and a suitable declared metric, a recomposition audit can
ask how sensitive that aggregate is to the hidden mix. This is a possible
application of those APIs, not a built-in plant-TEA integration.

| Part of the study | Question | Responsibility |
| --- | --- | --- |
| Process engineering | Can the design meet its requirements across the specified physical conditions and operating permissions? | `operange` and the supplied engineering model |
| Economic calculation | What are those physical outcomes worth under the chosen cost and financial assumptions? | Your TEA/economic model |
| Aggregate economic reporting | Could an unreported change in the retained operating mix alter a reported economic conclusion? | `updatesupport`, with suitable `updatesupport-finance` interfaces and caller-supplied metrics |

Reweighting precomputed economic rows does not re-solve a coupled plant or its
operating schedule. If a changed mix changes the physical response, the downstream
study must obtain consistent engineering evaluations. `updatesupport-finance`
currently focuses on financial model-risk reporting and disclosure analysis;
it is not a ready-made plant cash-flow model. Neither sibling package is a
runtime dependency of `operange`.

## Design lineage

Operange originated as `updatesupport-process` in the
[`updatesupport` repository](https://github.com/nahuaque/updatesupport/tree/08b3652bb6f8e9c7437b4329632c6e6583e9b96a/packages/updatesupport-process).
The extraction retains the MIT license and the `process_result/v1` artifact
format. It has its own release version and no compatibility import alias.


The uncertainty-set vocabulary also draws on [ROmodel, by Wiebe and Misener](https://doi.org/10.1007/s11081-021-09703-2),
which separates uncertain parameters, uncertainty sets and adjustable variables
in robust-optimization models. The implementation does not depend on ROmodel
or copy its code. Classical flexibility analysis and the related tools linked
above are foundations and complements, not capabilities claimed as new here.
