"""Finite heat-integration target screening, without an installed-network claim."""

from dataclasses import replace
import json

from operange import FiniteSet, Scenario
from operange.reference import HeatCascadeAdapter, SensibleHeatStream


def example():
    model = HeatCascadeAdapter(
        "Four-stream heat-integration targets",
        streams=tuple(
            SensibleHeatStream(name, kind, supply, target, cp, "Synthetic steady state")
            for name, kind, supply, target, cp in (
                ("hot_product", "hot", 180, 60, 2),
                ("hot_effluent", "hot", 150, 30, 1),
                ("cold_feed", "cold", 20, 140, 2),
                ("cold_wash", "cold", 80, 170, 1),
            )
        ),
        delta_t_min_k=20,
        hot_utility_capacity_kw=50,
        cold_utility_capacity_kw=50,
    )
    less_hot_flow = {"hot_product:heat_capacity_flow_kw_per_k": 1.5}
    more_cold_flow = {"cold_feed:heat_capacity_flow_kw_per_k": 2.5}
    domain = FiniteSet(
        model.input_space,
        tuple(
            Scenario(
                name, {**model.input_space.nominal, **changes}, "Synthetic finite case"
            )
            for name, changes in (
                ("nominal", {}),
                ("less_hot_flow", less_hot_flow),
                ("more_cold_flow", more_cold_flow),
                ("combined", {**less_hot_flow, **more_cold_flow}),
            )
        ),
    )
    return model, domain


def run_example():
    model, domain = example()
    claim = model.as_claim(domain)
    results = {
        "nominal": claim.evaluate_result(domain.scenario("nominal").values),
        "target_audit": claim.audit_result(),
        "enlarged_utility_target_audit": replace(model, hot_utility_capacity_kw=100)
        .as_claim(domain)
        .audit_result(),
    }
    return {name: result.to_dict() for name, result in results.items()}


def main():
    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
