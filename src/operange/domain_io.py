"""Reconstruct supported declarative domain manifests without executing models."""

from collections.abc import Mapping
import json

from .composition import Intersection, Product, Union
from .domains import FiniteSet, ParameterSpace
from .convex_hull import ConvexHullSet
from .geometries import BudgetSet, EllipsoidSet, SimplexSet
from .polytope import PolytopeSet
from .primitives import BoxSet, Parameter
from .vector import VectorParameter


def domain_from_manifest(manifest):
    if not isinstance(manifest, Mapping):
        raise ValueError("domain manifest must be a mapping")
    data = dict(manifest)
    kind = data.pop("kind", None)
    try:
        if kind == "box":
            if set(data) != {"parameters"}:
                raise ValueError("invalid box manifest fields")
            parameters = []
            for item in data["parameters"]:
                parameter = dict(item)
                tag = parameter.pop("kind", "scalar")
                if tag not in ("scalar", "vector"):
                    raise ValueError("unknown parameter kind")
                parameters.append(
                    (VectorParameter if tag == "vector" else Parameter)(**parameter)
                )
            return BoxSet(tuple(parameters))
        if "space" in data:
            data["space"] = ParameterSpace(**data["space"])
        if kind == "finite_set":
            return FiniteSet(**data)
        if kind == "convex_hull":
            return ConvexHullSet(**data)
        if kind in ("simplex", "ellipsoid"):
            return (SimplexSet if kind == "simplex" else EllipsoidSet)(**data)
        if kind in ("budget", "polytope"):
            data["envelope"] = domain_from_manifest(data["envelope"])
            if type(data["envelope"]) is not BoxSet:
                raise ValueError("geometry envelope must be a BoxSet")
            return (BudgetSet if kind == "budget" else PolytopeSet)(**data)
        if kind in ("intersection", "union", "product"):
            data["factors"] = tuple(domain_from_manifest(f) for f in data["factors"])
            return {"intersection": Intersection, "union": Union, "product": Product}[
                kind
            ](**data)
    except (TypeError, KeyError) as exc:
        raise ValueError("invalid domain manifest fields") from exc
    raise ValueError(f"unsupported domain kind: {kind}")


def domain_from_json(document):
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError("duplicate JSON key")
            result[name] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    return domain_from_manifest(
        json.loads(document, object_pairs_hook=pairs, parse_constant=constant)
    )
