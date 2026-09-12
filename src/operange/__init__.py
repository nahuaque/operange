"""Process-engineering sensitivity and robustness DSL with portable evidence."""

from .affine import AffineOutput, AffineProcessAdapter, AffineRequirement, AffineTerm
from .linear_process import LinearControl, LinearProcessAdapter
from .objectives import LinearObjective, ControlTarget, ControlTrackingObjective
from .controllers import AffineControlRule, AffineController
from .frozen_controller import FrozenController
from .controller_synthesis import ControllerSynthesis, synthesize_controller
from .certificate_verification import (
    CertificateVerification,
    VerificationCheck,
    verify_result,
)
from .vector import VectorParameter
from .profiles import PiecewiseLinearProfile
from .geometries import BudgetSet, EllipsoidSet, SimplexSet
from .polytope import LinearConstraint, PolytopeSet
from .composition import Intersection, Product, Union
from .domain_io import domain_from_json, domain_from_manifest
from ._geometry import LinearSupport
from .claim import AdapterCapabilities, Capability, Claim, ModelAdapter
from .changes import (
    ChangeComparison,
    ChangeResult,
    ContractComparison,
    EngineeringChange,
    FieldChange,
    RequirementSummary,
    compare_changes,
)
from .distance import Distance, NormalizedL2, NormalizedLInf
from .domains import (
    Coordinate,
    DomainCapabilities,
    FiniteSet,
    MembershipCheck,
    ParameterSpace,
    Scenario,
    UncertaintySet,
)
from .recourse import DecisionRule, RecoursePolicy
from .contract_types import (
    ArtifactRef,
    Axis,
    Bound,
    Cardinality,
    ConstraintCheck,
    ConstraintSpec,
    Coverage,
    Derivative,
    Diagnostic,
    Evidence,
    EvaluationPayload,
    Measurement,
    Membership,
    Objective,
    ProcessContract,
    QuantitySpec,
    QuantityValue,
    RobustnessPayload,
    Search,
    SensitivityPayload,
    Witness,
)
from .engineering_results import (
    EvaluationResult,
    RobustnessResult,
    SensitivityResult,
    result_from_json,
)
from .primitives import BoxSet, Parameter
from .convex_hull import ConvexHullSet
from .claim import bind_contract, rejected_result

__all__ = [
    "AffineOutput",
    "AffineProcessAdapter",
    "AffineRequirement",
    "AffineTerm",
    "LinearControl",
    "LinearProcessAdapter",
    "LinearObjective",
    "ControlTarget",
    "ControlTrackingObjective",
    "AffineControlRule",
    "AffineController",
    "FrozenController",
    "ControllerSynthesis",
    "synthesize_controller",
    "CertificateVerification",
    "VerificationCheck",
    "verify_result",
    "VectorParameter",
    "PiecewiseLinearProfile",
    "BudgetSet",
    "EllipsoidSet",
    "SimplexSet",
    "LinearConstraint",
    "PolytopeSet",
    "Intersection",
    "Product",
    "Union",
    "domain_from_json",
    "domain_from_manifest",
    "LinearSupport",
    "AdapterCapabilities",
    "Capability",
    "Claim",
    "EngineeringChange",
    "ChangeResult",
    "ChangeComparison",
    "ContractComparison",
    "FieldChange",
    "RequirementSummary",
    "compare_changes",
    "ModelAdapter",
    "Distance",
    "NormalizedLInf",
    "NormalizedL2",
    "Coordinate",
    "DomainCapabilities",
    "FiniteSet",
    "ConvexHullSet",
    "MembershipCheck",
    "ParameterSpace",
    "Scenario",
    "UncertaintySet",
    "DecisionRule",
    "RecoursePolicy",
    "ArtifactRef",
    "Axis",
    "Bound",
    "Cardinality",
    "ConstraintCheck",
    "ConstraintSpec",
    "Coverage",
    "Derivative",
    "Diagnostic",
    "Evidence",
    "EvaluationPayload",
    "EvaluationResult",
    "Measurement",
    "Membership",
    "Objective",
    "ProcessContract",
    "QuantitySpec",
    "QuantityValue",
    "RobustnessPayload",
    "RobustnessResult",
    "Search",
    "SensitivityPayload",
    "SensitivityResult",
    "Witness",
    "result_from_json",
    "BoxSet",
    "Parameter",
    "bind_contract",
    "rejected_result",
]
