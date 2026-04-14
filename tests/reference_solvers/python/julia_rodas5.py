"""Julia Rodas5 reference solver via DiffEqGPU."""

from tests.reference_solvers.python.julia_common import make_solver as _make_solver


def make_solver(
    system_name,
    *,
    system_config=None,
    ensemble_backend="EnsembleGPUArray",
):
    """Create a reusable Julia Rodas5 solver."""
    return _make_solver(
        "rodas5",
        system_name,
        system_config=system_config,
        ensemble_backend=ensemble_backend,
    )
