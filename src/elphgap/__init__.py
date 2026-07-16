from .allen_dynes import moments, tc_allen_dynes, tc_mcmillan
from .eliashberg import TcResult, max_eigenvalue, tc_eliashberg
from .eliashberg_aniso import AnisoState, max_eigenvalue_aniso, tc_aniso, tc_aniso_linearized
from .io import (
    A2FColumnError,
    A2FError,
    A2FParseError,
    A2FSpectrum,
    A2FWarning,
    Material,
    lambda_of,
    load_database,
    read_a2f,
)
from .units import K_TO_MEV, MEV_TO_K

__version__ = "0.1.1"


def example_a2f_path(name: str = "pb_like.a2f") -> str:
    """Filesystem path to a packaged example alpha^2F file.

    Works both from a source checkout and after ``pip install`` (the example is
    shipped as package data under ``elphgap/examples/``), so the documented
    ``elphgap inspect``/``tc`` commands run without a clone::

        elphgap inspect "$(python -c 'import elphgap; print(elphgap.example_a2f_path())')"
    """
    from importlib.resources import files

    return str(files("elphgap").joinpath("examples", name))

__all__ = [
    "__version__",
    "moments",
    "tc_allen_dynes",
    "tc_mcmillan",
    "TcResult",
    "max_eigenvalue",
    "tc_eliashberg",
    "tc_aniso",
    "tc_aniso_linearized",
    "max_eigenvalue_aniso",
    "AnisoState",
    "Material",
    "load_database",
    "read_a2f",
    "lambda_of",
    "example_a2f_path",
    "A2FSpectrum",
    "A2FWarning",
    "A2FError",
    "A2FParseError",
    "A2FColumnError",
    "K_TO_MEV",
    "MEV_TO_K",
]
