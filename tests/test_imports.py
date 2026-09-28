import importlib


MODULES = [
    "src.config",
    "src.data",
    "src.splits",
    "src.preprocessing",
    "src.features",
    "src.models",
    "src.training",
    "src.evaluation",
    "src.visualization",
    "src.tracking",
    "src.utils",
]


def test_project_modules_import() -> None:
    for module in MODULES:
        importlib.import_module(module)
