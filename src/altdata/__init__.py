"""Cross-sectional equity research with alternative (SEC EDGAR) data."""


def main() -> None:
    from .pipeline import main as run

    run()
