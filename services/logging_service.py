from __future__ import annotations

import logging
import warnings


_CONFIGURED = False


def configure_runtime_logging() -> None:
    """Reduce noisy third-party PDF parser warnings while keeping real errors visible."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    noisy_loggers = [
        "pdfminer",
        "pdfminer.pdfinterp",
        "pdfminer.converter",
        "pdfminer.layout",
        "pypdf",
        "pypdf._reader",
        "pypdf.generic",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore",
        message=r".*Multiple definitions in dictionary.*",
        module=r"pypdf.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Cannot set gray non-stroke color.*",
        module=r"pdfminer.*",
    )

    _CONFIGURED = True
