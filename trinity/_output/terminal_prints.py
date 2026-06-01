#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:39:35 2023

@author: Jia Wei Teh

This script contains functions that handle printing information in the terminal.
Uses logging for consistent output formatting across the TRINITY simulation.
"""

import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)


def _format_banner(message: str, width: int = 50) -> str:
    """Format a message as a banner with dashes."""
    separator = "-" * width
    return f"\n{separator}\n{message}\n{separator}"


def bubble():
    """Log message for bubble structure calculation."""
    logger.info(_format_banner("Calculating bubble structure"))


def phase0(time):
    """Log initialization message with timestamp."""
    logger.info(_format_banner(f"{time}: Initialising bubble"))


def phase(string: str):
    """Log phase transition message."""
    logger.info(_format_banner(string))


def shell():
    """Log message for shell structure calculation."""
    logger.info(_format_banner("Calculating shell structure"))


class cprint:
    """
    A class that deals with printing with colours in terminal.

    Example usage:
        print(f'{cprint.BOLD}This text is bolded{cprint.END}  but this isnt.')

    Attributes
    ----------
    BOLD : str
        ANSI escape code for bold cyan text (used for file saves)
    SAVE : str
        Alias for BOLD
    FILE : str
        Alias for BOLD
    LINK : str
        ANSI escape code for green text (used for links)
    WARN : str
        ANSI escape code for bold blue text (used for warnings)
    BLINK : str
        ANSI escape code for blinking text
    FAIL : str
        ANSI escape code for bold red text (used for errors)
    END : str
        ANSI escape code to reset all formatting
    """

    # Symbol prefix for file operations
    symbol = '\u27B3 '

    # Bolded text to signal that a file is being saved
    BOLD = symbol + '\033[1m\033[96m'
    # aliases
    SAVE = BOLD
    FILE = BOLD

    # Link (green)
    LINK = '\033[32m'

    # Warning message, but code runs still (bold blue)
    WARN = '\033[1m\033[94m'

    # Blink
    BLINK = '\033[5m'

    # FAIL (bold red)
    FAIL = '\033[1m\033[91m'

    # END and clear all colours. This should be included in the end of every operation.
    END = '\033[0m'


# =============================================================================
# Logging helper functions
# =============================================================================

def log_file_saved(filepath: str, description: str = "File saved"):
    """Log a file save operation with distinctive formatting."""
    logger.info(f"{cprint.SAVE}{description}: {filepath}{cprint.END}")


def log_warning(message: str):
    """Log a warning message with distinctive formatting."""
    logger.warning(f"{cprint.WARN}{message}{cprint.END}")


def log_error(message: str):
    """Log an error message with distinctive formatting."""
    logger.error(f"{cprint.FAIL}{message}{cprint.END}")


