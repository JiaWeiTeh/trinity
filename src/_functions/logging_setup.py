#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRINITY Logging Setup Module

This module provides centralized logging configuration for TRINITY simulations.
Supports:
- Console output (color-coded by log level)
- File output (.log files in simulation output directory)
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Integration with existing params dictionary
- Per-module logger configuration

Created: 2026-01-08
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


# ANSI color codes for terminal output
class LogColors:
    """ANSI color codes for colored terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Log level colors
    DEBUG = '\033[36m'      # Cyan
    INFO = '\033[32m'       # Green
    WARNING = '\033[33m'    # Yellow
    ERROR = '\033[31m'      # Red
    CRITICAL = '\033[35m'   # Magenta

    # Component colors
    TIME = '\033[90m'       # Gray
    MODULE = '\033[94m'     # Blue


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to terminal output.

    Colors are applied based on log level:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Magenta
    """

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    def format(self, record):
        """Format log record with colors."""
        # Get color for this log level
        level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)

        # Color the level name
        record.levelname = f"{level_color}{record.levelname:8s}{LogColors.RESET}"

        # Color the module name
        record.name = f"{LogColors.MODULE}{record.name}{LogColors.RESET}"

        # Format the message
        formatted = super().format(record)

        return formatted


def setup_logging(
    log_level: Union[str, int] = 'INFO',
    console_output: bool = True,
    file_output: bool = True,
    log_file_path: Optional[Union[str, Path]] = None,
    log_file_name: Optional[str] = None,
    use_colors: bool = True,
    format_string: Optional[str] = None,
    suppress_library_debug: bool = True,
) -> logging.Logger:
    """
    Set up TRINITY logging system.

    This function configures the root logger for TRINITY simulations with
    flexible output options (console, file, or both) and customizable formatting.

    Parameters
    ----------
    log_level : str or int, optional
        Logging level. Can be:
        - String: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        - Integer: logging.DEBUG (10), logging.INFO (20), etc.
        Default: 'INFO'

    console_output : bool, optional
        If True, log messages are printed to terminal (stdout).
        Default: True

    file_output : bool, optional
        If True, log messages are written to a .log file.
        Default: True

    log_file_path : str or Path, optional
        Directory where log file will be created.
        If None, uses current directory.
        Typically set to params['path2output'].value
        Default: None (current directory)

    log_file_name : str, optional
        Name of log file. If None, generates name from timestamp:
        'trinity_YYYYMMDD_HHMMSS.log'
        Default: None (auto-generate)

    use_colors : bool, optional
        If True, use colored output in terminal (ANSI colors).
        Set to False if running in environment without color support.
        Default: True

    format_string : str, optional
        Custom format string for log messages.
        If None, uses default format:
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        See Python logging documentation for format codes.
        Default: None

    Returns
    -------
    logging.Logger
        Configured root logger instance

    Examples
    --------
    Basic usage (console only, INFO level):
    >>> from src._functions.logging_setup import setup_logging
    >>> logger = setup_logging()
    >>> logger.info("Simulation started")
    2026-01-08 15:30:45 | INFO     | __main__ | Simulation started

    Console + file output with custom directory:
    >>> logger = setup_logging(
    ...     log_level='DEBUG',
    ...     console_output=True,
    ...     file_output=True,
    ...     log_file_path='outputs/my_simulation'
    ... )
    >>> logger.debug("Detailed debug information")
    >>> # Also written to: outputs/my_simulation/trinity_20260108_153045.log

    File output only (no console spam):
    >>> logger = setup_logging(
    ...     log_level='WARNING',
    ...     console_output=False,
    ...     file_output=True,
    ...     log_file_path=params['path2output'].value
    ... )

    Integration with TRINITY params:
    >>> def start_expansion(params):
    ...     # Set up logging using params
    ...     logger = setup_logging(
    ...         log_level=params.get('log_level', 'INFO'),
    ...         console_output=params.get('log_console', True),
    ...         file_output=params.get('log_file', True),
    ...         log_file_path=params['path2output'].value,
    ...     )
    ...     logger.info("Starting expansion phase")

    Notes
    -----
    - Call this function once at the start of your simulation (in main.py)
    - After setup, use module-level loggers in individual files:
        logger = logging.getLogger(__name__)
    - Log levels (from most to least verbose):
        DEBUG (10): Detailed debugging info (variables, loop iterations)
        INFO (20): General information (phase transitions, major events)
        WARNING (30): Warnings that don't stop simulation (clamped values, etc.)
        ERROR (40): Errors that may affect results but don't crash
        CRITICAL (50): Critical errors that stop simulation

    - ANSI colors work in most terminals but may not work in:
        - Windows Command Prompt (use Windows Terminal instead)
        - Some IDEs (check IDE settings)
        - Redirected output (e.g., python script.py > output.txt)

    - Log files use plain text (no colors) for better readability

    See Also
    --------
    get_module_logger : Get a logger for a specific module
    set_log_level : Change log level after initialization
    """

    # Convert string log level to integer
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers (avoid duplicates if called multiple times)
    root_logger.handlers.clear()

    # Default format string
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'

    # Date format
    date_format = '%Y-%m-%d %H:%M:%S'

    # =============================================================================
    # Console Handler (stdout)
    # =============================================================================
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Use colored formatter for console if requested
        if use_colors and sys.stdout.isatty():  # Only use colors if output is terminal
            console_formatter = ColoredFormatter(format_string, datefmt=date_format)
        else:
            console_formatter = logging.Formatter(format_string, datefmt=date_format)

        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # =============================================================================
    # File Handler (.log file)
    # =============================================================================
    if file_output:
        # Determine log file path
        if log_file_path is None:
            log_file_path = Path.cwd()
        else:
            log_file_path = Path(log_file_path)

        # Create directory if it doesn't exist
        log_file_path.mkdir(parents=True, exist_ok=True)

        # Generate log file name if not provided
        if log_file_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file_name = f'trinity_{timestamp}.log'

        # Full path to log file
        log_file_full_path = log_file_path / log_file_name

        # Create file handler
        file_handler = logging.FileHandler(log_file_full_path, mode='w', encoding='utf-8')
        file_handler.setLevel(log_level)

        # Use plain formatter for file (no colors)
        file_formatter = logging.Formatter(format_string, datefmt=date_format)
        file_handler.setFormatter(file_formatter)

        root_logger.addHandler(file_handler)

        # Log where the log file is being written
        root_logger.info(f"Log file: {log_file_full_path}")

    # =============================================================================
    # Suppress third-party library debug messages
    # =============================================================================
    # When using DEBUG level, third-party libraries can be very noisy.
    # This sets common noisy libraries to INFO level to keep logs focused on
    # TRINITY's science-related debug output.
    if suppress_library_debug and log_level <= logging.DEBUG:
        noisy_libraries = [
            'matplotlib',
            'PIL',
            'urllib3',
            'asyncio',
            'parso',
            'fontTools',
            'numba',
            'h5py',
        ]
        for lib in noisy_libraries:
            logging.getLogger(lib).setLevel(logging.INFO)

    return root_logger


def get_module_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    This is the recommended way to get a logger in individual Python files.
    The logger will inherit settings from the root logger configured by
    setup_logging().

    Parameters
    ----------
    name : str
        Module name. Typically use __name__ to get current module.

    Returns
    -------
    logging.Logger
        Logger instance for this module

    Examples
    --------
    At the top of your module file:
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>>
    >>> def my_function():
    ...     logger.debug("Starting function")
    ...     logger.info("Important event")
    ...     logger.warning("Something unexpected")
    ...     logger.error("Error occurred")

    Or use the helper function:
    >>> from src._functions.logging_setup import get_module_logger
    >>> logger = get_module_logger(__name__)
    """
    return logging.getLogger(name)


def set_log_level(level: Union[str, int], logger_name: Optional[str] = None):
    """
    Change log level after initialization.

    Parameters
    ----------
    level : str or int
        New log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    logger_name : str, optional
        Name of specific logger to change. If None, changes root logger.
        Default: None (root logger)

    Examples
    --------
    >>> # Change global log level to DEBUG
    >>> set_log_level('DEBUG')
    >>>
    >>> # Change specific module log level
    >>> set_log_level('WARNING', 'src.phase1_energy.run_energy_phase')
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    if logger_name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(logger_name)

    logger.setLevel(level)

    # Also update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)


def setup_logging_from_params(params):
    """
    Convenience function to set up logging from TRINITY params dictionary.

    This function reads logging configuration from params and calls setup_logging()
    with appropriate values.

    Parameters
    ----------
    params : DescribedDict
        TRINITY parameters dictionary

    Expected params keys (all optional with defaults):
    -------------------------------------------------
    - params['log_level'].value : str
        Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        Default: 'INFO'

    - params['log_console'].value : bool
        Enable console output
        Default: True

    - params['log_file'].value : bool
        Enable file output
        Default: True

    - params['log_colors'].value : bool
        Use colored console output
        Default: True

    - params['path2output'].value : str
        Output directory for log file
        Default: current directory

    Returns
    -------
    logging.Logger
        Configured root logger

    Examples
    --------
    In main.py:
    >>> from src._functions.logging_setup import setup_logging_from_params
    >>>
    >>> def start_expansion(params):
    ...     # Set up logging (reads from params)
    ...     logger = setup_logging_from_params(params)
    ...
    ...     logger.info("TRINITY simulation starting")
    ...     logger.info(f"Output directory: {params['path2output'].value}")
    ...
    ...     # ... rest of simulation ...

    In parameter file (.param):
    ```
    # Logging configuration
    log_level = INFO          # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_console = True        # Print to terminal
    log_file = True           # Write to .log file
    log_colors = True         # Use colored output (set False for plain text)
    ```
    """

    # Extract logging parameters with defaults
    log_level = params.get('log_level', 'INFO')
    if hasattr(log_level, 'value'):
        log_level = log_level.value

    log_console = params.get('log_console', True)
    if hasattr(log_console, 'value'):
        log_console = log_console.value

    log_file = params.get('log_file', True)
    if hasattr(log_file, 'value'):
        log_file = log_file.value

    log_colors = params.get('log_colors', True)
    if hasattr(log_colors, 'value'):
        log_colors = log_colors.value

    # Get output directory
    log_file_path = None
    if 'path2output' in params:
        log_file_path = params['path2output'].value if hasattr(params['path2output'], 'value') else params['path2output']

    # Set up logging
    logger = setup_logging(
        log_level=log_level,
        console_output=log_console,
        file_output=log_file,
        log_file_path=log_file_path,
        use_colors=log_colors,
    )

    return logger


# =============================================================================
# Usage Examples and Testing
# =============================================================================

if __name__ == "__main__":
    """
    Test and demonstrate logging functionality.
    Run this file directly to see logging examples:
        python src/_functions/logging_setup.py
    """

    print("=" * 80)
    print("TRINITY Logging System - Examples")
    print("=" * 80)

    # Example 1: Basic setup (console only)
    print("\n1. Basic Setup (Console Only, INFO level)")
    print("-" * 80)
    logger = setup_logging(
        log_level='INFO',
        console_output=True,
        file_output=False,
    )

    logger.debug("This won't show (below INFO level)")
    logger.info("This is an informational message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Example 2: DEBUG level (console + file)
    print("\n\n2. DEBUG Level (Console + File)")
    print("-" * 80)
    logger = setup_logging(
        log_level='DEBUG',
        console_output=True,
        file_output=True,
        log_file_path='.',
        log_file_name='test_trinity.log',
    )

    logger.debug("Detailed debug information")
    logger.info("R2 = 10.5 pc, v2 = 150 pc/Myr")
    logger.warning("Temperature below minimum, clamping to 1e4 K")
    logger.error("Shell structure calculation failed, retrying...")

    # Example 3: Module-specific loggers
    print("\n\n3. Module-Specific Loggers")
    print("-" * 80)

    # Simulate different modules
    logger_main = logging.getLogger('src.main')
    logger_phase1 = logging.getLogger('src.phase1_energy.run_energy_phase')
    logger_cooling = logging.getLogger('src.cooling.net_coolingcurve')
    logger_sb99 = logging.getLogger('src.sb99.update_feedback')

    logger_main.info("Starting TRINITY simulation")
    logger_phase1.info("Entering energy-driven phase")
    logger_cooling.debug("Interpolating cooling curve at T=1.5e6 K")
    logger_sb99.warning("Wind momentum rate near zero, setting v_mech_total=0")
    logger_phase1.info("Energy-driven phase complete")
    logger_main.info("Simulation finished")

    # Example 4: Changing log level dynamically
    print("\n\n4. Dynamic Log Level Changes")
    print("-" * 80)

    logger.info("Starting with DEBUG level")
    logger.debug("You can see this debug message")

    set_log_level('WARNING')
    logger.info("Changed to WARNING level - this INFO message won't show")
    logger.debug("Neither will this DEBUG message")
    logger.warning("But warnings still show")

    # Example 5: No colors (plain text)
    print("\n\n5. Plain Text Output (No Colors)")
    print("-" * 80)
    logger = setup_logging(
        log_level='INFO',
        console_output=True,
        file_output=False,
        use_colors=False,
    )

    logger.info("Plain text message")
    logger.warning("Plain text warning")
    logger.error("Plain text error")

    print("\n" + "=" * 80)
    print("Check 'test_trinity.log' to see file output")
    print("=" * 80)
