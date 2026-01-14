#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep Parameter Parser for TRINITY
==================================

Parse parameter files with list syntax for parameter sweeps.

Features:
- Detect and parse list syntax: mCloud [1e5, 1e7, 1e8]
- Separate base (constant) parameters from sweep (varying) parameters
- Generate all combinations via Cartesian product
- Generate run names following TRINITY convention: {mass}_sfe{sfe}_n{nCore}

Author: Claude Code
Date: 2026-01-14
"""

import itertools
import math
import logging
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Value Parsing Functions
# =============================================================================

def parse_value(val_str: str) -> Union[bool, float, str, List[Any]]:
    """
    Parse a string value into appropriate Python type.

    Extended to support list syntax: [val1, val2, val3]

    Precedence: list → boolean → number → fraction → string

    Parameters
    ----------
    val_str : str
        String value from parameter file

    Returns
    -------
    Parsed value (bool, float, str, or list)

    Examples
    --------
    >>> parse_value("1e5")
    100000.0
    >>> parse_value("[1e5, 1e7]")
    [100000.0, 10000000.0]
    >>> parse_value("True")
    True
    >>> parse_value("densPL")
    'densPL'
    """
    val_str = val_str.strip()

    # Check for list syntax: [val1, val2, ...]
    if val_str.startswith('[') and val_str.endswith(']'):
        return parse_list(val_str)

    # Boolean
    if val_str.lower() == 'true':
        return True
    elif val_str.lower() == 'false':
        return False

    # Number (float or int)
    try:
        return float(val_str)
    except ValueError:
        pass

    # Fraction (e.g., 5/3)
    try:
        return float(Fraction(val_str))
    except (ValueError, ZeroDivisionError):
        pass

    # String (fallback)
    return val_str


def parse_list(list_str: str) -> List[Any]:
    """
    Parse list syntax: [val1, val2, val3] -> [parsed_val1, parsed_val2, ...]

    Handles:
    - Scientific notation: [1e5, 1e7, 1e8]
    - Decimals: [0.01, 0.10, 0.30]
    - Mixed: [1e5, 100000, 1e8]
    - Strings: [densPL, densBE]
    - Booleans: [True, False]

    Parameters
    ----------
    list_str : str
        String in format "[val1, val2, ...]"

    Returns
    -------
    List of parsed values
    """
    # Remove brackets and split by comma
    inner = list_str[1:-1].strip()

    if not inner:
        return []

    # Split by comma, being careful with whitespace
    items = [item.strip() for item in inner.split(',')]

    # Parse each item individually (without list detection to avoid recursion)
    parsed = []
    for item in items:
        item = item.strip()

        # Boolean
        if item.lower() == 'true':
            parsed.append(True)
        elif item.lower() == 'false':
            parsed.append(False)
        else:
            # Try number
            try:
                parsed.append(float(item))
            except ValueError:
                # Try fraction
                try:
                    parsed.append(float(Fraction(item)))
                except (ValueError, ZeroDivisionError):
                    # String fallback
                    parsed.append(item)

    return parsed


# =============================================================================
# Sweep Parameter File Reading
# =============================================================================

def read_sweep_param(path2file: str) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """
    Read a sweep-enabled parameter file.

    Separates parameters into:
    - base_params: Single values (constant across all runs)
    - sweep_params: List values (will generate combinations)

    Parameters
    ----------
    path2file : str or Path
        Path to the sweep parameter file

    Returns
    -------
    base_params : dict
        Parameters with single values
    sweep_params : dict
        Parameters with list values

    Raises
    ------
    ValueError
        If parameter file has formatting errors
    FileNotFoundError
        If file does not exist

    Example
    -------
    For a file containing:
        mCloud    [1e5, 1e7]
        sfe       [0.01, 0.10]
        dens_profile    densPL

    Returns:
        base_params = {'dens_profile': 'densPL'}
        sweep_params = {'mCloud': [1e5, 1e7], 'sfe': [0.01, 0.10]}
    """
    path = Path(path2file)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    base_params = {}
    sweep_params = {}

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            # Remove inline comments
            if '#' in line:
                line = line[:line.find('#')]

            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Parse parameter line (format: key value)
            # Handle list values which may contain spaces after commas
            # Find the first whitespace that separates key from value
            parts = line.split(None, 1)

            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_num}: Expected format 'key value', got: '{line}'"
                )

            key, val_str = parts
            value = parse_value(val_str)

            # Categorize based on whether it's a list
            if isinstance(value, list):
                if len(value) == 0:
                    raise ValueError(
                        f"Line {line_num}: Empty list for parameter '{key}'"
                    )
                elif len(value) == 1:
                    # Single-element list is treated as base param
                    base_params[key] = value[0]
                    logger.debug(f"  {key}: {value[0]} (single-element list -> base)")
                else:
                    sweep_params[key] = value
                    logger.debug(f"  {key}: {value} (sweep, {len(value)} values)")
            else:
                base_params[key] = value
                logger.debug(f"  {key}: {value} (base)")

    logger.info(f"Read {len(base_params)} base params, {len(sweep_params)} sweep params")

    return base_params, sweep_params


# =============================================================================
# Combination Generation
# =============================================================================

def generate_combinations(
    base_params: Dict[str, Any],
    sweep_params: Dict[str, List[Any]]
) -> Iterator[Tuple[Dict[str, Any], str]]:
    """
    Generate all parameter combinations (Cartesian product).

    Parameters
    ----------
    base_params : dict
        Parameters with single values (constant across all runs)
    sweep_params : dict
        Parameters with list values (will be swept)

    Yields
    ------
    (params_dict, output_name) : tuple
        - params_dict: Complete parameter dictionary for one simulation
        - output_name: Generated name following convention {mass}_sfe{sfe}_n{nCore}
    """
    if not sweep_params:
        # No sweep parameters - just return base params with generated name
        name = generate_run_name(base_params)
        yield base_params.copy(), name
        return

    # Get keys and values in consistent order
    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]

    # Generate Cartesian product
    for combination in itertools.product(*value_lists):
        # Build complete parameter dictionary
        params = base_params.copy()
        for key, value in zip(keys, combination):
            params[key] = value

        # Generate output name
        name = generate_run_name(params)

        yield params, name


def generate_run_name(params: Dict[str, Any]) -> str:
    """
    Generate output folder name following existing TRINITY convention.

    Format: {mCloud}_sfe{sfe*100:03d}_n{nCore}
    Example: 1e7_sfe010_n1e4

    Parameters
    ----------
    params : dict
        Parameter dictionary containing mCloud, sfe, nCore

    Returns
    -------
    str
        Generated run name
    """
    # Extract key parameters (with defaults if missing)
    mcloud = params.get('mCloud', 1e6)
    sfe = params.get('sfe', 0.01)
    ncore = params.get('nCore', 1e5)

    # Format mCloud (scientific notation)
    mcloud_str = format_scientific(mcloud)

    # Format SFE (multiply by 100, zero-padded to 3 digits)
    sfe_int = int(round(sfe * 100))
    sfe_str = f"{sfe_int:03d}"

    # Format nCore (scientific notation)
    ncore_str = format_scientific(ncore)

    return f"{mcloud_str}_sfe{sfe_str}_n{ncore_str}"


def format_scientific(value: float) -> str:
    """
    Format a number in compact scientific notation.

    Always uses scientific notation for values >= 100 to match TRINITY naming.

    Examples:
    - 1000000.0 -> '1e6'
    - 100000.0 -> '1e5'
    - 5000000.0 -> '5e6'
    - 100.0 -> '1e2'
    - 0.01 -> '0.01'

    Parameters
    ----------
    value : float
        Number to format

    Returns
    -------
    str
        Formatted string
    """
    if value == 0:
        return '0'

    abs_value = abs(value)

    # For values >= 100 or very small numbers, use scientific notation
    # This matches TRINITY convention: n1e2, n1e3, n1e4, etc.
    if abs_value >= 100 or (abs_value < 0.01 and abs_value != 0):
        exp = int(math.floor(math.log10(abs_value)))
        mantissa = value / (10 ** exp)

        # Check if mantissa is close to an integer
        if abs(mantissa - round(mantissa)) < 1e-9:
            mantissa_int = int(round(mantissa))
            if mantissa_int == 1:
                return f"1e{exp}"
            else:
                return f"{mantissa_int}e{exp}"
        else:
            # Non-integer mantissa - use compact form
            return f"{mantissa:.2g}e{exp}"
    else:
        # For "normal" numbers (1-99), use regular formatting
        if value == int(value):
            return str(int(value))
        else:
            return str(value)


def count_combinations(sweep_params: Dict[str, List[Any]]) -> int:
    """
    Count total number of combinations without generating them.

    Parameters
    ----------
    sweep_params : dict
        Parameters with list values

    Returns
    -------
    int
        Total number of combinations
    """
    if not sweep_params:
        return 1

    count = 1
    for values in sweep_params.values():
        count *= len(values)
    return count


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Quick test of parsing functions
    logging.basicConfig(level=logging.DEBUG)

    # Test parse_value
    print("Testing parse_value:")
    test_cases = [
        ("1e5", 100000.0),
        ("[1e5, 1e7, 1e8]", [100000.0, 10000000.0, 100000000.0]),
        ("[0.01, 0.10, 0.30]", [0.01, 0.10, 0.30]),
        ("True", True),
        ("densPL", "densPL"),
        ("5/3", 5/3),
    ]

    for input_str, expected in test_cases:
        result = parse_value(input_str)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: parse_value('{input_str}') = {result}")

    # Test format_scientific
    print("\nTesting format_scientific:")
    sci_cases = [
        (100000.0, "1e5"),
        (1000000.0, "1e6"),
        (10000000.0, "1e7"),
        (5000000.0, "5e6"),
        (0.01, "0.01"),
    ]

    for value, expected in sci_cases:
        result = format_scientific(value)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: format_scientific({value}) = '{result}' (expected '{expected}')")

    # Test generate_run_name
    print("\nTesting generate_run_name:")
    params = {'mCloud': 1e7, 'sfe': 0.10, 'nCore': 1e4}
    name = generate_run_name(params)
    expected = "1e7_sfe010_n1e4"
    status = "PASS" if name == expected else "FAIL"
    print(f"  {status}: generate_run_name({params}) = '{name}' (expected '{expected}')")

    print("\nAll tests complete!")
