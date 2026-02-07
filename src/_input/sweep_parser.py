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
- TUPLE MODE: Specify explicit parameter tuples instead of Cartesian product

Tuple Mode Syntax:
    tuple(sfe, mCloud)    [0.01, 1e5] [0.10, 1e7] [0.30, 1e8]
    tuple(mCloud, nCore, sfe)    [1e5, 1e3, 0.01] [1e7, 1e4, 0.10]

This runs only the specified combinations, not the full Cartesian product.
Parameters not in the tuple are fixed across all runs.

Author: Claude Code
Date: 2026-01-14
"""

import itertools
import math
import logging
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union, Optional

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
# Tuple Mode Parsing
# =============================================================================

def parse_tuple_line(line: str) -> Optional[Tuple[List[str], List[List[Any]]]]:
    """
    Parse a tuple definition line.

    Syntax: tuple(param1, param2, ...)    [val1, val2, ...] [val1, val2, ...] ...

    Parameters
    ----------
    line : str
        Line from parameter file

    Returns
    -------
    (param_names, tuple_values) or None if not a tuple line
        - param_names: List of parameter names in the tuple
        - tuple_values: List of value lists, one per combination

    Examples
    --------
    >>> parse_tuple_line("tuple(sfe, mCloud)    [0.01, 1e5] [0.10, 1e7]")
    (['sfe', 'mCloud'], [[0.01, 100000.0], [0.1, 10000000.0]])
    """
    line = line.strip()

    # Check if line starts with 'tuple('
    if not line.lower().startswith('tuple('):
        return None

    # Extract the tuple definition: tuple(param1, param2, ...)
    # Find matching closing parenthesis
    paren_start = line.find('(')
    paren_end = line.find(')')

    if paren_start == -1 or paren_end == -1 or paren_end < paren_start:
        raise ValueError(f"Invalid tuple syntax: {line}")

    # Extract parameter names
    params_str = line[paren_start + 1:paren_end]
    param_names = [p.strip() for p in params_str.split(',')]

    if not param_names or any(not p for p in param_names):
        raise ValueError(f"Invalid tuple parameter names: {line}")

    # Extract the value tuples: [val1, val2] [val3, val4] ...
    rest = line[paren_end + 1:].strip()

    # Find all [...] groups
    tuple_values = []
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, rest)

    for match in matches:
        # Parse the values inside [...]
        values = []
        items = [item.strip() for item in match.split(',')]

        if len(items) != len(param_names):
            raise ValueError(
                f"Tuple value count mismatch: expected {len(param_names)} values "
                f"for params {param_names}, got {len(items)} in [{match}]"
            )

        for item in items:
            # Parse each value
            if item.lower() == 'true':
                values.append(True)
            elif item.lower() == 'false':
                values.append(False)
            else:
                try:
                    values.append(float(item))
                except ValueError:
                    try:
                        values.append(float(Fraction(item)))
                    except (ValueError, ZeroDivisionError):
                        values.append(item)

        tuple_values.append(values)

    if not tuple_values:
        raise ValueError(f"No tuple values found in: {line}")

    return param_names, tuple_values


# =============================================================================
# Sweep Parameter File Reading
# =============================================================================

@dataclass
class SweepConfig:
    """Configuration parsed from a sweep parameter file."""
    base_params: Dict[str, Any]
    sweep_params: Dict[str, List[Any]]
    tuple_params: Optional[List[str]] = None  # Parameter names in tuple
    tuple_values: Optional[List[List[Any]]] = None  # List of value tuples

    @property
    def is_tuple_mode(self) -> bool:
        """Check if this uses tuple syntax (pure tuple or hybrid)."""
        return self.tuple_params is not None and self.tuple_values is not None

    @property
    def is_hybrid_mode(self) -> bool:
        """Check if this is hybrid mode (tuple + sweep params)."""
        return self.is_tuple_mode and bool(self.sweep_params)


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


def read_sweep_config(path2file: str) -> SweepConfig:
    """
    Read a sweep-enabled parameter file and return a SweepConfig.

    Supports both modes:
    - Cartesian mode: Lists of values generate all combinations
    - Tuple mode: Explicit tuples specify exact combinations

    Parameters
    ----------
    path2file : str or Path
        Path to the sweep parameter file

    Returns
    -------
    SweepConfig
        Configuration object with base_params, sweep_params, and optional tuple info

    Example (Cartesian mode)
    ------------------------
    For a file containing:
        mCloud    [1e5, 1e7]
        sfe       [0.01, 0.10]
        dens_profile    densPL

    Returns SweepConfig with:
        base_params = {'dens_profile': 'densPL'}
        sweep_params = {'mCloud': [1e5, 1e7], 'sfe': [0.01, 0.10]}

    Example (Tuple mode)
    --------------------
    For a file containing:
        tuple(mCloud, sfe)    [1e5, 0.01] [1e7, 0.10] [1e8, 0.30]
        nCore    1e4
        dens_profile    densPL

    Returns SweepConfig with:
        base_params = {'nCore': 1e4, 'dens_profile': 'densPL'}
        sweep_params = {}
        tuple_params = ['mCloud', 'sfe']
        tuple_values = [[1e5, 0.01], [1e7, 0.10], [1e8, 0.30]]
    """
    path = Path(path2file)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    base_params = {}
    sweep_params = {}
    tuple_params = None
    tuple_values = None

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            # Remove inline comments
            if '#' in line:
                line = line[:line.find('#')]

            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for tuple syntax
            if line.lower().startswith('tuple('):
                if tuple_params is not None:
                    raise ValueError(
                        f"Line {line_num}: Multiple tuple definitions not allowed. "
                        "Use a single tuple() line with all combinations."
                    )

                try:
                    result = parse_tuple_line(line)
                    if result:
                        tuple_params, tuple_values = result
                        logger.info(f"  Tuple mode: {tuple_params} with {len(tuple_values)} combinations")
                except ValueError as e:
                    raise ValueError(f"Line {line_num}: {e}")
                continue

            # Parse parameter line (format: key value)
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
                    base_params[key] = value[0]
                    logger.debug(f"  {key}: {value[0]} (single-element list -> base)")
                else:
                    sweep_params[key] = value
                    logger.debug(f"  {key}: {value} (sweep, {len(value)} values)")
            else:
                base_params[key] = value
                logger.debug(f"  {key}: {value} (base)")

    # Validate: tuple params and sweep params must not overlap
    if tuple_params is not None and sweep_params:
        overlap = set(tuple_params) & set(sweep_params.keys())
        if overlap:
            raise ValueError(
                f"Parameter(s) {overlap} appear in both tuple() and as sweep parameters. "
                f"Each parameter can only be specified once."
            )

    logger.info(f"Read {len(base_params)} base params")
    if tuple_params:
        logger.info(f"Tuple mode: {tuple_params} with {len(tuple_values)} combinations")
    else:
        logger.info(f"Cartesian mode: {len(sweep_params)} sweep params")

    return SweepConfig(
        base_params=base_params,
        sweep_params=sweep_params,
        tuple_params=tuple_params,
        tuple_values=tuple_values
    )


# =============================================================================
# Combination Generation
# =============================================================================

def generate_combinations_from_config(config: SweepConfig) -> Iterator[Tuple[Dict[str, Any], str]]:
    """
    Generate parameter combinations from a SweepConfig.

    Handles three modes:
    - Cartesian mode: Lists of values generate all combinations
    - Tuple mode: Explicit tuples specify exact combinations
    - Hybrid mode: Tuple combinations × Cartesian product of sweep params

    Parameters
    ----------
    config : SweepConfig
        Configuration from read_sweep_config()

    Yields
    ------
    (params_dict, output_name) : tuple
        - params_dict: Complete parameter dictionary for one simulation
        - output_name: Generated name following convention {mass}_sfe{sfe}_n{nCore}
    """
    if config.is_tuple_mode:
        if config.sweep_params:
            # Hybrid mode: tuple combinations × sweep params Cartesian product
            sweep_keys = list(config.sweep_params.keys())
            sweep_value_lists = [config.sweep_params[k] for k in sweep_keys]

            for tuple_values in config.tuple_values:
                # Build base params with tuple values
                tuple_base = config.base_params.copy()
                for param_name, value in zip(config.tuple_params, tuple_values):
                    tuple_base[param_name] = value

                # Generate Cartesian product of sweep params
                for sweep_combo in itertools.product(*sweep_value_lists):
                    params = tuple_base.copy()
                    for key, value in zip(sweep_keys, sweep_combo):
                        params[key] = value
                    name = generate_run_name(params)
                    yield params, name
        else:
            # Pure tuple mode: use explicit combinations only
            for values in config.tuple_values:
                params = config.base_params.copy()
                for param_name, value in zip(config.tuple_params, values):
                    params[param_name] = value
                name = generate_run_name(params)
                yield params, name
    else:
        # Pure Cartesian mode: use existing logic
        yield from generate_combinations(config.base_params, config.sweep_params)


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

    Format: {mCloud}_sfe{sfe*100:03d}_n{nCore}[_profile_suffix]
    Examples:
        - 1e7_sfe010_n1e4 (default, no density profile suffix)
        - 1e5_sfe001_n1e4_PL0 (powerlaw alpha=0)
        - 1e5_sfe001_n1e4_PL-2 (powerlaw alpha=-2)
        - 1e5_sfe001_n1e4_BE14 (Bonnor-Ebert Omega=14.1)

    Parameters
    ----------
    params : dict
        Parameter dictionary containing mCloud, sfe, nCore, and optionally
        dens_profile, densPL_alpha, densBE_Omega

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

    base_name = f"{mcloud_str}_sfe{sfe_str}_n{ncore_str}"

    # Add density profile suffix if present
    dens_profile = params.get('dens_profile')
    if dens_profile:
        if dens_profile == 'densPL':
            alpha = params.get('densPL_alpha', 0)
            # Format alpha: 0 -> "0", -1 -> "-1", -2 -> "-2"
            alpha_int = int(alpha)
            base_name += f"_PL{alpha_int}"
        elif dens_profile == 'densBE':
            omega = params.get('densBE_Omega', 14.1)
            # Format omega: 14.1 -> "14", 7.0 -> "7"
            omega_int = int(round(omega))
            base_name += f"_BE{omega_int}"

    return base_name


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


def count_combinations_from_config(config: SweepConfig) -> int:
    """
    Count total number of combinations from a SweepConfig.

    Handles three modes:
    - Cartesian mode: Product of all sweep param lengths
    - Tuple mode: Number of explicit tuples
    - Hybrid mode: Number of tuples × product of sweep param lengths

    Parameters
    ----------
    config : SweepConfig
        Configuration from read_sweep_config()

    Returns
    -------
    int
        Total number of combinations
    """
    if config.is_tuple_mode:
        n_tuples = len(config.tuple_values)
        if config.sweep_params:
            # Hybrid mode: tuples × sweep combinations
            return n_tuples * count_combinations(config.sweep_params)
        else:
            # Pure tuple mode
            return n_tuples
    else:
        return count_combinations(config.sweep_params)


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

    # Test parse_tuple_line
    print("\nTesting parse_tuple_line:")
    tuple_cases = [
        ("tuple(sfe, mCloud)    [0.01, 1e5] [0.10, 1e7]",
         (['sfe', 'mCloud'], [[0.01, 1e5], [0.10, 1e7]])),
        ("tuple(mCloud, nCore, sfe)    [1e5, 1e3, 0.01]",
         (['mCloud', 'nCore', 'sfe'], [[1e5, 1e3, 0.01]])),
        ("TUPLE(a, b)    [1, 2] [3, 4] [5, 6]",
         (['a', 'b'], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
    ]

    for input_str, expected in tuple_cases:
        result = parse_tuple_line(input_str)
        # Compare with tolerance for floats
        if result is None:
            status = "FAIL"
        elif result[0] != expected[0]:
            status = "FAIL"
        elif len(result[1]) != len(expected[1]):
            status = "FAIL"
        else:
            status = "PASS"
            for r_vals, e_vals in zip(result[1], expected[1]):
                for r, e in zip(r_vals, e_vals):
                    if abs(r - e) > 1e-9:
                        status = "FAIL"
                        break
        print(f"  {status}: parse_tuple_line('{input_str[:40]}...')")
        if status == "FAIL":
            print(f"    Expected: {expected}")
            print(f"    Got: {result}")

    # Test non-tuple line returns None
    result = parse_tuple_line("mCloud    [1e5, 1e7]")
    status = "PASS" if result is None else "FAIL"
    print(f"  {status}: parse_tuple_line('mCloud [1e5, 1e7]') returns None")

    print("\nAll tests complete!")
