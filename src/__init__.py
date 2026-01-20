"""
TRINITY: A stellar feedback and bubble evolution simulation code.

This package provides tools for simulating the evolution of stellar feedback-driven
bubbles in molecular clouds, including energy-driven and momentum-driven phases.

Basic Usage:
    from src._input import read_param
    from src import main

    params = read_param.read_param('my_simulation.param')
    main.start_expansion(params)

For reading simulation outputs:
    from src._output.trinity_reader import TrinityOutput

    output = TrinityOutput.open('simulation.jsonl')
    times = output.get('t_now')
    radii = output.get('R2')
"""

__version__ = "1.0.0"
__author__ = "Jia Wei Teh"
