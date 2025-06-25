# diversify/alg/alg.py
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .algs.diversify import Diversify

ALGORITHMS = [
    'diversify'
]

def get_algorithm_class(algorithm_name):
    name = algorithm_name.lower()
    if name not in ALGORITHMS:
        raise NotImplementedError(
            f"Algorithm not found: {algorithm_name}"
        )
    return Diversify
