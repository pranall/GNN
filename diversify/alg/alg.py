from .algs.diversify import Diversify

__all__ = ['Diversify', 'get_algorithm_class']  # Add this line

ALGORITHMS = [
    'diversify'
]

def get_algorithm_class(algorithm_name):
    """Factory function for algorithm selection"""
    if algorithm_name not in ALGORITHMS:
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return Diversify
