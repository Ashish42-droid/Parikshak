"""PARIKSHAK — on-board procedure witness for crewed missions.

Layering rule, enforced by tests/test_layering.py:

    perception/  may not import  engine/
    engine/      may not import  perception/
    both import  belief/

Breaking it collapses the parallelism the whole schedule depends on.
"""

__version__ = "0.1.0"
