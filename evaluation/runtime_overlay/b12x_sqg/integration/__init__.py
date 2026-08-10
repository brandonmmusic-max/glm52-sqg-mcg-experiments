"""vLLM / SGLang integration shims for b12x_sqg kernels.

The maintainer's private fork carries the full FP4 glue under
``b12x_sqg/integration/`` (``tp_moe.py``, ``mla.py``, ...).  FP6 follows the
same pattern: thin framework adapters that call b12x_sqg's public
``plan`` / ``bind`` / ``run`` APIs.
"""
