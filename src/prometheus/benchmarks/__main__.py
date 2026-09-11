"""Entry point for ``python -m prometheus.benchmarks``.

The guard is not boilerplate. ``main()`` used to be called at module scope, so
IMPORTING this module ran the entire benchmark suite — and because ``main()``
parses ``sys.argv`` with argparse, an import under any other program read that
program's arguments, rejected them, and called ``sys.exit(2)``. Anything that
walks the package tree (a coverage sweep, a docs generator, an import audit,
``pkgutil.walk_packages``) therefore either ran benchmarks or died.

``python -m`` still works because ``-m`` sets ``__name__`` to ``"__main__"``,
which is the whole difference between "this module was invoked" and "this
module was looked at".
"""

from prometheus.benchmarks.runner import main

if __name__ == "__main__":
    main()
