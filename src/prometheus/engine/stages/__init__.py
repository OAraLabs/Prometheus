"""Turn stages: the loop's built-in behavior, one stage per module.

Each module holds a default hook (hook contract, ``docs/contracts/hooks.md``,
Appendix A) moved out of ``engine/agent_loop.py`` and called from exactly the
place its code sat. A default hook always runs, in its fixed order, whether or
not any pillar is installed.
"""
