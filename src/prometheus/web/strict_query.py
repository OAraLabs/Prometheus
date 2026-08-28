"""Reject unknown query parameters instead of silently ignoring them.

Starlette drops an unrecognised query parameter, so a MISTYPED one returns a plausible answer
rather than an error — and a plausible answer to a question nobody asked is worse than an error,
because nothing about it looks wrong.

The concrete case, 2026-08-28: ``GET /api/usage?window=7d`` returned the ALL-TIME rollup, byte
identical to ``?window=30d`` and ``?window=all``. Three identical responses read as a broken
window feature; the real parameter is ``days``. The response even carried ``window_days: null``
saying so, and that was not enough — the wrong request had already produced a right-looking
answer, and the answer is what gets believed.

Fail closed, the same principle as the sandbox's path guard and install-local's refusal to install
a build older than HEAD: when the input cannot be honoured as written, say so immediately rather
than proceeding with a defensible interpretation of it.

The error names what was rejected AND what the route accepts, because "unknown parameter: window"
alone leaves the caller to guess a second time.

NOT applied to:
  * the StaticFiles mount — it is not an APIRoute, so cache-busting query strings on static
    assets keep working.
  * ``/docs``, ``/redoc``, ``/openapi.json`` — FastAPI registers those during construction,
    before the route class is swapped in, so Swagger UI is untouched.
"""

from __future__ import annotations

from fastapi.dependencies.models import Dependant
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.requests import Request
from starlette.responses import Response


def known_query_names(dependant: Dependant) -> set[str]:
    """Every query name a route accepts, INCLUDING those declared by sub-dependencies.

    Walking ``dependencies`` matters: a name reachable only through ``Depends(...)`` is still a
    name the route accepts, and rejecting it would break a working caller. The recursion is over
    FastAPI's own resolved tree, so it stays correct as dependencies are added.
    """
    names = {p.alias for p in dependant.query_params}
    for sub in dependant.dependencies:
        names |= known_query_names(sub)
    return names


class StrictQueryRoute(APIRoute):
    """An APIRoute that 400s a request carrying a query parameter it does not declare.

    Install BEFORE registering routes::

        app = FastAPI(...)
        app.router.route_class = StrictQueryRoute
    """

    def get_route_handler(self):
        original = super().get_route_handler()
        # Resolved once at startup, not per request: the dependant tree is fixed after wiring.
        accepted = known_query_names(self.dependant)

        async def strict_query_handler(request: Request) -> Response:
            # set() because .keys() returns a LIST, which has no set-difference. Starlette's
            # multidict already collapses "?a=1&a=2" to one key, so this is not doing the
            # de-duplicating — an earlier comment here claimed it was, and was simply wrong.
            unknown = sorted(set(request.query_params.keys()) - accepted)
            if unknown:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"unknown query parameter{'s' if len(unknown) > 1 else ''}: "
                        + ", ".join(unknown),
                        "unknown": unknown,
                        "accepted": sorted(accepted),
                    },
                )
            return await original(request)

        return strict_query_handler
