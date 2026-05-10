"""Health check endpoint.

Simple liveness probe for monitoring and load balancers.
"""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Health check",
    description="Returns service health status. Used by load balancers and monitoring.",
    response_model=dict,
)
def health_check() -> dict[str, str]:
    """Return service health status.

    Returns:
        Dict with status field set to 'ok'.
    """
    return {"status": "ok"}
