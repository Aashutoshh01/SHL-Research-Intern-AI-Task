"""Chat endpoint — the main API surface.

Receives full conversation history, runs the LangGraph pipeline,
and returns structured recommendations. Completely stateless.
"""

from fastapi import APIRouter, HTTPException

from app.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Recommendation,
)
from app.core.logging import get_logger
from app.graph.workflow import get_compiled_graph
from app.utils.text import sanitize_input

logger = get_logger(__name__)

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    summary="Conversational assessment recommender",
    description=(
        "Receives the full conversation history and returns assessment "
        "recommendations from the SHL catalog. Stateless — no server-side "
        "session memory."
    ),
    response_model=ChatResponse,
)
def chat(request: ChatRequest) -> ChatResponse:
    """Process a conversation turn and return recommendations.

    The endpoint is completely stateless. All context comes from
    the full conversation history in the request body.

    Args:
        request: ChatRequest containing the full message history.

    Returns:
        ChatResponse with reply, recommendations, and end_of_conversation flag.

    Raises:
        HTTPException: If processing fails.
    """
    try:
        # Convert Pydantic messages to dicts and sanitize input
        messages = [
            {"role": msg.role, "content": sanitize_input(msg.content)}
            for msg in request.messages
        ]

        logger.info(
            "chat_request_received",
            message_count=len(messages),
            last_role=messages[-1]["role"] if messages else "none",
        )

        # Initialize graph state
        initial_state = {
            "messages": messages,
            "conversation_state": None,
            "intent": "",
            "ranked_assessments": [],
            "reply": "",
            "recommendations": [],
            "end_of_conversation": False,
        }

        # Execute the LangGraph pipeline
        graph = get_compiled_graph()
        result = graph.invoke(initial_state)

        # Build response from graph output
        recommendations = [
            Recommendation(**rec) for rec in result.get("recommendations", [])
        ]

        response = ChatResponse(
            reply=result.get("reply", "I'm sorry, I couldn't process your request."),
            recommendations=recommendations,
            end_of_conversation=result.get("end_of_conversation", False),
        )

        logger.info(
            "chat_response_sent",
            recommendations_count=len(recommendations),
            end_of_conversation=response.end_of_conversation,
        )

        return response

    except Exception as e:
        logger.error("chat_processing_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}",
        )

