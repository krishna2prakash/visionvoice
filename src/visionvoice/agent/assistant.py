"""The provider-agnostic agent loop.

Given a user question and the current perception context, the assistant runs a bounded
observe → decide → act cycle: it asks the model what to do, executes any tool calls,
feeds the observations back, and repeats until the model produces a spoken answer.
The exact same loop runs on the cloud, local, or mock backend.
"""

from __future__ import annotations

from visionvoice.agent.tools import TOOL_SPECS, PerceptionContext, execute_tool
from visionvoice.providers.base import ModelProvider
from visionvoice.types import Message

SYSTEM_PROMPT = (
    "You are VisionVoice, a real-time assistant that helps a blind or low-vision user "
    "understand their surroundings through a camera. You have tools to look at the scene, "
    "list objects, read text, and assess safety.\n"
    "Guidelines:\n"
    "- Call the tool(s) needed to answer, then reply.\n"
    "- Answer in ONE or TWO short spoken sentences — this is read aloud.\n"
    "- Be concrete about direction (left / ahead / right) and distance.\n"
    "- Lead with hazards or the most important thing.\n"
    "- Never invent objects you were not told about."
)


class Assistant:
    """Runs the agent loop for a single question."""

    def __init__(self, provider: ModelProvider, max_iterations: int = 4) -> None:
        self.provider = provider
        self.max_iterations = max_iterations

    def ask(self, query: str, ctx: PerceptionContext) -> str:
        """Answer a user query about the current scene, using tools as needed."""
        messages: list[Message] = [Message(role="user", content=query)]

        last_text = ""
        for _ in range(self.max_iterations):
            response = self.provider.chat(
                messages, system=SYSTEM_PROMPT, tools=TOOL_SPECS
            )
            last_text = response.text or last_text

            if not response.wants_tools:
                return response.text or last_text

            # Record the assistant's tool request, then execute each tool.
            messages.append(
                Message(role="assistant", content=response.text, tool_calls=response.tool_calls)
            )
            for call in response.tool_calls:
                observation = execute_tool(call.name, call.arguments, ctx)
                messages.append(
                    Message(role="tool", content=observation, tool_call_id=call.id)
                )

        # Exhausted the loop — return whatever we last have, or a safe fallback.
        return last_text or "I'm not sure how to describe that right now."
