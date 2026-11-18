# agent.py
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import (
    langchain,
    cartesia,
    deepgram,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from graph import create_workflow

load_dotenv(".env.local")


class InterviewAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a professional interviewer conducting a job interview. "
                "The LangGraph workflow will drive the conversation flow. "
                "Simply speak the questions and responses as they come from the graph. "
                "Be conversational, professional, and helpful throughout the interview process."
            )
        )


async def entrypoint(ctx: agents.JobContext):
    # 1) Build/compile the LangGraph app (Runnable)
    interview_workflow = create_workflow()

    # 2) Wrap it as an LLM for LiveKit via the LangChain plugin
    lg_llm = langchain.LLMAdapter(graph=interview_workflow)

    # 3) Configure the rest of the realtime pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=lg_llm,
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=InterviewAgent(),
    )

    print("Starting interview workflow...")
    await session.generate_reply(
        instructions="Start the interview with the first question."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
