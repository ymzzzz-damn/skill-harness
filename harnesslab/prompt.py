"""Own task policy instructions; SDK API/knowledge are explicit dependencies."""
from __future__ import annotations
from typing import Any

def build_prompt(adapter: Any, card: Any, *, max_turns: int,
                 images: str, feedback: str) -> str:
    from pyrualean.prompt import api_reference, knowledge_text
    image_text = ("Current camera images accompany code cells that move the robot."
                  if images == "on-motion" else
                  "Request camera images with robo.show(camera) in a code cell."
                  if images == "on-demand" else "Automatic camera images are disabled.")
    return f"""You control one RoboDojo Test scene by writing Python against robo.

Follow the native task instruction below. Observe the current scene, write a
short action policy, execute it through python(code), inspect the physical
feedback, and adjust within this same scene. Variables persist across cells.
Use the supplied grasp tool when appropriate; it is a bounded physical operation,
not a task-completion oracle. A successful grasp does not replace the evaluator.

You have at most {max_turns} Python calls and the native action allowance in the
task card. Each motion can spend multiple native actions. No scene reset is
available. Do not access backend internals, host files, hidden layouts or other
episodes. Use only the documented robo API and ordinary computation.
Python returns stdout/errors and structured trailing values ({feedback} feedback).
{image_text}
When the goal is reached or you cannot progress, call finish(status, summary).
The benchmark alone determines environment_success; your finish status is separate.

The same frozen strong_grasp tool is supplied for every scene:
robo.grasp(target, arm='auto', lift_m=0.12, max_attempts=2, max_steps=160,
           grasp_xyz=None, yaw=None, pitches=(90,65,50))
Read its structured report. An invalid or failed grasp is useful feedback.
The global grasp(target, **kwargs) is an equivalent convenience wrapper.

# Robot API
{api_reference(adapter.robot_cls, primitives='no-vla')}

# Supplied robot knowledge
{knowledge_text(adapter.knowledge, primitives='no-vla')}

# Native task card
{card.render()}
"""
