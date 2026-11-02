"""Own task policy instructions; SDK API/knowledge are explicit dependencies."""
from __future__ import annotations
from typing import Any

def build_prompt(adapter: Any, card: Any, *, max_turns: int,
                 images: str, feedback: str, task_conditions: bool = False,
                 tool_set: str = "strong-grasp") -> str:
    from pyrualean.prompt import api_reference, knowledge_text
    if tool_set not in ("strong-grasp", "pyrua"):
        raise ValueError("unknown tool_set")
    image_text = ("Current camera images accompany code cells that move the robot."
                  if images == "on-motion" else
                  "Request camera images with robo.show(camera) in a code cell."
                  if images == "on-demand" else "Automatic camera images are disabled.")
    supplied_tool = ("""The same frozen strong_grasp tool is supplied for every scene:
robo.grasp(target, arm='auto', lift_m=0.12, max_attempts=2, max_steps=160,
           grasp_xyz=None, yaw=None, pitches=(90,65,50))
Read its structured report. An invalid or failed grasp is useful feedback.
The global grasp(target, **kwargs) is an equivalent convenience wrapper.
""" if tool_set == "strong-grasp" else
        "Use the original PyRUA Robot API documented below. No additional robot manipulation primitive is installed.")
    conditions = CONDITION_PROTOCOL if task_conditions else ""
    grasp_policy = ("Use the supplied grasp tool when appropriate; it is a bounded physical operation,\n"
                    "not a task-completion oracle. A successful grasp does not replace the evaluator."
                    if tool_set == "strong-grasp" else
                    "A successful primitive does not by itself establish overall task completion.")
    return f"""You control one RoboDojo Test scene by writing Python against robo.

{conditions}

Follow the native task instruction below. Observe the current scene, write a
short action policy, execute it through python(code), inspect the physical
feedback, and adjust within this same scene. Variables persist across cells.
{grasp_policy}

You have at most {max_turns} Python calls and the native action allowance in the
task card. Each motion can spend multiple native actions. No scene reset is
available. Do not access backend internals, host files, hidden layouts or other
episodes. Use only the documented robo API and ordinary computation.
Python returns stdout/errors and structured trailing values ({feedback} feedback).
{image_text}
When the goal is reached or you cannot progress, call finish(status, summary).
The benchmark alone determines environment_success; your finish status is separate.

{supplied_tool}

# Robot API
{api_reference(adapter.robot_cls, primitives='no-vla')}

# Supplied robot knowledge
{knowledge_text(adapter.knowledge, primitives='no-vla')}

# Native task card
{card.render()}
"""


CONDITION_PROTOCOL = """TASK CONDITION PROTOCOL — applies before the general operating knowledge.

Before physical execution, classify the task into a structured task_plan using
only the native task language, initial RGB, and documented public observations.
Never inspect evaluator code, hidden object state, host artifacts, or other trials.
Explicit task conditions take priority over general advice below, including any
blanket instruction to open grippers, release objects, or home both arms. Cleanup
is required only when justified by this task; do not undo a necessary held state.

Classify five kinds of conditions:
* final: the conditions that establish completion of the whole instruction;
* precondition: what must be true before a subtask starts;
* completion: what evidence establishes that a subtask achieved its purpose;
* maintain: a state that must continue to hold through specified later subtasks;
* cleanup: task-required finishing behavior, if any. Set required=true only when
  the instruction requires it; an empty cleanup set is legitimate.

For every condition distinguish mode='event' (it happened once) from mode='state'
(it remains true now). Do not interpret "was achieved" as "is still true".
Check dependencies and decide which final conditions must hold simultaneously.
For coupled manipulation, retain earlier support/holding/position conditions
until the final relation is established and checked. If the wording leaves this
uncertain, record the ambiguity and a conservative assumption in the plan;
do not silently declare a one-time achievement sufficient or invent hidden rules.

The python tool accepts bookkeeping metadata, without changing any Robot Tool:
task_plan = {
  "conditions": [{"id": "goal", "description": "task-specific final condition",
    "kind": "final", "mode": "state", "source": "instruction",
    "source_text": "relevant words from the task", "evidence_required": "what public evidence establishes it"},
    {"id": "phase_done", "description": "this phase achieved its purpose",
    "kind": "completion", "mode": "event", "source": "inferred",
    "source_text": "why this phase is necessary", "evidence_required": "relevant physical and visual checks"}],
  "subtasks": [{"id": "phase", "description": "one bounded phase",
    "preconditions": [], "completion": ["phase_done"], "maintain": []}],
  "simultaneous": [], "ambiguities": []
}
This abbreviated example shows the schema. Build the actual conditions and
dependencies for this scene; do not copy the generic condition descriptions.
Submit a valid task_plan before the first physical action. Observation-only code
is allowed before registration. A replacement plan is allowed but resets all
assessments to unknown; retain meaningful obligations when replanning.

Assessment metadata uses:
task_update = [{"id": "condition_id", "status": "satisfied|unsatisfied|unknown",
  "evidence": [{"cell": 0, "detail": "specific observation and its implication"}],
  "note": "remaining uncertainty if any"}]
Use ONE of the three status strings, not the example containing vertical bars.
Cell 0 is the supplied initial RGB at the initial native step. Other references
must point to completed cells with public observations, physical feedback, or
images. A state assessment needs evidence at the CURRENT physical revision and
native step. The framework checks references/freshness, not semantic ground truth.
If identity, grasp stability, placement, or the relevant relation is occluded or
ambiguous, mark unknown and obtain better observations. Tool confidence alone
does not prove the task-specific physical condition.

task_update is applied BEFORE code, so it can establish a subtask's prerequisites
from previous observations. Every physical dispatch invalidates prior state
assessments, while event achievements persist. After motion, inspect the fresh
feedback/images and reassess before the next bounded phase. Read-only observation
does not itself expire state. You may return a trailing dictionary containing
task_update, computed from public Robot return values, for assessments AFTER the
same cell. task_update_after metadata is also applied after the cell; never use
it to assert visual facts from images you have not yet inspected.

Each physical code cell also needs:
task_intent = {"subtask": "registered_phase_id", "purpose": "what this cell achieves",
               "preserve": ["all conditions this subtask must maintain"]}
The native dispatch guard checks the first physical action against the subtask's
current preconditions/maintain assessments and declared preserve list. Multiple
primitive calls inside that one cell are one bounded phase; keep cells short and
inspect feedback at phase boundaries. The guard cannot predict whether your
motion will preserve a condition: you must plan the geometry and check afterwards.

Before calling finish(status='success'), assess every final condition from current
evidence, including simultaneous groups and explicitly required cleanup. The
framework rejects unsupported/stale success declarations; failure/stuck remains
available. Agent assessments and finish claims never replace the native verdict.
If the native episode ends during a primitive, stop further motion. A truncated
flag alone does not tell you whether the task succeeded or ran out of actions;
preserve the partial evidence and let the native evaluator report the outcome.
"""
