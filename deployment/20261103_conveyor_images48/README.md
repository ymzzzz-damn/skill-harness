# Conveyor conditions with bounded image history

This cohort runs the image-management fix in harness0.2.1 with a real48-image
request limit. It compares `task_conditions=True` and `False` using exactly the
same unchanged original PyRUA tools (`ffad6d439f926eea450e9e3d3b3a106438b141b1`,
Apache-2.0). Framework origin is `0e6080a269273c01aeac95ed4aba498a332b7035`,
source tree `f549ccbc02fa0c460e4659578f3503bac5b0d535`; the final publication
commit can be recorded by `HARNESS_PUBLICATION_COMMIT` during preparation.
The freeze's complete source hashes, rather than a mutable remote HEAD, govern
execution. The control is the updated framework's conditions-disabled ablation;
`ec2b94afb0161e6aab91d213221d33f5797fee86` preserves the original behavior.
Historical PyRUA11/55 (11/52 valid) is context, not the new paired control.

There is no separate smoke. Eight workers immediately receive eight distinct
formal treatment cases, then take pending cases from a shared FIFO queue. The
assignment of each layout to GPU lane, physical UUID, Pod UID and node is
atomically persisted **before launch**. Only after all55 treatment cases have been attempted and closed does control start. Control preserves each layout's
hardware and orders cases by descending observed treatment duration within that
lane. Each lane has at most one simulator and Agent. Boot concurrency is4.
This reduces idle gaps caused by uneven treatment durations; the control's
fixed hardware mapping can still have a final lane imbalance. Stage order is
user-requested, not randomized. Seed0 comparisons do not establish performance
on unseen layouts; cache warming and time-varying provider conditions remain
possible confounds.

Both arms use `pick_from_conveyor_by_image`, eval seed0/layouts0–54,
`gpt-6-astra/high`,100 cells,5400s Agent budget,700 native steps, Codex0.159.2,
RGB/no-vla/on-demand/pure. Each scene has a fresh Agent, Python namespace,
simulator and Codex home. Both disable `shell_snapshot`, `memories` and
`shell_tool`. Neither imports strong_grasp, trains or carries scene memory.
`run_episode.py` passes `max_context_images=48` to the actual runner. Reports
verify saved and generation proxy metadata, fixed Responses/no-WebSocket/
no-request-compression transport, preserved3 initial references, all forwarded
requests including late terminal audit rows, and the frozen implementation
hashes. Native stop can cancel a streaming request: pending/disconnected proxy
rows remain visible and do not independently invalidate an otherwise valid
native outcome. Actual runtime fatal/error classification remains authoritative.

There is one Agent process per case, with no implicit scene/Agent replay. The
underlying provider's request/stream retries remain10; the image proxy performs
no automatic replay. Only recognized initialization failures with no first
observation/RGB, model phase, native action, task card, native report or video,
and proven client cleanup may retry, at most3 initializations. Original bytes
and timing are archived. A normal native score0 and case-specific API/planner, missing native report,
inspection or video errors are recorded and the remaining cases continue.
Only a global source/control, allocation/UID/GPU/mapping invariant failure or
unconfirmed native-process cleanup stops new dispatch and the next stage;
already active cases finish. Invalid outcomes retain their own denominators.

## Runtime and scheduling

All remote code, output and runtime files live within the independent harness
project `/mnt/auwomo-data/yumuzhen/harness/`:

- `deployment/20261103_conveyor_images48/`: reviewed scripts, manifest and freeze.
- `experiments/20261103_conveyor_images48/full/{conditions,control}/`: formal cases.
- `runtime/20261103_conveyor_images48/`: lane caches, leases, dependencies and health receipts.

PyRUA-Lean supplies its existing environment, original tools, SAM3 checkpoint
and pinned editable SAM3 source, RoboDojo assets/native runner, Codex binary and
private provider configuration. The existing local OpenCV4.10.0.84 wheel is
installed into the harness's own runtime without dependency resolution/network.
`HARNESS_PLANNER_CONFIG` references the actual PyRUA `.secrets` path; values are
never embedded in this repository or receipts. `registry-secret` is referenced
only by name. Read-only shared assets are mounted separately from the writable
harness project. The execution dependency and copied first-party adaptations are
recorded in `ORIGINS.json`.

Before any case, bootstrap copies the already verified URDF package
`isaacsim.asset.importer.urdf-2.4.31+107.3.3.lx64.r.cp311` from the prior harness
runtime into this run's own cache and creates all8 unused lane entrances.
The canonical tree SHA256 is
`e5709552257704c948b0f5cd3ec83ba2d271d69c50c06723b889594cfee54b34`.
Preparation and bootstrap recalculate it. Existing differing, broken or real
entries are never replaced automatically. SDK source and package versions do
not change.

The Job `yumuzhen-conveyor-images48-20261103` uses the standard priority0 shared
queue,8GPUs, requests48CPU/256Gi, limits64CPU/384Gi, and accepts product labels
`NVIDIA-RPBZZZ6` or `NVIDIA-GeForce-RTX-5090`. It has no positive fixed-node
selector, custom PriorityClass, explicit Pod preemptionPolicy or added taint
tolerations. The cluster controls ordinary admission/preemption; the deployment
never manually evicts other workloads. The required affinity excludes
`192.168.48.173` because the2026-11-03 direct audit found GPUUnhealthy=True even
though Ready=True. This negative health filter does not target another node.

Bootstrap first reads its actual Pod API and matches Downward API, owning Job
UID, image and allocation. It completes CPU imports, cache work and CPU tests,
then writes `runtime/.../node-health/<PodUID>.request.json`. A separately launched
host guardian waits for that exact Job/Pod/Running allocation and marker, then
reads the selected Node once with the host's existing kubeconfig. Bootstrap
requires the matching healthy receipt within300s and age<=180s before any CUDA
probe, simulator or model. The Pod SA only has namespace `get pods`; no host
kubeconfig or nodes privilege enters it. Node health failure is not overridden
by a small CUDA probe.

## Prepare and activate

CPU-only local checks (no simulator/model/API inference):

```bash
python3 -B deployment/20261103_conveyor_images48/prepare.py --dry-run
python3 -B deployment/20261103_conveyor_images48/batch.py --freeze unused --dry-run
python3 -B -m unittest discover -s deployment/20261103_conveyor_images48 -p 'test_*.py' -v
```

Upload the reviewed deployment and source under the remote harness project.
Run `prepare.py` with the access host's Python3; the policy-venv interpreter is
container-only. Preparation writes `freeze.json` and `rbac-and-job.json` and
never submits. It verifies all55 layouts, native budget, original PyRUA Git
cleanliness, SAM3 editable registration, checkpoint, wheel, binary, source and
URDF hashes. It refuses to overwrite prepared files. Review and server-dry-run
the manifests before applying through `p6000` with the existing kubeconfig.
After obtaining the actual Job UID, start the deployment's host guardian:

```bash
python3 -B /mnt/auwomo-data/yumuzhen/harness/deployment/20261103_conveyor_images48/node_health.py \
  --job-name yumuzhen-conveyor-images48-20261103 --job-uid ACTUAL_JOB_UID \
  --namespace dreamzero-training --kubeconfig /root/.kube/p6000-user.conf \
  --receipt-dir /mnt/auwomo-data/yumuzhen/harness/runtime/20261103_conveyor_images48/node-health
```

The guardian has a24-hour bounded wait and exits after the request/receipt
exchange. This is part of this deployment, not a recurring Codex automation.
Do not resubmit or restart a measured batch to recover ordinary task failures.

## Evidence and utilization

`batch_status.json`, `worker-live.json`, `treatment-mapping.json`,
`control-plan.json`, `stage_times.json`, workers' records and `full-report.json`
show progress and pairing. Reports independently verify actual tool/native
handler hashes, conditions guards, GPU UUID/Pod/node/layout identity, native
result schema, timing, and first/last decode of all3 video cameras. Pending
cases are not labeled as completed failures or zero-second measurements.

The lightweight sampler records NVML utilization/memory, case/stage/lane,
assigned worker count and cgroup memory/CPU throttling counters every30s under
`utilization/`. Native control in flight is labeled separately; the Agent phase
includes model wait, Python and perception and cannot prove a particular model
request is active. Hourly sampled utilization means and GPU-active fractions
are separate from worker-assigned fractions. An assigned worker can legitimately
wait on the LLM with low GPU utilization. The monitor creates no artificial
load or simulator/thread/budget changes. It records metadata errors separately.

`task_wall_s` is monotonic from initial task card/tools ready to the first
observed native end/conclusion, including LLM and control. Initialization and
teardown remain separate. After each stage (and batch exceptions), the copied
stdlib helper writes `scene-excluded-timing.json`: a clearly labeled mixed-clock
estimate from first RGB to native end, including the Tool-readiness gap and
excluding scene loading. Original results are untouched. Reports give attempted
and valid-only denominators, sums/means/medians, native/action/model accounting,
paired gains/losses and uncertainty. Sums of parallel case durations are not
stage elapsed wall time. The fixed-order and seed0 limitations remain explicit.
