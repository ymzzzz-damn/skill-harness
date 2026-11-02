# Conveyor task conditions experiment

This deployment compares the same harness source with `tool_set="pyrua"` and
`task_conditions=True` versus `False`. Both arms use the unchanged original
PyRUA-Lean API, perception and motion tools at commit
`ffad6d439f926eea450e9e3d3b3a106438b141b1`. Neither arm imports or installs
the custom grasp package. The pre-change harness is preserved at GitHub commit
`ec2b94afb0161e6aab91d213221d33f5797fee86`; this experiment's control is the
updated runner's conditions-disabled ablation with original PyRUA tools.
The historical PyRUA result, 11/55 (11/52 valid), is an additional reference,
not this experiment's paired control.

The user requested this fixed order: treatment-only smoke on layouts 3 and 21,
then all 55 treatment layouts, then all 55 control layouts. Smoke has separate
fresh episodes and is excluded from full results. Each layout always uses lane
`layout % 8`; both full stages stay in the same Pod and use the same physical GPU
UUID. The order is not randomized; cache warming and stage-time provider changes
remain possible confounds. These seed0 results do not measure unseen-layout
generalization.

Controls: `pick_from_conveyor_by_image`, eval seed0, layouts0–54,
`gpt-6-astra/high`, 100 Python cells, 5400s Agent budget, native700 actions,
Codex0.159.2, RGB/no-vla/cells/on-demand/pure. Each scene has a new Agent,
Python namespace, simulator and Codex home. No training or cross-scene memory.
Both arms disable the same Codex features: `shell_snapshot`, `memories` and
`shell_tool`. Actual generation and runner receipts are checked for this policy.
The pinned CLI's read-only capability check is recorded in `ORIGINS.json`.
Failures after the first observation, model phase or native action are never
replayed. Only recognized initialization errors with no such evidence and
confirmed native process cleanup can retry, at most three initializations;
all earlier bytes and elapsed time remain archived.

## Files and dependency boundaries

Remote project: `/mnt/auwomo-data/yumuzhen/harness/`.
Deploy this directory to `deployment/20261102_conveyor_conditions/` and the
reviewed framework to `source/`. Results are written only under
`experiments/20261102_conveyor_conditions/{smoke,full}/{conditions,control}/`;
runtime files belong to `runtime/20261102_conveyor_conditions/`.

PyRUA-Lean supplies its existing Python environment, SAM3 source/checkpoint,
RoboDojo source/assets, Codex executable and private provider configuration.
The registered SAM3 editable source is
`PyRUA-Lean/runtime/vendor/sam3-c1f6b1e83967c912f675d8975d437ccdc41e17d8/`.
Preparation validates and freezes its `direct_url.json`, `.pth` loader and
finder mapping using plain metadata reads; it never executes the container-only
venv Python on the access host. Bootstrap later checks the imported SAM3 path
inside the runner container.
The private configuration path is supplied through `HARNESS_PLANNER_CONFIG`;
values are read only by each activated episode and are never included in source
or manifests. `registry-secret` is referenced by name as an existing image-pull
Secret; no credential is copied. A reviewed local OpenCV4.10.0.84 wheel is reused
from the previous experiment and installed without dependency resolution or
network access into this harness's own runtime directory. `ORIGINS.json`
records first-party r6 startup/recovery adaptations and the Apache-2.0 PyRUA
runtime dependency. No third-party library source is vendored here.

The Job uses `default-scheduler`, requests eight GPUs and eight workers, and
accepts current product labels `NVIDIA-RPBZZZ6` or
`NVIDIA-GeForce-RTX-5090`. It specifies `preemptionPolicy=Never`, no fixed
hostname/nodeName, custom priority or additional taint tolerations. This shared queue is explicitly
authorized for this experiment. Bootstrap reads its own actual Pod through
the Kubernetes API, checks UID, node, phase, image and eight-GPU allocation
against the Downward API, then checks actual NVML/CUDA models and UUID mapping
before any simulator/model call. A dedicated ServiceAccount has only namespace
`get pods` permission for that identity read. Existing services are not changed.

## Prepare and review

CPU-only, no simulator or API model requests:

```bash
python3 -B deployment/20261102_conveyor_conditions/prepare.py --dry-run
python3 -B deployment/20261102_conveyor_conditions/batch.py --freeze unused --dry-run
python3 -B -m unittest discover -s deployment/20261102_conveyor_conditions -p 'test_*.py' -v
```

After uploading the reviewed source and deployment files, on the access host:

```bash
python3 -B /mnt/auwomo-data/yumuzhen/harness/deployment/20261102_conveyor_conditions/prepare.py
```

That command fingerprints all framework/deployment/PyRUA/SAM3 source files,
the selected native task/evaluator/launch files, all55 layout files, the Codex
binary and reviewed OpenCV wheel. It streams the3.45GB checkpoint hash once
and records its size and modification time for subsequent cheap verification.
It creates `freeze.json` and `rbac-and-job.json` without submitting them and
refuses to overwrite an existing prepared deployment. Review them before
applying the manifest through the `p6000` access host:

```bash
kubectl --kubeconfig=/root/.kube/p6000-user.conf apply -f /mnt/auwomo-data/yumuzhen/harness/deployment/20261102_conveyor_conditions/rbac-and-job.json
```

Applying the reviewed manifest activates the already-authorized experiment
when normal scheduling allocates a node. It does not bypass the shared queue.
The native startup fix isolates per-episode Kit logs/configs and per-lane caches,
reuses the existing read-only extension cache and limits concurrent native
initializations to two. Evidence/runtime checks, not task score, gate the smoke.

## Progress and comparison

`batch_status.json`, `workers/`, `stage_times.json`, `smoke-report.json` and
`full-report.json` live in the experiment output directory. Reports can also be
rebuilt read-only at any time:

```bash
python3 -B /mnt/auwomo-data/yumuzhen/harness/deployment/20261102_conveyor_conditions/report.py
```

The primary episode metric is `task_wall_s`: initial task card and robot tools
ready to first observed native end/conclusion, including planning, Agent/model
and control time. Scene and Tool initialization are excluded; the first captured
simulator RGB timestamp remains separate in initialization receipts.
Startup/sceneload, policy time, Tool time
and total lifecycle time remain separate. Sums of parallel episode durations
are not stage elapsed wall time; `stage_times.json` records the latter. Reports
provide all attempted and valid-only counts, missing/invalid native outcomes,
per-layout mean/median and sums, action/cell/model accounting, paired gains/losses,
seeded paired bootstrap intervals and exact McNemar p-values. Missing timing or
model-accounting data remain null, never silently zero. Task failure is distinct
from invalid execution. Native result task/layout identity, score scales and
framework agreement are checked independently of success.
