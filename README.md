# 自研 harness：抓取 Tool 对照测试

源码位于 /mnt/auwomo-data/yumuzhen/harness/source。框架独立编排每一场 Test：
原生场景初态与指令 → Agent 生成 Python 动作代码 → 持久 Python 执行 →
抓取/运动工具 → 结构化结果与相机反馈 → Agent 调整 → 原生任务评分和录像。

每个场景新建 Agent、Python 命名空间与模拟器；同场内允许检查、补救和重写代码。
Test 无 reset、无跨场景学习，不自动重放失败场景。finish 的自报结果与
environment_success 分开，后者来自原生 evaluator。提供的抓取工具被冻结供各场使用，
本版本没有凭空增加训练或 Playground 成果。

## 代码边界

- harnesslab/policy.py：自己的持久 Python 执行、decision budget、反馈与落盘。
- harnesslab/agent.py：自己的 Agent 进程、结束/超时边界、事件和用量汇总。
- harnesslab/prompt.py：自己的任务协议与工具说明。
- harnesslab/runner.py：自己的 episode 生命周期、依赖哈希与原生最终结果。
- tests/：CPU 假机器人/假运行时验证，不调用模型或 GPU。

与 PyRUA-Lean 共用明确的软件依赖：冻结源码中的 RGB Robot API、几何/运动/
感知 primitive、native host/backend、XPolicyLab bridge、sandbox guard、API 文档、
机器人知识文本、MCP wire server 及 provider/catalog 参数兼容函数。
这里不调用 pyrualean.play.play_episode 或其 ArmToolkit。
原生 RoboDojo/XPolicyLab evaluator 继续拥有物理步进、动作上限、任务评分及视频。
共享 strong_grasp 源码在 /mnt/auwomo-data/yumuzhen/exp/grasp_tool_20261029/source/strong_grasp。
安装发生于 host.boot 后、Robot 构造及 API 文档生成之前。
模型运行时、SAM3 和 checkpoint 由 PyRUA-Lean/runtime 提供，认证由部署入口在运行时读取；
不在本仓库或结果目录保存真实凭据。源码路径和每个 Python 文件的 SHA256 写入每场结果。

## 部署接口与预检查

配对测试入口由 exp/grasp_tool_20261029/deployment/run_episode.py 管理。
框架调用接口为 harnesslab.runner.run_episode(cell, out_dir, **kwargs)，兼容配对 runner
使用的 cells/codex/no-vla/RGB Test 参数，未知参数拒绝而非静默忽略。

PYTHONPATH 同时包含本 source、共享 Tool 的 source 及冻结 PyRUA source/src。
python -m harnesslab.cli framework-check 只检查导入，不启动模型/仿真。
python -B -m unittest discover -s tests -v 运行 CPU 验证。

默认 rich/on-motion 可自动回传结构化物理结果及运动后图片；配对测试会显式传入
预注册的 images/feedback（可与 PyRUA 的 on-demand/pure 一致），各自写入 result.json。
这个反馈选项不能在比较时隐瞒。代码策略末尾的 dataclass 结果始终转成结构化 JSON。
纯代码 cells 的执行成本、知识和新工具不同于 EmbodiedEvo 的四 primitive 协议，
流程相同不代表各系统每一次决策的动作粒度相同。

GPU 固定 worker 192.168.48.153；运行前必须由部署入口核验实际 Pod 的 spec.nodeName。
SSH 访问仍为 p6000。节点不可用时不自动换节点。本次创建/CPU 检查不代表已执行真实 Test。
