# skill-harness

独立的 RoboDojo 代码 Agent 框架。每个 Test 场景创建新的 Agent、Python 命名空间和模拟器；同一场景内保留变量、观察反馈并调整代码。原生 RoboDojo/XPolicyLab 负责物理步进、动作预算、评分和录像。Agent 的 finish 声明始终与原生结果分开记录。

P6000 项目目录为 /mnt/auwomo-data/yumuzhen/harness/：source/ 保存本仓库源码，experiments/ 保存运行产物，deployment/ 保存实际部署副本。运行时和凭据不提交到 Git。停止中的历史训练、seed0 队列和自动化保持原状态。

## 版本与 Tool

修改前源码已完整提交为 ec2b94afb0161e6aab91d213221d33f5797fee86。v0.2 增加通用任务条件管理，可通过 task_conditions=False 关闭。为隔离模块效果，消融两组运行同一份 v0.2 runner，仅改变这个开关；同时固定下述 Tool 和运行配置。修改前提交仍保留原来的强抓取 Tool 默认行为，不能直接把它当作原始 PyRUA Tool 的对照。

harnesslab.runner.run_episode(..., tool_set="pyrua") 完全使用原始 PyRUA Robot API：不导入、安装或提供 strong_grasp，不提供全局 grasp，不更改原始机器人 primitive。工具源为 PyRUA-Lean commit ffad6d439f926eea450e9e3d3b3a106438b141b1。原有 tool_set="strong-grasp" 入口保留用于已授权的旧抓取协议，不用于本次传送带实验。

## 条件管理

Agent 根据任务语言、初始 RGB 和公开观测注册五类条件：最终目标、子任务前置条件、阶段完成条件、持续保持条件、收尾要求。每项记录来源和证据要求，区分一次性事件与当前状态，并明确依赖、同时成立关系和歧义。

元数据附在现有 python/finish 接口上：task_plan、task_update、task_intent，无需新增机器人 Tool。原生 Toolkit 的实例级调用入口在每个代码单元首次物理动作前检查计划和前置条件；之后每次物理 dispatch 都使历史状态证据失效。一个代码单元可以包含一段有界的多 primitive 操作；它的内部动作不会逐个重新证明保持条件，因此应保持阶段简短并在阶段间检查。

条件状态为已满足、未满足、无法确认。证据引用必须指向已存在的观测单元，当前状态还必须匹配当前物理修订和原生步数。finish(success) 需要新鲜的最终条件评估。框架校验结构、引用与新鲜度；视觉语义仍由 Agent 判断，不等于真值证明。运行时不读取隐藏评测源码、目标对象真值或其他场景结果，没有针对传送带或目标高度的硬编码。原生终止不会因 Agent 未完成声明而改写为任务失败。

每场保存 task_metadata/、task_contract.json 和 task_condition_events.json，包括失败的计划提交、动作边界、旧状态失效和完成声明拒绝。任务条件协议优先于 SDK 通用知识中的无条件松手、回位建议；原始知识文本不改动。

## 图片上下文管理（v0.2.1）

当前实验使用显式 base_url 的 Responses 兼容模型接口。v0.2.1 在模型请求发出前裁剪图片历史，默认 max_context_images=48；这是针对本轮服务实际返回的 50 张限制预留余量，并非 OpenAI API 的通用上限。

每次请求保留初始任务的全部参考图片，以及最新一次带图观察的完整图片组；其余名额优先保留近期图片。较早的观察图片替换为明确的“已移出上下文、原图仍在实验产物中”文字占位。裁剪覆盖普通消息和工具返回值中的图片，不改文字反馈、工具调用 ID、代码参数、条件状态或推理内容，也不删除磁盘上的原图、录像、反馈和会话记录。不进行额外模型摘要调用。

实现遵循 [Responses 手工管理会话历史](https://developers.openai.com/api/docs/guides/conversation-state) 的输入结构。独立的本机转发层只连接配置好的上游，子进程使用临时本机凭据；请求使用 HTTP/SSE 和未压缩 JSON。服务端隐式历史、不支持的图片结构、初始参考图缺失或保护图片超出上限均明确报错，不能绕过裁剪继续发送。max_context_images=None 可显式关闭裁剪；没有显式 base_url 的 Codex 账户认证入口保留原行为，并记录 enabled=false、reason=no_explicit_responses_provider，该入口不声称提供图片数量保证。

每场保存 image_context_requests.jsonl（数量、被移出的位置和错误码）及 image_context.json 汇总，同时写入 generation.image_context。审计不保存图片内容、请求凭据或推理内容。未来修改组与消融组使用同一图片策略，条件模块开关仍单独控制。

## 2026-11-02 传送带对照

任务：pick_from_conveyor_by_image。按用户指定顺序：修改组 smoke（独立布局 3、21）→ 修改组全部 55 布局 → 关闭条件模块的消融组全部 55 布局。smoke 不进入正式统计。固定阶段先后是本次设计限制，没有做组先后随机化。

两组均为原始 PyRUA Tool、gpt-6-astra/high、seed0、布局 0..54、700 原生动作、100 次 Python 决策、5400 秒、RGB/no-vla、on-demand/pure；保持相同布局的 GPU lane 映射。无跨场景学习和控制阶段失败重放；仅允许满足严格证据条件的无首帧、无模型、无动作初始化失败有限重试。

主要指标为原生成功数/55，另报有效运行成功率、接口/基础设施错误与成对结果。历史 PyRUA 的 11/55（排除接口错误 11/52）只作为历史参考；模块的消融对照来自本次两组新结果。

该批次两个 smoke 实际因模型接口的 50 张图片限制中断，正式两组尚未启动。v0.2.1 的图片历史修复不改写该批次结果；后续正式实验需要建立包含新源码和统一图片策略的新冻结批次。

task_wall_s 从初始任务卡、RGB 与机器人 Tool 全部就绪计至首次观察到原生结束或显式结束完成，包含框架规划、模型推理、Tool 控制和反馈；排除 GPU 排队、场景与 Tool 初始化和清理。若在 Agent 内发生原生结束，轮询首次观察到的时间为边界。异常场景保留耗时和终点未确认标志。policy_wall_s 保留用于与历史 Agent 运行耗时比较；startup_s、total_wall_s 单列。汇总分别给出每场耗时之和与并行阶段的实际经过时间，避免混为同一指标。

用户于 2026-11-02 取消固定 .153 限制：本轮采用 P6000 / RTX 5090 共享队列、8 张卡、普通调度。启动前核验实际 Pod UID、节点和 GPU 型号/数量/UUID。

## 代码与依赖

- harnesslab/policy.py：持久 Python、预算、反馈、条件协议和落盘。
- harnesslab/task_contract.py：条件结构、证据、当前状态和事件生命周期。
- harnesslab/agent.py：Agent 进程、结束/超时边界、事件和用量汇总。
- harnesslab/image_context.py：发给模型的图片历史裁剪、HTTP/SSE 转发和数量审计。
- harnesslab/prompt.py：任务协议与原样引入的 SDK API/知识。
- harnesslab/runner.py：独立 episode 生命周期、原生调用入口检查、依赖哈希和计时。
- deployment/20261102_conveyor_conditions/：冻结版本、共享队列、阶段运行和汇总脚本；部署后写入项目自身目录。
- tests/：CPU 假机器人/假运行时检查，不调用模型或 GPU。

PyRUA-Lean 只作为明确的依赖提供 RGB API、运动/感知 primitive、native host/backend、XPolicyLab bridge、sandbox guard、API 文档、机器人知识、MCP wire server 和 provider/catalog 参数兼容函数。本框架不调用 PyRUA episode runner 或 ArmToolkit。每场记录各依赖 Python 文件 SHA256。

部署复用 /mnt/auwomo-data/yumuzhen/PyRUA-Lean/runtime/ 的机器管理环境、SAM3/checkpoint 和 RoboDojo 执行资源；原生 runner 镜像与执行资源源自既有 EmbodiedEvo 部署。认证由部署入口从所属项目真实私密配置路径读取。实验部署说明记录具体冻结路径和依赖，不复制真实凭据到源码或输出。

CPU 测试：PYTHONPATH=.:/path/to/pristine/PyRUA-Lean/src python -B -m unittest discover -s tests -v。
只读依赖预检：python -m harnesslab.cli framework-check --tool-set pyrua。

CPU 检查与提交到队列均不代表已获得原生任务分数；成功率与耗时以对应实验产物为准。
