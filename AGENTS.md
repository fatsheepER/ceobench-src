# 关于 PF 的读取压缩

读取压缩的 delta / cache read 是此前实验的主设计，目前已经确定在这个 Benchmark 中效果微乎甚微，这是意料之中的现象。所有的分析和报告里至多简单报告节省 tokens 的数值，不报告也可以，但不要浪费时间在检查和反馈这个机制没有起效果上。

# 实验结果文档

实验结果文档至少要包含 tokens 消耗比对、分析 PF 工具使用情况、整体 Multi-agent 框架执行情况，并结合具体轨迹中的例子说明结论。经营指标、token 用量和成本作为辅助证据。

- 分析以可调整的实验变量为重点：实验 harness、角色提示与交接、PF 接口及工具输出。优先查明它们是否传错、漏传或误导模型，是否把未检查、旧版本或不完整结果呈现为可信的当前结果。
- 每项问题须串起代码逻辑、实际调用与回执、接收方模型输入及后续行为，写清可控原因、证据强度和具体改进位置。区分已证实的错误输出、可能误导的表达，以及已观测到的行为影响；不能仅凭后续决策错误倒推工具有问题。
- 模型已拿到正确、完整、表达清楚的资料后发生的纯算术、推理或经营判断错误，不作为报告重点、PF 缺陷或实验设计缺陷，也不据此提出改进模型能力的建议。这些不是本实验的变量。只有证据表明 harness 或 PF 的输入输出促成了错误时，才纳入对应问题。
- 工具未使用、提醒未处理等现象，先核查能力是否可发现、调用契约是否清楚、回执是否提供足够的范围与状态信息、框架是否支持后续处理，再判断可控问题；不要只归结为模型能力不足。
- 明确的表格对比配对组的 tokens 消耗、耗时。
- PF / 来源工具使用用表格呈现：按实验组与角色列调用/成功/失败。覆盖本批冻结 capability 下实际可用的 PF CLI 子命令、来源声明工具 `escalate`（若开启）、以及 CEO 的 `ask_analyst` 等；已关闭的文本登记工具不要列表或填 0。区分主动与自动；从调用日志与回执统计，续跑去重后计数。结合后续模型输入与决策，说明工具返回的信息是否被使用、起到了什么作用。
- Multi-agent 框架执行情况覆盖角色分工、CEO 追问、分析师交接及信息是否进入接收方输入，以及权限、并行执行、检查点、恢复、重试和停止条件。结合代码与执行日志判断异常的原因、严重程度，以及是否需要在下一轮实验前修复。
- 具体轨迹案例注明实验组、日期和角色（证据位置统一放到报告末尾），围绕上述可控问题串起工具调用、返回信息、角色决策或修订与实际结果，据此说明对后续实验执行与 PF 系统设计的影响。
- 以上是文档中最基本的组成部分，不代表其他重要内容不提交。

# 报告写法

- 面向用户的实验分析文档（实验结果报告及其解读，不含验收报告）每次都须解释本次涉及的经营术语和业务机制。在术语首次出现或案例开头，用朴素语言说明客户群、套餐、模型档位、容量、获客、续费、成本与预测等实际涉及的概念，区分容易混淆的对象，并写清数值的单位和口径。默认用户熟悉 Benchmark 的公司设定与总体目标，重点补足理解本次业务所需的细节，不假定用户记得此前报告的术语解释。
- 案例先讲清角色正在解决什么经营问题、哪些事实和约束影响选择，再依次解释实际工具调用、返回内容、角色后续行动和结果。说明 PF 具体提供了哪份信息、它是否改变了后续工作；为修复登记或检索自身问题而产生的调用单独说明，不能只用工具名称或抽象功能概括作用。
- 报告中展示 PF 相关输出时，可以调整换行、缩进或省略部分内容，但必须在该段输出旁明确说明做了哪些展示处理，并标明省略位置；区分工具原始输出、实际进入模型的内容与报告中的摘录，不得把报告作者的排版或删节写成工具本身的行为。
- 正文直接写结论和理由，不要在段落间穿插「证据在 + 文件路径」。需要的证据位置在报告末尾用一个简短的「证据位置」列表集中列出。
- 不要把证据压缩包、日志打包或原始输出上传到 Notion。证据留在 sheep-rog 上，报告里写明位置即可；证据只需可追溯，不要求完整附上。

# 运行可靠性修改

- 修改超时、取消、重试、并行执行、证据捕获或检查点时，先追踪实际公共入口、宿主工作、回执、模型发送与 attempt 收尾。明确工作归属、结束证明，以及允许继续或必须暂停的条件。
- 分别处理服务端结果、响应发送、客户端接收和后台资源关闭。故障现场保留在原 attempt；新 attempt 从操作者选定的检查点恢复，检查点后的操作全部作废。重复收尾接受一致结果，冲突结果明确失败。
- 工具和业务子调用保留现有的期限与资源关闭契约；模型网络重试不设置总等待预算或全局截止时间。取消须唤醒实际阻塞操作；清理与持久化有独立且有界的预算，外层强杀期限须容纳它们。监测和停止通路独立于被监测工作占用的资源。
- 回归从真实公共客户端和运行包装层进入。涉及模型发送时覆盖 `ModelUsage`，涉及停止时覆盖 controller；用本地 HTTP、真实子进程和假模型响应验证客户端先退出、服务端晚完成、断连、重复收尾及伙伴失败。断言下一次调用实际继续或按预期暂停、资源终态、attempt 状态与检查点恢复资格。
- SQL 收尾与 PF 期限从 `tests/test_sql_capture_residuals.py`、`tests/test_pf_deadline.py` 定位；停止与资源隔离从 `tests/test_team_stop_residuals.py`、`tests/test_preflight_memory_guards.py` 定位。新增故障测试须证明修复前会失败，再验证修复后的完整路径。
- HTTP 捕获调用编号的格式拒绝、完整失败回执、后续模型继续及真实捕获故障阻断，从 `tests/test_public_request_limits.py` 定位；shared-v1 下业务写入前拒绝与 Git/PF 路径从 `tests/test_shared_identity.py` 定位。
- 两组批量来源声明、精确文本/文件/产物/操作目标、缺失与修订、跨角色来源及存放身份从 `tests/test_relation_batch.py`、`tests/test_relation_contracts.py` 定位；真实公共操作引用、同值/部分结果与晚完成从 `tests/test_relation_operation_receipts.py` 定位；逐项回执入模、同调用去重、持久化未知 HOLD 与选点恢复从 `tests/test_relation_recovery.py` 定位。
- 双组公共来源完整闭环、跨周 Q0 恢复、后次 Bash 从实际查询文件生成 F1/F2、CEO 追问驱动修订、双向查询及旧版本读取、覆写后的跨角色采用与真实 ModelUsage 输入从 `tests/test_source_workflow.py` 定位。
- Git 文件登记即时快照、来源与存放角色、覆写后读取、Git 回收保留及 checkpoint 恢复从 `tests/test_git_registration_snapshot.py` 定位；prefix 登记时捕获匹配从 `tests/test_prefix_week_binding.py` 定位。
- Git/PF 共享业务流水、实际提交段、同值设置、公开列、旧世界补表与恢复引用从 `tests/test_business_ledger.py`、`tests/test_business_ledger_public.py`、`tests/test_business_ledger_faults.py` 定位。公共来源分类来自角色的模型与注册脚本独立监听器。
- PF 跨调用/跨周候选、同内容多事件、稳定分页、详情、权限复核、模型投递范围与选点恢复从 `tests/test_pf_candidates.py`、`tests/test_pf_candidates_team.py` 定位。
- SQL 授权表列观察、视图展开与同名 CTE、公开目录、失败/拒绝/截断/未知、稳定分页与权限复核、真实模型投递及 checkpoint 恢复从 `tests/test_sql_access.py`、`tests/test_sql_access_team.py` 定位；访问观察不是行级血缘或模型采用。
- 只读双向关系、旧文本与 Git 精确版本、操作目标、声明/推断/执行成员候选、稳定分页/实时权限/完整详情和模型投递/恢复从 `tests/test_pf_relations.py`、`tests/test_pf_relations_team.py` 定位。
- 操作短编号的公共回执、SQL 与关系查询、跨角色一致性和回退不复用从 `tests/test_operation_refs.py` 定位；独立运行的编号共用、重复恢复和权威缺失从 `tests/test_private_record_recovery.py` 定位；编号权威缺失或冲突时的 controller 与独立包装层 HOLD、选点保留和零新增模型请求从 `tests/test_reference_hold.py` 定位。
- 登记原件与工具笔记私有、公开完整修订、旧 checkpoint 历史隔离从 `tests/test_private_records.py` 定位；PF 精确短引用、同内容采集、note 检索、分页与权限复核从 `tests/test_pf_public_projection.py` 定位；精确采集 note、普通读取及 Git 独立输出快照从 `tests/test_note_visibility.py` 定位；Git 短哈希唯一性和嵌套证据投影从 `tests/test_public_reference_details.py` 定位。
- 脚本与角色请求的公开回执从 `tests/test_public_framework_receipts.py` 定位；PF 错误的私有诊断、公开状态及实际模型输入从 `tests/test_pf_public_errors.py` 定位；外层 Bash 捕获边界的初始提示、help 与两次调用从 `tests/test_pf_capture_guidance.py` 定位。
- 高频回执的摘要、完整捕获清单、设置差异与实际模型输入从 `tests/test_receipt_outputs.py` 定位；SQL 极小结果上限下的错误发送及公共客户端继续从 `tests/test_public_sql.py`、`tests/test_sql_capture_residuals.py` 定位。
- 职责写权完整入口矩阵、专家操作回执与互读路径从 `tests/test_responsibility_authority.py` 定位；跨角色历史与私有层隔离从 `tests/test_preflight_multi_agent.py`、`tests/test_preflight_team_evidence.py` 定位。
- 企业协商的客户标识、拒绝与七天超时，以及公共 SDK 的定向 ops 四范围和 dev 组映射替换契约，从 `tests/test_enterprise_sdk_contracts.py` 定位。
- 两阶段 D56 批次、种子排除、policy 枚举与单组选取、真实 driver 队列/自然终止/STOP/HOLD/人工恢复从 `tests/test_round2_batch.py` 定位；观察终点配对汇总从 `tests/test_round2_results.py` 定位。
- 第三轮固定 s1048/s1049 四条 D70 批次、能力清单、无副作用预检、真实 driver 队列与选点恢复、D70 后零请求/脚本、自然终止/STOP/HOLD 从 `tests/test_round3_batch.py` 定位；D70 配对汇总从 `tests/test_round2_results.py` 定位。
- 推进失败终态、客户端先退出及关闭交错从 `tests/test_advance_terminal.py` 定位。
- 真实 driver 的 chat/Responses 样本导出、工具结果与标记计数、空输入/零请求及重复收尾从 `tests/test_checkpoint_driver_samples.py` 定位；真实双 API 运行和恢复仍从 `tests/test_round4_batch.py` 定位。
- 显式历史选点、初始周界、真实 driver 重复恢复与旧 attempt 保留从 `tests/test_checkpoint_driver.py` 定位；必要恢复数据、空会话、周状态及 checkpoint 发布失败从 `tests/test_checkpoint_recovery.py` 定位。晚完成推进的共享资源收尾、清理超时后 controller HOLD 接管和恢复作废记账从 `tests/test_checkpoint_close.py` 定位；缺失继承用量、角色范围及跨目录祖先链的未知值从 `tests/test_checkpoint_usage_gaps.py` 定位。
- 自动周检、重放及状态传播已删除。周初看板、注册脚本、全文交接与检查点恢复的真实模型输入从 `tests/test_week_start_observation.py` 定位；主动来源查询不重放、不判断条件及旧 findings 不再投递从 `tests/test_pf_readonly_sources.py` 定位。
- 第三轮独立能力开关、旧调用拒绝后继续、Git 附加保存关闭、PF 非文本来源与历史读取、三角色真实模型输入和 `make_team`/checkpoint 配置恢复从 `tests/test_agent_capabilities.py` 定位。
- 第三轮按能力组合的 standalone、三角色首次及恢复 ModelUsage 提示、公共 PF help、真实引用的示例声明与修订，从 `tests/test_round3_prompts.py` 定位；旧文本提示与帮助分支继续保留。
- 公开来源工具 `escalate` 的改名、非文本逐项回执入模、旧名拒绝和历史声明恢复去重从 `tests/test_escalate.py` 定位；历史关系分类字段继续保留。
- 第三轮三角色旧名拒绝后纠正、非文本采集短引用、私有记录隔离、准确旧版本与真实 ModelUsage 输入的公共闭环从 `tests/test_round3_acceptance.py` 定位。
- 第四轮 PF 同伴路径归一、精确旧版本、越界与私有文件拒绝、逐端点错误及纠正、超过六项的执行捕获清单和实际 ModelUsage 输入从 `tests/test_round4_pf_contracts.py` 定位；来源持久化未知与重复回执仍从 `tests/test_relation_recovery.py` 定位。
- 第四轮六条 D56 分配、显式 agent/模拟器配置、Responses 推理与工具续传、独立 STOP/HOLD/恢复及无请求启动检查从 `tests/test_round4_batch.py` 定位；永久额度错误、普通限流、真实 HTTP 断连/取消与冻结请求监测从 `tests/test_round4_models.py` 定位。
- OpenDesign 双 choice 工具遗漏、流式结束证明、异常终态拒绝、原始回执与推理恢复从 `tests/test_opendesign_transport.py`、`tests/test_opendesign.py` 定位；`insufficient_balance` 永久余额拒绝与真实 controller HOLD/人工恢复从 `tests/test_round4_models.py` 定位。
- 第五轮 A01/A17 的 shared-v1 共享目录、职责权限、namespace 文件来源、Git 并发和三类保存、宿主撞名、私有观察缺口与角色记忆恢复，从 `tests/test_shared_workspace.py`、`tests/test_shared_identity.py` 定位；旧布局默认保留，checkpoint 不跨布局转换。
- 第五轮 A02/A03/A04/A05/A06/A09/A10/A11/A18 的 PF 稳定执行快照、独立比较基底、多文件原子发布、候选私有性、重复收尾及真实进程持久化故障，从 `tests/test_pf_edit_base.py`、`tests/test_pf_publication.py`、`tests/test_pf_publication_recovery.py` 定位；checkpoint 全链路留待 R5-1.4。
- 第五轮 A07/A08/A12–A16/A22 的版本确认、精确修订与权限差异、同 Bash 草稿处理后再编辑、作者周初提醒与周末显式确认、CEO 独立推进、三 provider 真实 ModelUsage、注册源码与当前公共快照、API 成功后零重放、私有 Git 对象隔离与持久化故障，从 `tests/test_round5_workflows.py` 定位。
- PF 独立周推进的 shell 引号、字面 `$`、续行、`cd /workspace &&`、混合写入拦截及草稿确认，从 `tests/test_round5_workflows.py` 定位；无关私有失败回执不得阻断公开 `escalate`、实际私有/越界目标仍拒绝及后续 ModelUsage 继续，从 `tests/test_round5_sources.py` 定位。
- 第五轮 A18–A20 发布静止点、必要权威与对象闭包、真实 driver 发布阶段杀进程/断连、controller 发布锁等待 STOP、含草稿/确认/脚本源码的历史选点恢复、已成功 API/脚本零重放、缺失发布权威零模型请求 HOLD，从 `tests/test_round5_checkpoint.py`、`tests/round5_checkpoint_fixture.py` 和 `tests/test_checkpoint_driver.py` 定位。
- 第五轮 A21 精确发布事件、跨调用比较基底、同内容回退来源、当前/历史/依赖与旧 namespace、私有候选事件及历史元数据隔离、真实 ModelUsage 作者来源绑定，从 `tests/test_round5_sources.py` 定位；私有原文与公共读取同请求的发送/记录继续从 `tests/test_round5_checkpoint.py` 定位。
- 第五轮 A01/A07/A22/A23 的共同主文件空模板、两组三角色首次/恢复/问答 ModelUsage、普通消息原样投递与文件内预测、+7/+28 到期现金摘要、问答及交接检查点的摘要去重、D56 零请求/脚本，从 `tests/test_round5_prompts.py` 定位；PF 快照中的 HTTP/SDK/CLI/Bash 子进程职责权限从 `tests/test_shared_identity.py` 定位。
- 第五轮 A20/A23/A26 四条 shared-v1 D56 主批、真实启动包装与零请求预检、实际执行源码/SDK/tokenizer 漂移拒绝、D28 全量保留和人工选点恢复、D56 零新增请求/脚本、独立 STOP/HOLD 与组队列，从 `tests/test_round5_batch.py`、`tests/round5_batch_fixture.py` 定位；保留/waste/pending/未知归属的分角色 tokens、费用和耗时表从 `tests/test_round5_results.py`、`tests/test_checkpoint_usage_gaps.py`、`tests/test_recovery_waste.py` 定位。
- 第五轮 A26 永久额度拒绝 `GoUsageLimitError` 的真实模型回执、controller HOLD、资源关闭与人工选点恢复从 `tests/test_round4_models.py` 定位；普通限流、持续重试和独立 STOP 仍从 `tests/test_model_recovery_retry.py`、`tests/test_continuous_run.py`、`tests/test_continuous_stop.py` 定位。
- 第四轮按能力的提示、PF help、跨角色精确来源示例、空反向查询，以及三角色首次/恢复的公开资料与定向研发公式回执从 `tests/test_round4_guidance.py` 定位；旧能力提示与外层 Bash 捕获仍从 `tests/test_round3_prompts.py`、`tests/test_pf_capture_guidance.py` 定位。
- CEO 周初开场、同会话收尾、开场追问、脚本顺序、更新看板与全文交接一次投递，以及开场追问和交接就绪检查点恢复从 `tests/test_ceo_weekly_opening.py` 定位。
- 三角色同步请求、FIFO、循环等待拒绝、整批工具后的模型投递、原会话唤醒与问答权限从 `tests/test_team_requests.py` 定位。
- 登记脚本的工作区相对路径、真实源码快照文件、`__file__`、相邻数据与模块、完整输出及关闭后清理从 `tests/test_registered_script_environment.py` 定位；自动批次只读经营接口、正常角色保留写权、拒绝回执入模和恢复不重放从 `tests/test_registered_script_observation.py` 定位。 旧客户端及帮助更新、真实旧 checkpoint 的嵌套登记和安全文件替换从 `tests/test_registered_script_legacy.py` 定位。
- 注册脚本每周一次、会话初始化与周任务/临时问答独立状态、脚本故障现场不得伪装周界，以及现有静止点恢复从 `tests/test_team_script_state.py` 定位；脚本执行、分类和关闭契约仍从 `tests/test_registered_script_reliability.py` 定位。
- 故障后该组进入 HOLD，记录原因并等待人工从 checkpoint 续跑。controller 每次启动对每个运行只启动一次 worker；正常完成后继续该组队列中的下一项，不自动回滚或重启，也不累计同位置失败次数。
- 人工续跑可以反复指定任意历史 checkpoint 或 `--week N`，含第 1 周的初始周界。历史 HOLD、STOP、审批与哈希信息只作提醒。snapshot、manifest、D0、环境、工作区、RNG、Bash 策略与 scope 校验只记日志及 attempt 提示；`--force-resume` 记录操作者明确选择继续。
- 全部 checkpoint 保留。故障现场保留在原 attempt，人工恢复使用新 attempt，并将选点后的执行记为作废。`scripts/team_driver.py` 支持旧配对批次和单组选择。每组默认 MemoryHigh 16G、MemoryMax 20G；主动 STOP 仍可停止当前执行。
- 回退作废轨迹的用量按组与角色记入 waste，配对比较只使用保留轨迹。组耗时按检查点继承链扣除作废执行段，角色耗时为模型逻辑请求耗时，缺失字段保持 null。定位见 `tests/test_recovery_waste.py`。
- 所有角色与模拟器默认持续重试模型网络错误（`model_request_attempts=0`），失败后间隔依次为 10、20、40、80 秒，此后保持 80 秒。显式正数参数仍表示有限尝试次数。首次回复最多等 600 秒，无回复超时记为一次网络失败；流式回复开始后不按累计时长截断。不设总等待预算或全局截止时间。成功响应中的无效内容进入有限纠错，每轮第四次模型请求后仍无有效回复则 HOLD。网络失败本身不因累计次数进入 HOLD。显式有限尝试耗尽、业务操作结果未知、必要状态无法保存或资源无法关闭时进入 HOLD，等人工续跑。已结束且结果明确、资源已关闭的脚本超时和非零退出把真实部分输出交回原角色，空 stdout 保持空；伙伴继续执行。不可重试的 HTTP 拒绝、提供方错误和配置错误直接记录 HOLD，显式 STOP 和操作者取消保留。缺失 usage、费用和耗时保持 null，不阻断执行。持续重试、失败输出与预测回滚终态定位见 `tests/test_continuous_run.py`，真实 controller STOP 定位见 `tests/test_continuous_stop.py`。聚焦检查定位见 `tests/test_model_recovery_retry.py`、`tests/test_manual_hold.py`、`tests/test_resume_advisories.py`、`tests/test_preflight_team_recovery.py`。

# 测试执行与验收成本

- 修改阶段只运行受影响的测试；新增用例先单独执行，排除导入、夹具和构建错误。专项与相邻路径复验也用 `scripts/run_release_gate.py <目录> <测试文件…>` 运行，保证资源上限与临时目录和全量一致。
- 「全量通过」只能由 `scripts/run_release_gate.py` 产生。手动拼接或分段的 pytest 不算全量通过；出现环境类失败时改走 gate，不逐个补环境参数。
- 同一版本只完成一轮最终全量。多个分支或 worktree 最终要合并时，分支内只做专项复验，最终全量只在合并后的提交上做一轮。
- 资源占用：sheep-rog 上同一时间最多一个 gate 或其他长测试（预计超过 5 分钟）在跑；长测试可以与正式实验同时运行，应尽快跑，不为等实验暂停而推迟。上一轮长测试确认所有子进程关闭后才能开始下一轮。
- 复用：完整通过的 gate 回执绑定被测提交与配置；仅文档、报告或不影响执行的提交元数据变化时复用已有结果。开跑实验前执行启动脚本 `--check` 并引用实际被测提交的测试回执；修复后旧批次从检查点自动恢复，历史冻结回执保留为历史证据。
- 必需检查只以本文件为准；不在本文件中的检查不作为开跑或验收条件。复盘得到的新检查须先落到回归测试或启动检查，再在此处导航。
- 全套失败后，先汇总并区分实现缺陷、夹具错误和环境故障，集中修复并复验失败项及关联测试，全部通过后再重跑全套。相同失败再次出现时先查原因，保留生产期限、资源限制和安全保护。
- 全量运行期间固定源码、public、测试、依赖和测试配置；补充探索在独立目录进行。确需修改被测版本时，先停止所属测试并确认后代关闭，再修改和专项复验；混合版本的结果不作为验收通过。
- 未明确指定 TMPDIR 时，离线测试使用空间充足的独立 `/dev/shm` 或 tmpfs 目录；涉及真实磁盘持久化、崩溃恢复和磁盘故障的检查使用目标文件系统。日志、JUnit 和验收回执保存在测试临时目录之外，清理只针对本轮所属临时数据。
- 启动长测试前按上一份 gate 回执的耗时设置预算。监测实际进度、错误和资源状态；超过预算或持续无进展时，保存诊断并停止所属测试，确认关闭后再处理。中断轮和已执行前缀单独记录，不计作全套通过。
