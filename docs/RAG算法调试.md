# RAG 算法追踪日志

开发阶段的检索、重排和证据判定统一生成结构化 Trace：一份写入 `rag.trace` JSON 日志，一份经异步队列写入后台调用链。两者使用同一个 `trace_id` 串起一次问答，方便定位故障、制作评测集、比较算法版本和统计误召回原因。

## 环境开关

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_TRACE_ENABLED` | `true` | 是否输出结构化算法追踪事件 |
| `RAG_TRACE_CONTENT_ENABLED` | 开发环境 `true`，生产环境 `false` | 是否记录完整问题、回答和候选片段 |
| `RAG_TRACE_CANDIDATE_DETAILS_ENABLED` | 开发环境 `true`，生产环境 `false` | 是否逐条输出召回和重排候选事件 |
| `RAG_TRACE_CONTENT_MAX_CHARS` | `50000` | 单个正文字段最大记录字符数 |
| `RAG_TRACE_PERSISTENCE_ENABLED` | `true` | 是否把结构化事件异步写入后台调用链数据库 |
| `RAG_TRACE_RETENTION_DAYS` | `30` | 调用链保留天数，过期摘要和事件级联删除 |
| `RAG_TRACE_QUEUE_SIZE` | `500` | 单进程异步写库队列上限，队列满时丢弃追踪事件但不阻塞问答 |
| `RAG_TRACE_MAX_EVENT_BYTES` | `131072` | 单个入库事件最大约 128 KiB；超限时保留标识和可容纳指标并标记截断 |
| `RAG_TRACE_MAX_EVENTS_PER_RUN` | `500` | 单个 Trace 最多持久化的阶段事件数，摘要仍会继续更新 |
| `APP_ENV` | 宿主机 `development`，Docker `production` | 决定正文追踪的安全默认值 |
| `APP_VERSION` / `APP_REVISION` | `dev` / 空 | 发布版本和 Git revision，由镜像构建自动注入 |
| `LOG_LEVEL` | `INFO` | Python 日志级别 |

生产环境即使关闭正文追踪，也会保留字符数和 SHA-256，便于关联重复问题而不保存原文。生产环境也默认关闭逐候选事件，但保留阶段汇总与最终选择结果。需要在线调试正文时应显式设置 `RAG_TRACE_CONTENT_ENABLED=true` 和 `RAG_TRACE_CANDIDATE_DETAILS_ENABLED=true`，完成后及时关闭。
关闭正文追踪时，用户名、问题/回答原文、文件名、查询命中片段、重排理由和上游异常正文不写入结构化 Trace；仅保留 ID、类型、字符数和摘要。

每个事件都携带 UTC `timestamp`、`trace_schema_version`、`app_version`、`app_revision` 和 `trace_id`。当前 Trace schema 为 `v2`：它将右侧展示候选、实际生成依据和直接命中数拆开；历史 `v1` 事件及 `export_schema_version=1` 的导出文件仍可读取，但评测和统计必须先按 `trace_schema_version` 分组，不能把两套 `hit_count` 语义直接混算。

## 主要事件

| 事件 | 用途 |
| --- | --- |
| `chat.request` | 用户、会话、知识库、检索参数和原问题；始终是正常问答调用链的起点 |
| `conversation.context_resolved` | 多轮追问判断、独立检索问题、历史消息数和上一轮证据复用数量 |
| `conversation.reference_unresolved` | 追问缺少可消解对象，直接要求用户补充信息且不盲目检索 |
| `intent.model_result` / `intent.model_error` | 意图模型原始分类是否被接受、精确拒绝原因、模型与 Prompt 版本；空响应同时记录 `finish_reason`、choice 数、推理内容长度和 token 用量，但不记录模型推理正文 |
| `intent.routing_decision` | 规则、模型和安全策略合并后的最终响应模式与检索策略 |
| `chat.pipeline_selected` | 本轮唯一执行器：显式回滚 `v1`、知识检索 `v2` 或无检索 `direct`；V2 合同异常不会静默回退 V1 |
| `chat.turn_reclaimed` | 同一 `request_id` 的 `accepted/generating` 执行租约已过期，由当前请求安全接管并重新执行 |
| `direct.plan` | 已验证的通用交流、平台帮助或内联写作直答计划；该链路明确不执行检索 |
| `query.plan` | V2 本地查询形状、检索子问题、answer/bridge requirement 与是否需要澄清 |
| `retrieval.plan` | 是否检索、检索策略、候选数、Top K 和查询硬约束 |
| `retrieval.candidate` | 每个召回片段的向量、关键词、三元组、RRF 排名和分数 |
| `retrieval.completed` | 召回是否成功、候选数、激活通道与耗时 |
| `retrieval.channel_error` / `retrieval.error` | 单通道或整体召回失败，以及是否由其它通道或上一轮合法证据恢复 |
| `rerank.candidate` | 主题相关度、可回答性、产品/版本约束、证据角色和模型理由 |
| `rerank.completed` | 重排是否尝试、成功、降级原因、候选数和耗时 |
| `evidence.ambiguity_assessed` / `evidence.clarification_required` | 根据通过准入的真实相关文档判断是否存在互斥产品、版本或项目范围；要求澄清时禁止生成 |
| `evidence.clarification_created` / `evidence.clarification_repeated` / `evidence.clarification_resolved` | 证据范围选项的保存、无效短回复重复和最终选择；pending state 只保存候选，不授予执行权限 |
| `evidence.scope_filter_applied` | 用户选择后应用的知识库、文档和选项数量；`global_fallback_allowed=false` 表示禁止退回全库检索 |
| `evidence.selection` | 直接证据、相近资料、淘汰数量及最终证据状态 |
| `generation.context` | 实际注入模型的文档、角色和上下文字符数 |
| `generation.skipped` | 因证据范围待澄清而跳过回答模型 |
| `generation.completed` | 模型生成完成、回答长度、token 与耗时 |
| `chat.response` | 最终回答、来源摘要、token 和证据状态 |
| `chat.cancelled` | 用户停止、浏览器断连或服务关闭导致流取消，调用链立即标记为已中断 |
| `chat.error` | 失败阶段、异常类型和当时的证据状态 |
| `chat.persistence_error` | 回答已生成但会话持久化失败 |
| `search_test.request` / `search_test.completed` / `search_test.error` | 后台检索测试的输入、结果和失败事件 |

字段语义：

- `retrieval_score`：召回层排名融合分，不是概率。
- `topic_relevance`：重排模型判断的主题相关程度。
- `answer_support`：候选是否足以支撑回答的评分。
- `constraint_status`：`exact`、`compatible`、`unknown`、`mismatch` 或 `neutral`。
- `evidence_role`：`direct` 是回答依据，`related` 是相近资料，`irrelevant` 不进入上下文。
- `results` / `displayed_result_count`：右侧检索面板展示的候选及其数量，可能包含直接依据和相近资料，不等于生成依据。
- `answer_sources` / `answer_source_count`：实际进入本轮生成上下文、随后可随历史回答保存的片段及其数量；它与 `results` 明确分离。
- `context_evidence_count`：实际注入生成模型的片段数，正常情况下等于 `answer_source_count`；单独保留该指标便于检查生成上下文是否与持久化来源一致。
- `hit_count` / `direct_evidence_count`：通过直接证据门控的数量。Trace schema `v2` 中两者语义一致；历史 `v1.hit_count` 曾表示前端候选数，只能按旧口径解释。

## 当前 V2 检索与证据链路

1. 后端先编译并校验 `rag_task_contract.v1`。知识问答和“依据知识库写作”进入 V2；问候、平台帮助和已附原文写作进入独立 `direct` runner。只有部署显式配置 `RAG_PIPELINE_VERSION=v1` 才使用旧主链，V2 合同缺失、漂移或越权一律拒绝执行。
2. V2 本地规划器只按问题结构拆分 `fact / process / list / comparison / multi_part / multi_hop`。诸如“普通员工的餐补”会生成最终 answer requirement 和身份到等级的 bridge requirement，不在代码中猜测 D 级、金额或其它业务值。
3. 向量、PostgreSQL FTS 和 `pg_trgm` 并行召回，使用 RRF 做稳定排序。文档准入只接受真实词面得分或达到绝对门槛且接近本轮最佳值的原始向量分；RRF 名次本身不能冒充相关性。
4. 小文档全文和结构邻居扩展只发生在已授权、已由首轮候选锚定的文档内，并受文档数、片段数、字符数和共享期限限制。扩展超时只标记降级并保留首轮证据；主检索失败与正常零命中严格区分。
5. 产品、版本、项目和用户选择的文档范围在代码层硬过滤。互斥且都相关的版本/产品必须先产生结构化澄清；选择后仅查询服务端保存并重新授权的 KB/doc allow-list，禁止退回全库。
6. 每个进入上下文的片段必须有正向 evidence role 和 `supports_requirement_ids`。多跳答案必须同时证明 bridge，并用同一个中间值连接最终标准；“普通员工→D级”不能与“A级标准”拼成完整答案。缺 bridge 或子问题覆盖不全时保持 partial/insufficient，不能伪报 complete。
7. V2 不调用旧的生成式 reranker；`rerank.completed` 会明确记录 `attempted=false`。最终回答模型最多调用一次，并且只看到预算内的已授权 evidence bundle。`error / no_hit / version_mismatch` 等无上下文终态由本地固定文案直接返回，避免再花时间让模型改写失败信息。

旧 V1 仍保留作为显式紧急回滚路径，其 `rerank.candidate`、`topic_relevance` 和 `answer_support` 字段仅用于 V1 Trace；不能与 V2 的确定性证据口径混算。

## 回答交付与恢复

1. 浏览器为每个逻辑请求生成 `request_id`；服务端同时用 `user_id + request_id` 和 `conversation_id + request_id` 保证幂等。新会话即使尚无 `conversation_id`，网络重试也会定位到原会话。
2. `ChatTurn` 按 `accepted → generating → generated → completed` 推进。生成正文先写入恢复账本，再写 assistant 消息；保存失败停在 `generated/persist_failed`，相同 request 重试只补保存，不重新检索或调用模型。`accepted/generating` 使用有界执行租约，进程异常退出后可由同一请求接管，正常运行中的任务不会被重复执行。
3. 路由失败、流异常和用户取消分别落入明确终态；保存最多有限重试三次。回答提交与 pending 澄清状态使用分离事务，后者 CAS 冲突不会回滚已经保存的回答，也不会发送虚假的澄清 ACK。
4. 回答历史保存最终 `search_snapshot`，候选最多 20 条且不保存正文。重新打开历史或重放幂等请求时，服务端按当前 RBAC、文档 active/ready 状态重新加载正文；撤权后保留用户自己的回答文本和计数，但不再披露旧来源。
5. 前端绑定响应头中的 Trace/Turn/Request ID。`error` 后即使收到尾随 `done` 也保持失败状态；保存/传输不确定时复用原 request 恢复，已完成回答的“重新生成”会创建新的逻辑 request。

## 路由判定矩阵

知识库是否已勾选只是意图分类的弱先验，不能让每个问题都自动检索。最终执行遵循以下边界：

| 输入类型 | 预期执行 |
| --- | --- |
| 企业制度、流程、知识库文档、外部产品配置/部署/接口问题 | 必须检索；即使模型误判成通用交流，策略层也会纠正 |
| 问候、一般常识、与企业资料无关的概念解释 | 直接由对话模型回答；已选择知识库也不触发检索 |
| 当前 RAG 平台自身如何上传文档、创建知识库、查看检索结果 | 直接使用平台帮助模式，不检索业务知识库 |
| 当前输入已经附带原文的润色、翻译、改写 | 直接使用写作模式 |
| 明确要求根据知识库、手册、制度或文档完成写作 | 先检索再写作 |
| 主意图模型空响应或因长度截断 | 若对话模型与意图模型不同，使用对话模型做一次二级分类；分类结果仍经过确定性策略保护 |
| 意图模型超时、401/500、低置信度，或两个分类模型都失败 | 除明确问候、平台帮助和已附原文写作外，保守要求检索 |

意图模型的短 JSON 输出预算为 512 tokens，兼容先产生 reasoning tokens 的模型。若仍返回空正文或 `finish_reason=length`，且模型管理中的意图模型与对话模型不同，系统只追加一次对话模型分类；401、500 和超时不会重复调用同一模型。Trace 通过 `attempt=primary/fallback_chat_model` 记录两次结果、拒绝原因和各自耗时。

策略层的企业产品词典集中维护在 `query_constraints.py`，当前覆盖云枢 / CloudPivot、钉钉 / DingTalk、企业微信 / 企微 / WeCom、泛微 OA / Weaver OA。只有“已知企业产品 + 配置、部署、接入、同步、接口等操作”才会纠正错误的 `general_chat`；Python、Redis、PostgreSQL 等通用技术问题仍由意图分类决定，不会仅因出现“配置”就被强制查库。新增企业产品时应补充别名组和表驱动路由回归。

## 多轮追问与证据复用

1. 每轮最多读取最近 6 条消息，并把历史正文限制在 6000 字符内，避免会话无限增长挤占模型上下文。
2. 只有“这些配置”“上述内容”“那 8.6 呢”或“云枢中如何配置”这类动作完整但宾语省略的问题才继承上一轮；“换个问题”等显式新话题，以及“云枢中如何配置默认密码”这类已经包含对象的完整问题不会继承。
3. 当前问题中明确出现的产品和版本优先于历史约束。系统生成独立检索问题，并统一交给意图路由、约束提取、召回和当前执行器的证据判定。
4. 上一轮来源不能直接作为本轮事实：系统会按当前用户选择的知识库范围，重新检查文档是否启用且处理完成，再加载真实片段。
5. 上一轮片段与本轮新召回按 chunk 去重，并针对本轮问题重新检索和判定；旧的召回分和回答支持分不会沿用。
6. 新会话直接询问“这些配置有什么影响”时，系统返回确定性澄清提示，不调用意图模型、检索器或生成模型。
7. 若系统刚要求补充“配置什么”，用户下一轮只回答“登录用户名枚举”等短对象，系统会填回上一轮缺失槽位并生成“云枢中如何配置登录用户名枚举”，不会再次丢失上下文；完整的新问题不会被该规则吞并。
8. 历史 assistant 回答只帮助模型理解指代，不是事实证据。只有本轮通过证据门控的知识片段可以支撑确定回答；`answer_support < 0.1` 的资料会直接淘汰，达到相近资料最低门槛但未达到直接证据门槛的片段也不能冒充确定回答依据。
9. 每条 assistant 历史回答只保存本轮实际进入生成上下文的 `answer_sources`；更宽的 `results` 候选仅供当轮右侧检索面板和 Trace 诊断，不冒充历史回答依据。重新打开会话时仍会按当前用户知识库范围、文档启用状态和处理状态过滤保存的来源；普通问答用户可展开当轮依据，点击全文预览仍必须具备 `doc:read` 权限。历史版本若在 `no_hit`、`skipped` 或 `error` 状态下误存了宽候选，读取时会按无引用处理。

当前 `pg_trgm` 是中文词面召回的安全网，在大知识库上可能产生较高扫描成本。数据量增长后应用真实数据运行 `EXPLAIN (ANALYZE, BUFFERS)`，再决定是否增加持久化词面搜索列或独立倒排索引。

## 后台调用链

具备 `log:read` 的角色可进入 `/admin/rag-traces`。列表可按 Trace ID、会话 ID、时间、请求类型和最终状态筛选；详情按真实执行顺序显示多轮解析、意图、检索、重排、证据筛选和生成事件。

- `rag_trace_runs` 保存短摘要，列表查询不会扫描大 JSON。
- `rag_trace_events` 保存逐阶段 JSONB，并通过 `trace_id + sequence` 保证同一进程内的执行顺序。
- 事件经有界内存队列批量写库，不等待 PostgreSQL 才向用户发送 SSE；队列为成功、失败和中断终止事件预留容量，队列满时优先驱逐最旧的非终止明细。队列或数据库不可用只影响追踪，不影响问答。
- `success` / `error` 只表示整次请求的最终结果。可恢复的 `retrieval.error` 会在时间线中标红，但最终已成功返回答案时不会把整条调用链误记为失败。
- 主动停止或连接取消会立即标记为 `interrupted`；异常退出且没有终止事件的 `running` 调用在 15 分钟后兜底标记为 `interrupted`。调用链默认保留 30 天，后台详情每次按执行顺序加载 50 个事件，避免候选较多时一次返回过大 JSON。
- 生产环境默认不保存问题、回答、文件名和候选正文；后台只显示哈希、字符数、指标和对象 ID。正文开关同时约束容器日志与数据库事件。
- 详情中的“下载 AI 分析文件”会导出 `rag-trace-{trace_id}.json`，包含已经入库的摘要、事件时间线、阶段索引、固定诊断快照、版本、完整性检查和 AI 分析说明。接口要求 `log:read`；若调用链含开发环境业务正文，详情和导出还必须是超级管理员。不回查聊天消息、文档正文、模型设置或密钥，也不会绕过正文关闭策略；每次下载会写入 `rag_trace.export` 操作审计，但审计详情不记录正文。单次文件最多选取 500 个事件，最终编码文件硬限制为 24 MiB，优先保留请求、上下文、路由、阶段汇总和最新终止状态，再采样候选明细；若被截断，响应头和 `diagnostic_index.integrity` 都会记录省略数量。
- 下载前后会再次递归清理 API Key、Token、Cookie、密码以及 HTTP、PostgreSQL、Redis、AMQP 等连接串中的 userinfo、query 和 fragment。脱敏不能替代业务数据审批：开发 Trace 仍可能含问题、回答和文档正文，上传外部 AI 前必须检查。

## 查看与导出

查看某次请求：

```bash
docker compose logs app | rg 'rag.trace' | rg '你的-trace-id'
```

后台下载的 JSON 可以直接交给 AI 分析。`diagnostic_index.snapshot` 汇总了上下文、主/备用分类、执行器选择、V2 查询规划或 direct 计划、召回、证据和生成结论；V1 调用链还会保留重排模型与阈值，V2 则明确记录未执行模型重排。快照同时保留 RRF、三元组门槛、候选池、Prompt 版本、生成参数和 system prompt 指纹，便于跨版本复现；`ai_analysis_guide` 提供按 `v1 / v2 / direct` 分支分析的提示及不可信数据边界；`diagnostic_index.recommended_checks` 给出检查顺序。`diagnostic_index.integrity` 只检查已入库行与导出选择：队列在写库前丢弃的事件尚未分配 sequence，因此连续编号不能证明原始链路零丢失。`data_policy.content_included=false` 表示生产环境未保存正文，AI 只能基于哈希、指标和对象 ID 判断链路，不能从导出接口恢复原文。

导出 JSONL（日志前缀在第一个 `{` 前被移除）：

```bash
docker compose logs app --no-log-prefix \
  | rg 'rag.trace:' \
  | sed 's/^[^{]*//' \
  > rag-trace.jsonl
```

统计版本不匹配候选：

```bash
jq -c 'select(.event == "rerank.candidate" and .constraint_status == "mismatch")' rag-trace.jsonl
```

追踪日志用于算法评估，不替代审计日志。导出文件可能包含企业问题和回答，必须按业务数据保护要求保存和清理。

Compose 中容器日志默认仅轮转保留 `3 × 10MB`，适合短期排查，不适合当作长期评测数据库。需要跨版本分析时，应定期导出 JSONL，或接入 Loki / ELK 等集中日志系统并设置存储周期。
