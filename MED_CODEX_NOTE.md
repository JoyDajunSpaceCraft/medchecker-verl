# Medical Checker：交给 Codex 的代码修改任务

请在 `medchecker-verl` 仓库启动 Codex，把本文件完整交给它。本文是实施任务书，不是实验结果。

## 目标与工作边界

你是本项目的研究代码工程师。请对照 `revision_inputs/paper.pdf`、`revision_inputs/reviews.md` 和仓库，实际完成可复现性修复、奖励链路测试、CRC 适配及实验入口。不要只给建议；完成能独立推进的修改。缺少历史数据时，继续完成不依赖它的工作，并在 `BLOCKERS.md` 列出所需文件。

保留已有实验和用户未提交的改动。先读 AGENTS.md（如有）、git status、当前分支与 commit。可使用独立工作分支；本任务不包含 push、删除历史 checkpoint、覆盖原始评估结果、付费 API 调用或自行提交完整训练矩阵。完成 CPU 测试；若已在用户分配的 Slurm 作业中，可执行下述有限 GPU smoke。整轮训练先提供具体命令和预算，由用户启动。

参考审计基线：2026-09-05 检查的 commit `2b684a342be840f8d9918e7f755730520b888552`。若本地代码更新，应重新验证发现，不能机械套用。

## P0：先重建“论文—配置—训练标量—原始输出”的对应关系

1. 阅读 `analysis/NAACL_REVISON_AUDIT.md`。它已有很有价值的审计，不要重复建立另一套互相矛盾的计划。但它关于缺少 LaTeX 的判断可能已过期：当前仓库已有 `paper_naacl_revision/arxiv_paper.tex`。
2. 建立 `analysis/revision_evidence.csv`：reviewer/concern、paper section/table、已有结果路径、checkpoint、commit、seed、数据划分、实际 reward manager/score function、证据状态、剩余动作。状态至少区分 VERIFIED、PAPER_ONLY、MISSING、CONTRADICTED。论文表格有数值不等于已有可重算的实验。
3. 新版论文已包含跨模型表、无 checker/loop-only 对照（附录22/Table13）、alpha sweep（附录26/Table14）和额外 benchmark。先寻找这些结果的原始 provenance，不要把 review 里的“补实验”全部照抄重跑。
4. 重点核对论文附录22“同一 backbone 只换 scorer”的表述：主文 likelihood 使用 Meditron，而 classifier 是另一模型，不能据此声称架构固定。附录26的 GPT BERTScore .547→.591→.584 不支持其 caption 中全区间单调下降的描述。输出具体 discrepancy，不预设修订结论。

## P0：修复并证明 checker reward 真的参与优化

请逐段跟踪以下真实文件，而不是只检查工具返回了 reward：

- `verl/tools/checker_tool.py`
- `verl/experimental/agent_loop/tool_agent_loop.py`
- `verl/experimental/agent_loop/agent_loop.py`
- `verl/workers/reward_manager/naive.py` 以及实际配置选用的其他 manager
- `verl/utils/reward_score/med_rag_checker.py`
- `verl/trainer/ppo/core_algos.py`

审计基线中的 `med_rag_checker.py` 返回 `base_score + format_penalty + search_bonus`；`tool_rewards` 主要用于记录；可见 naive manager 没有把该独立字段接入论文乘法公式。先确认实际启动路径和其他 reward 注入点。这说明当前可见路径与论文描述不一致，不能直接推断所有历史训练都无效。

要求：

1. 明确版本化的 reward contract。保留可运行的 `legacy` 模式；将论文公式作为独立、显式选择的模式实现，不能默默改变历史默认值。历史公式若不能重建，标记新结果为 corrected implementation，不能假冒复现旧表。
2. 明确定义 `R = r_base * (1 + alpha * phi_check) + P_fmt` 的适用范围、claim 权重、clip、空 claim、低置信度、缺证据、API/NLI 错误、多次 check 聚合、最后一步 attribution、语言/长度惩罚。`E=1,N=0,C=-1.5` 与 `phi∈[-1,1]` 的关系必须在代码和文档里一致。错误应有显式状态，不能自动变成 neutral/no-gap/满分。
3. 将 checker 输入通过实际 manager 接到训练使用的标量；避免重复计入奖励。tool 返回值、终局 raw reward、KL 调整后 reward/advantage 分开记录。
4. 加端到端有意义的测试：固定 answer、evidence 和 claim，只改变 checker label，训练 scalar 必须按公式改变；alpha=0 时 checker 对 reward 的影响为零但工具反馈仍在；无 checker 与 alpha=0 不是同一个条件。覆盖全 neutral、contradiction、空 claim、截断、格式错误、多次 check、错误重试。
5. 检查 `examples/sglang_multiturn/search_r1_like/run_qwen2.5-7b_search_checker_ablation_2gpu.sh` 的 `actor_rollout_ref.rollout.n=1`。当前非向量化 GRPO 实现对 singleton group 使用 mean=0/std=1，因此**不能写成“优势必定为零”**；但这不等于论文所称的组内相对学习。核实实际 estimator 和 group UID，复制历史配置并另建 group size≥2 的受控新配置；不要悄悄改旧实验定义。
6. 每个 prompt 保存完整 rollout group：qid/group id、seed、模型版本、evidence/claims、verdict/confidence、checker aggregate、base/format/search/checker components、raw R、KL contribution、advantage、response mask。报告组内 reward std、near-zero group fraction、rank changes，以及同组 `A_full−A_alpha0`。仅比较 checker reward 方差不足以说明 GRPO 的有效训练信号：组内共同平移/缩放可能被归一化抵消。

## P0：建立公平实验入口与 CRC 配置

统一脚本 `scripts/revision/train_med.sh`（新建入口），提供 `--mode`、`--seed`、`--max-steps`、`--config`、`--run-dir`、`--dry-run`；参数必须映射到本仓库支持的配置键。调用旧脚本前先审计参数差异。当前 ablation 启动脚本按模式同时改变 batch、response length、entropy、工具配置等，不能当作只改变 checker 的干净对照。

主矩阵固定 Qwen2.5-7B-Instruct、数据 split、retriever/index、claim extractor、token/search/check budget、batch、rollout group size、学习率、KL、entropy、checkpoint 选择与训练预算。区分：

- A：无 checker loop；B：同一 loop、alpha=0；C：同一 loop+likelihood；D：同一 loop+MedNLI；E：同一 loop+GPT 未加额外防护。
- F：GPT+完整防护是另一个系统干预；与 E 的差别要逐项记录，不能用 D/F 直接证明单一 checker backend 因果效应。

先复用可验证的历史 raw runs。需要重跑时，C/D/E/F 各3 seeds是核心候选；A/B用于隔离 feedback 与 reward，需要同等 seed 协议才能作稳健差异结论。不要一次提交18个作业。

CRC 环境：账号 yuj49、Slurm account hdaqing；项目根目录 `/vast/hdaqing/yuj49`。设置可配置的 MODEL_ROOT/DATA_ROOT/RESULTS_ROOT/HF_HOME，清理生效配置中的 `/ocean/projects/...`、`/jet/home/...`。保留历史记录里的原始路径，不全仓字符串替换。环境单独建立，不能升级另一个项目共用的 env。

Slurm 分配2张 A100 NVLink 80GB作为初始训练 smoke。保留 `CUDA_VISIBLE_DEVICES`，不写死 `0,3`；trainer world size与实际GPU数一致。清点 policy、reference、critic（若实际使用）、rollout、retriever、local checker 的显存和CPU服务占用；GPU服务也计入申请，不能占其他用户的卡。API checker的等待不是GPU不足。

服务 endpoint 可配置；启动健康检查有超时；记录模型和index版本；只清理本作业启动的子进程。离线缓存准备与GPU训练分开；禁止在每个 seed 内重新下载语料/模型。记录 CUDA/PyTorch/attention/SGLang/NCCL 环境；不要直接升级本仓库的 verl fork 来适配新 GPU。

## 最小实验计划（按 gate 执行）

| 顺序 | 假设/任务 | 设计与输出 | 起步资源与停止条件 |
|---|---|---|---|
| M0 | 实际 reward 与指定公式一致 | mock + 最小 manager→advantage 测试；可重算 JSONL | CPU；契约失败时不训练 |
| M1 | scorer 机制而非抽取差异造成 collapse | 冻结相同 claim/evidence bank；同一 causal LM 比较原 likelihood 与正确标签 log-prob；另比 classifier/GPT；context长度单独因素 | 小量 CPU/1GPU 或API；先≤100对诊断，冻结后再扩大；不混作独立重复 |
| M2 | checker 改变组内有效学习信号 | 每模式10–20更新步、少量完整组；检查 reward/advantage、mask、内存和吞吐 | 2×A10080，30–60分钟 gate；无有效信号/NaN/错误率异常则停 |
| M3 | 效果不依赖单一 seed | 受控矩阵3 seeds，先一次只跑一个；预定等更新/token预算并保存中途失败 | 每run先申请4–8h，按M2估计修正；不能声称论文2H100耗时等于CRC耗时 |
| M4 | 支持率改进对独立评估有效 | 冻结外部 evaluator；盲评分层样本（含无搜索/失败/不同语言）；answer correctness、support、contradiction、coverage分别算 | 离线分析优先；人评不由本checker冒充；先约100答案预算试标 |
| M5 | 泛化到另一模型 | 复用已有跨模型数据；不足才在一个第二backbone上补最关键两条件 | M3可解释后再分配；不直接扩到30–70B |

统计以独立 training seed 为重复；同一测试题配对比较，区分 question bootstrap CI与训练seed方差。报告mean/std和effect size，评估题数不能充当训练重复数。预先固定主指标、容忍区间和缺失结果处理；p不显著不等于等效。正文可将结论收窄为特定 policy/checker/config 的现象，不强行维护“中等checker普遍更好”。

## 完成标准

提交代码变更、必要测试、CRC启动脚本、单一配置表、artifact manifest、`analysis/revision_evidence.csv`、`analysis/revision_results.md` 和 `BLOCKERS.md`。结果文档只引用真实日志；未运行明确标注 NOT_RUN。每个论文表格应能追到输入JSONL与汇总命令。修复错误数据集引用与自评/外评命名；缺少canonical论文源时提供patch建议，不凭空修改PDF。

立即执行 M0及CRC适配，完成可做的测试与代码；随后返回：改了什么、测试证据、尚缺什么、第一条GPU smoke命令及预算。不要以“我可以帮你”结束。
