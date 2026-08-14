# RLinf × LIBERO-OOD 项目交接文档

最后更新：2026-08-13

## 1. 项目目标

本项目的目标是在 RLinf 中复用原有 Ray/Hydra embodied pipeline，支持使用
OpenVLA-OFT 和 StarVLA Qwen2.5-VL-OFT checkpoint 在 `pi0-text-latent` 提出的
LIBERO-OOD benchmark 上评估，并进一步支持在 OOD suite 上进行 GRPO 训练。

核心 benchmark 为：

- `libero_goal_ood`：10 个任务；
- `libero_spatial_ood`：10 个任务；
- `libero_object_ood`：10 个位置过拟合诊断任务，不属于论文核心 20-task 汇总。

重要研究边界：如果直接在 Goal-OOD 或 Spatial-OOD 上训练，这些任务就不再是零样本
OOD 测试集，后续分数衡量的是目标套件上的在线适应/强化学习能力。

## 2. 本机路径与运行环境

```text
RLinf 仓库：
/home/sen/data/new_python_project/RLinf

modified LIBERO：
/home/sen/data/new_python_project/RLinf/third_party/modified_libero

StarVLA 源码：
/home/sen/data/new_python_project/CRL_Project/starVLA

StarVLA Qwen2.5-VL-OFT checkpoint：
/home/sen/data/new_python_project/CRL_Project/starVLA/playground/Pretrained_models/Qwen2.5-VL-OFT-LIBERO-4in1

StarVLA 虚拟环境：
/home/sen/data/new_python_project/RLinf/.venv-starvla

OpenVLA-OFT 虚拟环境：
/home/sen/data/new_python_project/RLinf/.venv-openvla-oft
```

目标 OpenVLA-OFT 权重曾使用：

```text
RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
```

modified LIBERO 依赖应锁定 `robosuite==1.4.1`，不要直接升级到最新版。

## 3. 当前完成状态

### 3.1 评估链路

已经独立接入 RLinf 原有 Ray/Hydra 评估框架：

```text
evaluations/run_libero_ood_eval.sh
  → evaluations/eval_libero_ood_agent.py
  → LiberoOODEvalRunner
  → LiberoOODEvalEnvWorker
  → LiberoOODEnv
  → third_party/modified_libero
```

相关文件：

- `evaluations/run_libero_ood_eval.sh`
- `evaluations/eval_libero_ood_agent.py`
- `rlinf/runners/libero_ood_eval_runner.py`
- `rlinf/workers/env/libero_ood_eval_worker.py`
- `rlinf/envs/libero/libero_ood_env.py`
- `rlinf/envs/__init__.py`
- `evaluations/libero_ood/*.yaml`

### 3.2 已验证的 StarVLA Goal-OOD 结果

一次完整 Goal-OOD 评估已经成功跑通：

```text
任务数：10
每任务轨迹数：10
总轨迹数：100
整体 success_once：0.31
```

当次每任务结果：

| task_id | success_once | trajectories |
| --- | ---: | ---: |
| 0 | 0.30 | 10 |
| 1 | 0.00 | 10 |
| 2 | 0.00 | 10 |
| 3 | 0.00 | 10 |
| 4 | 0.80 | 10 |
| 5 | 0.10 | 10 |
| 6 | 0.00 | 10 |
| 7 | 0.90 | 10 |
| 8 | 1.00 | 10 |
| 9 | 0.00 | 10 |

这证明 Goal-OOD 的模型加载、环境注册、100 条 rollout、整体指标和逐任务指标链路均已
跑通。Spatial-OOD 和 Object-OOD 已有同构配置，但仍建议分别做一次完整实跑确认。

## 4. LIBERO-OOD benchmark 接入方式

modified LIBERO 没有改变 Panda 机器人、7 维动作接口、相机观测或主要仿真环境。
主要变化为：

1. 新增 Goal/Spatial/Object OOD BDDL 文件；
2. 注册三个 OOD benchmark；
3. 将 `On(object, object)` 的 XY 中心距离阈值从 0.03 放宽到 0.1；
4. 新增炉灶 `base_region`，OOD stove 任务允许放到更大的区域；
5. OOD rollout 使用 BDDL placement sampler 随机 reset，不读取固定 init-state 文件。

`third_party/modified_libero/libero/__init__.py` 是必要的兼容文件，用于确保 Python 优先
导入仓库内的 modified LIBERO，而不是虚拟环境里安装的标准 LIBERO。不要删除。

运行脚本会使用独立路径：

```text
LIBERO_CONFIG_PATH=<RLinf>/logs/libero_ood_config
```

这样不会改写用户主目录下的标准 LIBERO 配置。

## 5. 评估协议

### 5.1 随机初始化

OOD suite 没有：

```text
init_files/libero_goal_ood
init_files/libero_spatial_ood
init_files/libero_object_ood
```

因此不能调用：

```python
task_suite.get_task_init_states(task_id)
env.set_init_state(...)
```

`LiberoOODEnv` 使用：

```python
env.reset()
```

每个 `(task_id, trial_id)` 是逻辑 ID，不对应磁盘上的固定物理状态。评估环境使用基准
seed 7，连续 reset 推进 BDDL placement sampler 的随机数流。

### 5.2 精确 300 步

StarVLA 每次输出 8 个动作。专用 OOD 评估 worker 执行：

```text
前 37 个 chunk：37 × 8 = 296 步
最后 1 个 chunk：仅执行前 4 步
总计：300 步
```

`LiberoOODEvalEnvWorker` 只用于独立 OOD evaluator，避免影响 RLinf 其他 benchmark。

Object-OOD 当前配置为 280 步，可以被 8 整除，不需要截断。

### 5.3 成功指标

环境记录：

- `success_once`：轨迹中是否曾成功；
- `task_id`：轨迹对应的任务；
- `trial_id`：逻辑 trial；
- `task_XX/success_once`：每个任务的成功率；
- `task_XX/num_trajectories`：每个任务的有效轨迹数。

评估输出既有 suite 整体成功率，也有逐任务成功率。

## 6. StarVLA OOD 评估命令

```bash
cd /home/sen/data/new_python_project/RLinf
source .venv-starvla/bin/activate

export MODEL_PATH=/home/sen/data/new_python_project/CRL_Project/starVLA/playground/Pretrained_models/Qwen2.5-VL-OFT-LIBERO-4in1
```

Goal-OOD：

```bash
bash evaluations/run_libero_ood_eval.sh \
  libero_goal_ood_starvla_qwen25_oft_eval
```

Spatial-OOD：

```bash
bash evaluations/run_libero_ood_eval.sh \
  libero_spatial_ood_starvla_qwen25_oft_eval
```

Object-OOD：

```bash
bash evaluations/run_libero_ood_eval.sh \
  libero_object_ood_starvla_qwen25_oft_eval
```

如果 modified LIBERO 移动到其他位置：

```bash
export LIBERO_OOD_ROOT=/path/to/modified_libero
```

该路径必须包含：

```text
libero/libero/bddl_files/libero_goal_ood
```

## 7. OOD 训练实现

### 7.1 没有新增训练入口

仍使用 RLinf 原有入口：

```text
examples/embodiment/run_embodiment.sh
  → examples/embodiment/train_embodied_agent.py
```

`run_embodiment.sh` 会根据配置名前缀识别 Goal/Spatial/Object-OOD，自动：

1. 将 modified LIBERO 放到 `PYTHONPATH` 最前面；
2. 设置独立 `LIBERO_CONFIG_PATH`；
3. 强制 `LIBERO_TYPE=standard`；
4. 设置 MuJoCo、Numba、Matplotlib 相关路径；
5. 再启动原有 Actor/Rollout/Env Worker。

### 7.2 训练配置

已新增：

- `examples/embodiment/config/libero_goal_ood_grpo_starvla_qwen25.yaml`
- `examples/embodiment/config/libero_spatial_ood_grpo_starvla_qwen25.yaml`

两个 suite 当前分别训练，没有在同一 run 内混合 20 个任务。若后续必须混合训练，需要新增
组合 benchmark 或实现跨 suite task scheduler，目前尚未实现。

### 7.3 GRPO group

默认训练配置：

```yaml
algorithm:
  group_size: 10

env:
  train:
    total_num_envs: 10
    group_size: 10
    rollout_epoch: 4
```

同一个逻辑 `(task_id, trial_id)` 被复制给一个 10-trajectory GRPO group。组内环境使用相同
任务、BDDL 和构造 seed，从尽可能一致的初始布局出发，再通过策略随机性生成不同轨迹。

训练构造 seed 为：

```python
base_seed + task_id * num_trials_per_task + trial_id
```

如果同一环境进程没有切换任务，则不会在每次 reset 时重新 seed，而是继续推进当前 BDDL
sampler 的 RNG，从而获得后续随机布局。

### 7.4 训练 horizon

StarVLA 动作形状为：

```text
[batch, 8, 7]
```

普通 RLinf 训练 worker 要求 horizon 能被动作 chunk 长度整除，因此训练及训练内验证使用：

```text
304 = 38 × 8
```

每次模型 forward 后执行完整 8 个动作，再根据最新观测重新规划。正式 benchmark 结果必须
使用独立 OOD evaluator 的精确 300 步协议，不能直接把训练内 304 步 validation 当成论文
口径结果。

### 7.5 Actor 与 Rollout

RLinf 创建两个 StarVLA 副本：

- Actor：FSDP 训练、计算梯度；
- Rollout：Hugging Face backend 推理、生成动作。

`runner.weight_sync_interval: 1`，每次更新后将 Actor 权重同步给 Rollout。

为适配单卡 RTX 4090 D，配置使用：

```yaml
actor:
  micro_batch_size: 1
  enable_offload: true

rollout:
  enable_offload: true
```

这只能降低显存压力，不能保证 3B VLM、优化器和 rollout 模型一定能在 24 GB 显存上稳定
训练；若进程被 `Killed`，应同时检查 GPU OOM 和系统内存 OOM。

## 8. StarVLA OOD 训练命令

```bash
cd /home/sen/data/new_python_project/RLinf
source .venv-starvla/bin/activate

export STARVLA_MODEL_PATH=/home/sen/data/new_python_project/CRL_Project/starVLA/playground/Pretrained_models/Qwen2.5-VL-OFT-LIBERO-4in1
```

Goal-OOD：

```bash
bash examples/embodiment/run_embodiment.sh \
  libero_goal_ood_grpo_starvla_qwen25
```

Spatial-OOD：

```bash
bash examples/embodiment/run_embodiment.sh \
  libero_spatial_ood_grpo_starvla_qwen25
```

单步连通性测试：

```bash
STEPS=1 SAVE_INTER=1 \
bash examples/embodiment/run_embodiment.sh \
  libero_goal_ood_grpo_starvla_qwen25
```

即使只运行一个 RL step，仍需要完整采集一轮环境轨迹，耗时不会很短。

## 9. 当前工作区状态

当前工作区是 dirty worktree。接手者必须保留这些修改，不要执行：

```bash
git reset --hard
git checkout -- .
git clean -fd
```

截至本文生成时，`git status --short` 包含：

```text
 M docs/source-en/rst_source/examples/embodied/starvla.rst
 M docs/source-zh/rst_source/examples/embodied/starvla.rst
 M examples/embodiment/config/env/libero_goal_ood.yaml
 M examples/embodiment/config/env/libero_spatial_ood.yaml
 M examples/embodiment/run_embodiment.sh
 M rlinf/envs/libero/libero_env.py
 M rlinf/envs/libero/libero_ood_env.py
 M rlinf/utils/metric_utils.py
 M tests/unit_tests/test_libero_ood_eval.py
?? evaluations/libero/libero_spatial_starvla_qwen25_eval.yaml
?? evaluations/libero_ood/libero_goal_ood_starvla_qwen25_oft_eval.yaml
?? evaluations/libero_ood/libero_object_ood_starvla_qwen25_oft_eval.yaml
?? evaluations/libero_ood/libero_spatial_ood_starvla_qwen25_oft_eval.yaml
?? examples/embodiment/config/libero_goal_ood_grpo_starvla_qwen25.yaml
?? examples/embodiment/config/libero_spatial_grpo_starvla_qwen25.yaml
?? examples/embodiment/config/libero_spatial_ood_grpo_starvla_qwen25.yaml
?? tests/unit_tests/test_metric_utils.py
?? third_party/modified_libero/libero/__init__.py
?? LIBERO_OOD_HANDOFF.md
```

注意：Git 未跟踪文件同样包含必要实现或用户配置，不能因为是 `??` 就删除。

## 10. 已完成的验证

以下检查已经通过：

1. StarVLA Goal-OOD 完整 100-trajectory 实跑；
2. modified LIBERO 实际导入路径确认：
   `/home/sen/data/new_python_project/RLinf/third_party/modified_libero/libero/__init__.py`；
3. `libero_goal_ood` 注册为 10 个任务；
4. `libero_spatial_ood` 注册为 10 个任务；
5. Goal/Spatial 训练 YAML 的 Hydra compose 与 resolve；
6. `bash -n examples/embodiment/run_embodiment.sh`；
7. Ruff check/format；
8. `git diff --check`；
9. 相关单元测试 11 项通过。

测试命令：

```bash
PYTHONPATH=/home/sen/data/new_python_project/RLinf \
LIBERO_CONFIG_PATH=/tmp/rlinf-libero-ood-test-config \
MPLCONFIGDIR=/tmp/rlinf-mpl-config \
.venv-starvla/bin/pytest -q \
  tests/unit_tests/test_libero_ood_eval.py \
  tests/unit_tests/test_metric_utils.py
```

最近结果：

```text
11 passed
```

Sphinx 全量文档构建没有完成，原因是当前调用到的 Sphinx 环境缺少 `myst_parser`；这不是
RST 内容错误。中英文 StarVLA 页面已做命令、配置名和内容一致性检查。

## 11. 已知限制与下一步建议

1. Spatial-OOD 和 Object-OOD 尚应各进行一次完整 StarVLA 实跑确认；
2. OOD 训练尚未在当前单卡上完成至少一个反向更新实跑；
3. 4090 D 上可能出现 GPU 或主机内存 OOM；
4. 当前 Goal/Spatial 是两个独立训练 run，没有 20-task 联合 scheduler；
5. 训练内 validation 是 304 步、每次少量轨迹，不是正式 benchmark；
6. 最终训练 checkpoint 可能需要导出/转换为 StarVLA evaluator 可直接加载的推理格式；
7. 不要将 OOD 训练后的分数描述成零样本 OOD 泛化；
8. 提交代码前应重新检查所有 untracked 文件并决定哪些属于本次正式改动。

推荐接手顺序：

```text
阅读本文件
→ 查看 git status 和 git diff
→ 跑 Goal-OOD 单步训练连通性测试
→ 处理显存/内存问题
→ 确认 checkpoint 保存和推理导出格式
→ 分别完整验证 Spatial-OOD/Object-OOD
→ 整理提交范围和文档
```

## 12. 新 Codex 账号接手提示词

切换账号后，在仓库根目录开启新会话并发送：

```text
请先完整阅读 /home/sen/data/new_python_project/RLinf/LIBERO_OOD_HANDOFF.md，
然后检查当前 git status 和 git diff。工作区包含用户未提交的重要修改和 untracked 文件，
不要 reset、checkout 或 clean。请基于交接文档继续 RLinf × LIBERO-OOD 工作，并在采取
修改前先确认现有评估、训练配置、modified LIBERO 路径以及 300/304 步协议。
```

