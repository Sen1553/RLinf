在 LIBERO 上进行顺序 LoRA 微调
=================================

本示例在 RLinf 原有 embodied pipeline 上增加单任务与顺序任务训练。其思路参考
`continual-vla-rl <https://github.com/UT-Austin-RobIn/continual-vla-rl>`_ 的
sequential LoRA fine-tuning：在一个 LIBERO 任务上更新 LoRA adapter，保存策略状态，再用
该状态初始化下一个任务。

该实现同时支持标准 LIBERO 和修改版 LIBERO-OOD。目前实现的是 sequential LoRA
fine-tuning，不包含 continual-vla-rl 中的 EWC、ER、DER、weight merge 等其他 baseline。

实现原理
------------

每个任务都是一次独立的 RLinf run：

.. code-block:: text

   冻结的 StarVLA 基础 checkpoint + 新 LoRA adapter
      → 任务 0 LoRA 训练 → 任务 0 LoRA-wrapped full_weights.pt
      → 任务 1 LoRA 训练 → 任务 1 LoRA-wrapped full_weights.pt
      → ...

对于每个任务，启动器会：

1. 将 ``env.train.task_id_filter`` 设置为一个任务 ID；
2. 强制设置 ``actor.model.is_lora=true``，默认创建 rank-32 LoRA；
3. 启动常规 ``train_embodied_agent.py`` pipeline，PEFT 冻结基础参数，optimizer 只接收
   注入的 LoRA 张量；
4. 在最后一步保存 ``actor/model_state_dict/full_weights.pt``；
5. 下一次 run 重建相同 LoRA 拓扑，并通过 ``runner.ckpt_path`` 加载该文件。

``runner.ckpt_path`` 会将 LoRA-wrapped 模型状态加载到 actor 和 rollout 副本。每个任务
都会重新创建 optimizer、scheduler 和 global step。这是有意设计；
``runner.resume_dir`` 会恢复上一 run 的完整训练状态，不符合这里的逐任务 fine-tuning
语义。

RLinf 版本差异
----------------

上游 continual-vla-rl 基于 RLinf v0.1：它保存 PEFT adapter 目录，并通过
``actor.model.lora_path`` 把上一任务传给下一任务。本实现面向 RLinf v0.3。v0.3 的 FSDP
saver 即使只训练 LoRA，也统一输出 ``full_weights.pt``；actor 和 rollout worker 都支持
``runner.ckpt_path``。因此该文件包含完整的 LoRA-wrapped 序列化状态，但梯度更新仍然只
作用于 LoRA 参数。

不要把 v0.1 的 ``actor/`` adapter 目录作为 ``runner.ckpt_path``。可选 checkpoint 参数
必须是使用相同模型、LoRA rank 和 target-module 拓扑生成的 v0.3 ``full_weights.pt``。

安装
------------

按照 :doc:`StarVLA <starvla>` 安装 StarVLA 和 LIBERO，然后设置初始
Qwen2.5-VL-OFT checkpoint：

.. code-block:: bash

   source .venv-starvla/bin/activate
   export STARVLA_MODEL_PATH=/path/to/Qwen2.5-VL-OFT-LIBERO-4in1

快速开始
------------

默认配置为 ``crl_experiment/libero_spatial_grpo_starvla_qwen25``。

只训练 LIBERO-Spatial 的任务 0：

.. code-block:: bash

   bash examples/crl_experiment/run_embodiment_sequential.sh 0

按顺序训练任务 0 到任务 4，包含两端：

.. code-block:: bash

   bash examples/crl_experiment/run_embodiment_sequential.sh "0,4"

完整参数为：

.. code-block:: text

   run_embodiment_sequential.sh \
      TASK_ID_OR_RANGE [INITIAL_LORA_FULL_WEIGHTS] [MAX_EPOCHS] [CONFIG_NAME] [SEED]

``TASK_ID_OR_RANGE`` 接受一个整数或包含两端的 ``START,END`` 区间。区间训练是顺序
训练，不是多任务联合训练：任意时刻的 rollout 数据只来自一个任务。

默认 LoRA rank 为 32。若要修改，必须为所有任务保持一致：

.. code-block:: bash

   CRL_LORA_RANK=8 \
      bash examples/crl_experiment/run_embodiment_sequential.sh "0,4"

LIBERO-OOD
----------

每个任务训练 10 个 epoch，按顺序训练 Goal-OOD 的任务 0 到 4：

.. code-block:: bash

   bash examples/crl_experiment/run_embodiment_sequential.sh \
      "0,4" "" 10 libero_goal_ood_grpo_starvla_qwen25 42

Spatial-OOD 使用 ``libero_spatial_ood_grpo_starvla_qwen25``。OOD 启动流程会自动选择
``third_party/modified_libero`` 及其随机 BDDL reset 协议。

单张 A800 的 Goal-OOD 配置
------------------------------

对于单张 A800 80GB、14 个 CPU 核心和约 120GB 主机内存，使用以下硬件专用配置
顺序训练全部 10 个 Goal-OOD 任务：

.. code-block:: bash

   export STARVLA_MODEL_PATH=/path/to/Qwen2.5-VL-OFT-LIBERO-4in1
   bash examples/crl_experiment/run_embodiment_sequential.sh \
      "0,9" "" 10 \
      crl_experiment/libero_goal_ood_grpo_lora_starvla_a800_1gpu \
      42

该配置使用一个包含 10 条 rollout 的 GRPO group、10 个并行环境、
``micro_batch_size=4`` 和 BF16，并让 actor 与 rollout 常驻 GPU。同时关闭训练内
评估、视频、模型 offload 和 gradient checkpointing，避免 GPU 空转和不必要的 CPU
传输。如果第一次真实参数更新发生 GPU OOM，先将 ``actor.micro_batch_size``
降为 2；如果仍然内存不足，再开启 actor/rollout offload，并设置
``actor.micro_batch_size=1``。

.. important::

   在 OOD 任务上训练会使它们成为训练数据。此后的分数衡量持续目标套件适应能力，
   而不是零样本 OOD 泛化能力。

从已有权重开始
------------------------------

可选的第二个参数可以是 v0.3 LoRA-wrapped ``full_weights.pt`` 文件，也可以是包含该文件
的目录：

.. code-block:: bash

   bash examples/crl_experiment/run_embodiment_sequential.sh \
      "3,4" \
      /path/to/task_2/actor/model_state_dict/full_weights.pt \
      10 \
      crl_experiment/libero_spatial_grpo_starvla_qwen25 \
      42

对于一个任务区间，传入的 checkpoint 只初始化第一个任务；后续任务会自动使用前一个
任务的输出。全参数 checkpoint 或使用其他 LoRA rank 创建的 checkpoint 无法通过这里的
严格 state-dict 加载。

输出与失败处理
----------------------------

默认输出位于：

.. code-block:: text

   logs/continual/<config>/<run-id>/
      task_0/
      task_1/
      ...
      latest_checkpoint.txt

如果任一任务失败，或者找不到最终的 v0.3 LoRA ``full_weights.pt``，脚本会立即停止。
脚本还会拒绝复用已有输出目录，避免意外覆盖。可以通过 ``CRL_RUN_ID`` 或
``CRL_OUTPUT_ROOT`` 控制输出位置。

顺序训练启动器会关闭训练内验证，并且每个任务只保留一个最终 checkpoint。若要衡量对
先前任务的遗忘，应使用标准 LIBERO 或 LIBERO-OOD 独立评估器评估中间 checkpoint。
