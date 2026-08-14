StarVLA 模型强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/starvla.png
   :align: center
   :width: 70%

   StarVLA：模块化的 VLM backbone + action head。

使用 RLinf 对 **StarVLA** 模型进行强化学习微调。StarVLA 是一个开源的
Vision-Language-Action 工具箱，支持将 VLM backbone 与 action head 以模块化方式组合；
本示例采用 **QwenOFT** 设置，在 **LIBERO** 上使用 GRPO 训练。

概览
----------------------------------------

在 LIBERO Spatial 上用 GRPO 微调 StarVLA（QwenOFT）。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO

   .. grid-item-card:: 算法
      :text-align: center

      GRPO

   .. grid-item-card:: 任务
      :text-align: center

      LIBERO Spatial

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · GPU

| **你将完成：** 安装 → 下载 StarVLA checkpoint 与 base VLM → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一个 StarVLA LIBERO checkpoint 与 Qwen2.5-VL 基座（见下文）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

根据环境、任务族以及配置或权重工件选择对应的模型页面。

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - LIBERO
     - LIBERO-Spatial
     - ``libero_spatial_grpo_starvla``
     - 在 LIBERO 上使用 GRPO 微调 StarVLA。
   * - Modified LIBERO
     - LIBERO-Goal-OOD
     - ``libero_goal_ood_grpo_starvla_qwen25``
     - 使用随机 BDDL reset 在目标套件上进行 GRPO 训练。
   * - Modified LIBERO
     - LIBERO-Spatial-OOD
     - ``libero_spatial_ood_grpo_starvla_qwen25``
     - 使用随机 BDDL reset 在目标套件上进行 GRPO 训练。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - 按 StarVLA 格式组织的 LIBERO 图像观测与机器人状态。
   * - Action
     - 通过 StarVLA 策略 API 生成的连续机器人控制命令。
   * - Reward
     - GRPO 使用的 LIBERO 任务成功信号或 shaped reward。
   * - Prompt
     - LIBERO 自然语言任务指令。

接口约定
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 RLinf 的 StarVLA wrapper 中，``env_obs`` 为 batch-first 的 dict（第 0 维为 batch size ``B``）。

必选字段：

* ``main_images``：主视角 RGB，``torch.uint8``，形状 ``[B, H, W, 3]``（常用 ``H=W=224``）。
* ``states``：本体状态，``torch.float32``，形状 ``[B, D_state]``。
* ``task_descriptions``：自然语言任务描述，``list[str]``，长度为 ``B``。

可选字段：

* ``wrist_images``：腕部视角 RGB，``torch.uint8``，形状 ``[B, H, W, 3]``。
* ``extra_view_images``：其他视角 RGB，推荐形状 ``[B, V, H, W, 3]``（``V`` 为额外视角数）。若仅提供单个额外视角，也允许 ``[B, H, W, 3]``，等价视为 ``V=1``。

在 LIBERO 的默认实现中，``states`` 的常见定义为末端位置 ``(x, y, z)``（3 维）、
末端姿态轴角 ``(rx, ry, rz)``（3 维）与夹爪状态（原始 2 维），因此常见
``D_state = 3 + 3 + 2 = 8``。若 checkpoint 期望 7 维状态，wrapper 会将 2 维夹爪状态压缩为
``[x, y, z, rx, ry, rz, g_mean]``，其中 ``g_mean = 0.5 * (g0 + g1)``。

StarVLA 推理输出动作块 ``[B, T, D_action]``，其中
``T = actor.model.num_action_chunks``（planning horizon），
``D_action = actor.model.action_dim``（LIBERO 常用 7）。Rollout 采用 receding-horizon：
每次 forward 产生长度 ``T`` 的动作序列，环境执行前 ``N`` 步（``1 <= N <= T``）后重新规划。

安装
----------------------------------------

.. include:: _setup_common.rst

**选项 1：Docker 镜像** —— 镜像标签 ``agentic-rlinf0.3-maniskill_libero``：

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-maniskill_libero
      # 国内镜像加速：docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-maniskill_libero

   # 进入容器后，切换到 StarVLA 虚拟环境：
   source switch_env starvla

**选项 2：自定义环境** —— 安装套件 ``--env maniskill_libero``：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 --use-mirror。
   bash requirements/install.sh embodied --model starvla --env maniskill_libero
   source .venv/bin/activate

下载模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下载 StarVLA checkpoint 与 base VLM：

.. code-block:: bash

   # 方式1：使用 git clone
   git lfs install
   git clone https://huggingface.co/StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1
   git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

   # 方式2：使用 huggingface-hub（国内可设置 HF_ENDPOINT=https://hf-mirror.com）
   pip install -U huggingface-hub
   hf download StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1 --local-dir ./Qwen2.5-VL-OFT-LIBERO-4in1
   hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

.. note::

   下载完成后，请修改 ``Qwen2.5-VL-OFT-LIBERO-4in1/config.yaml`` 中的
   ``framework.qwenvl.base_vlm``，使其指向 ``Qwen2.5-VL-3B-Instruct`` 的本地路径。

运行
----------------------------------------

**1. 配置**

StarVLA + GRPO + LIBERO Spatial 使用
``examples/embodiment/config/libero_spatial_grpo_starvla.yaml``。将模型路径指向你的下载，并设置动作接口：

.. code-block:: yaml

   defaults:
      - env/libero_spatial@env.train
      - env/libero_spatial@env.eval

   rollout:
     model:
       model_path: "/path/to/model"

   actor:
     model:
       model_path: "/path/to/model"
       action_dim: 7
       num_action_chunks: 8
       action_stats_source: "minmax"
       starvla:
         framework_name: "QwenOFT"
         expected_action_dim: ${actor.model.action_dim}
         expected_num_action_chunks: ${actor.model.num_action_chunks}
         enable_state_input: False

**2. 启动**

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_starvla

评估建议采用 RLinf 统一的评估流程，详见 :doc:`LIBERO 评测指南 <../../evaluations/guides/libero>`。

在 LIBERO-OOD 上训练
----------------------------------------

LIBERO-OOD 配置使用 ``third_party/modified_libero`` 下的修改版 LIBERO。
无需新建另一份启动脚本：``run_embodiment.sh`` 会识别 OOD 配置，将修改版包置于
``PYTHONPATH`` 的最前面，并在 ``logs/libero_ood_config`` 下生成隔离的 LIBERO 路径配置。

设置 Qwen2.5-VL-OFT checkpoint，然后启动所需的目标套件：

.. code-block:: bash

   export STARVLA_MODEL_PATH=/path/to/Qwen2.5-VL-OFT-LIBERO-4in1

   # 10 个 Goal-OOD 任务。
   bash examples/embodiment/run_embodiment.sh \
      libero_goal_ood_grpo_starvla_qwen25

   # 10 个 Spatial-OOD 任务。
   bash examples/embodiment/run_embodiment.sh \
      libero_spatial_ood_grpo_starvla_qwen25

如果修改版仓库不在默认位置，请在启动前将 ``LIBERO_OOD_ROOT`` 指向包含
``libero/libero`` 的目录。

.. important::

   一旦在 Goal-OOD 或 Spatial-OOD 上训练，这些任务就成为训练任务。此时结果衡量的是
   目标套件上的在线适应能力，而不是零样本 OOD 泛化能力。若要研究后者，应只在标准
   LIBERO 上训练，并将全部 20 个 OOD 任务保留为测试集。

StarVLA 每次 forward 预测 8 个动作。常规训练 worker 因此在训练及训练内验证中使用
可整除的 304 步 horizon。训练结束后，应使用独立评估器执行基准严格规定的 300 步、
每任务 10 次协议：

.. code-block:: bash

   MODEL_PATH=/path/to/exported/checkpoint \
   bash evaluations/run_libero_ood_eval.sh \
      libero_goal_ood_starvla_qwen25_oft_eval

提供的 OOD 训练配置启用了 actor/rollout offload，并将 micro batch 设为 1，以降低单卡
显存压力。但 3B VLM、优化器和 rollout 模型在部分 checkpoint 或运行环境中仍可能超过
24 GB；此时需要多卡训练，或继续调整 batch 与优化器配置。

可视化与结果
----------------------------------------

关注任务成功率指标 ``env/success_once``。各项指标的含义见
:doc:`训练指标 <../../reference/metrics>`。

参考曲线（采用的模型来自
`LIBERO_BASELIEN_FORJINHUI_10K_QWENOFT <https://huggingface.co/JasonYang66/LIBERO_BASELIEN_FORJINHUI_10K_QWENOFT>`_）：

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/libero_goal_starvla_baseline.png
   :alt: LIBERO Goal StarVLA baseline result curve
   :width: 95%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/libero_object_starvla_baseline.png
   :alt: LIBERO Object StarVLA baseline result curve
   :width: 95%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/libero_spatial_starvla_baseline.png
   :alt: LIBERO Spatial StarVLA baseline result curve
   :width: 95%
   :align: center
