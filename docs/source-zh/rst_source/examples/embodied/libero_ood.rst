在 LIBERO-OOD 上评测 OpenVLA-OFT
================================

本示例使用 RLinf 评测
`LIBERO-OOD 基准 <https://github.com/QuanyiLi/pi0-text-latent>`_ 上的
OpenVLA-OFT checkpoint。核心基准由 ``libero_goal_ood`` 与
``libero_spatial_ood`` 组成，共 20 个任务；``libero_object_ood`` 应作为诊断
套件单独报告。

环境
----

- **模拟器**：``pi0-text-latent`` 提供的 modified LIBERO
- **模型输入**：256×256 主视角图像和任务文本
- **动作空间**：LIBERO 7 维增量动作
- **协议**：每任务 10 次随机 reset，seed 7，不读取固定 init-state 文件

依赖安装
--------

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate
   python -m pip install "robosuite==1.4.1"

将 modified LIBERO 源码放在 RLinf 仓库的
``third_party/modified_libero``。该目录下应包含
``libero/libero/bddl_files/libero_goal_ood``。

快速开始
--------

.. code-block:: bash

   export MODEL_PATH=/path/to/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   bash evaluations/run_libero_ood_eval.sh libero_goal_ood_openvlaoft_eval
   bash evaluations/run_libero_ood_eval.sh libero_spatial_ood_openvlaoft_eval

启动脚本会在 ``logs/libero_ood_config`` 下生成独立的 LIBERO 路径配置，并将
modified LIBERO 放到 ``PYTHONPATH`` 最前面；它不会修改 ``~/.libero`` 或标准
LIBERO 安装。
如果需要把源码放在其他位置，可在启动前将 ``LIBERO_OOD_ROOT`` 设置为
modified LIBERO 的根目录。
发布的 Base-Lora 目录已经包含完整模型分片，因此评测时将它作为完整
checkpoint 加载，并设置 ``is_lora: false``，不需要单独加载其中的
``lora_adapter`` 子目录。

评测
----

终端与配置的指标后端会输出套件整体 ``eval/success_once``，以及每个任务的
``eval/task_XX/success_once`` 和 ``eval/task_XX/num_trajectories``。核心基准分数
应以 Goal-OOD 与 Spatial-OOD 共 200 条轨迹的成功数合并计算。完整的 reset、
步数上限和 scorer 协议见 :doc:`../../evaluations/guides/libero_ood`。
