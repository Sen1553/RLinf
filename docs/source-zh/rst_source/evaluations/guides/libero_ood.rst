LIBERO-OOD 评测
===============

RLinf 将 `pi0-text-latent <https://github.com/QuanyiLi/pi0-text-latent>`_ 的
LIBERO-OOD 任务定义与 scorer 改动接入现有 Ray/Hydra 具身评测链路。由于 OOD
套件没有固定 init-state 文件，该流程与标准 LIBERO 评测分开。

运行方式
--------

激活安装了 OpenVLA-OFT 和 ``robosuite==1.4.1`` 的环境，然后设置：

.. code-block:: bash

   export MODEL_PATH=/path/to/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

启动脚本默认读取 ``third_party/modified_libero``。如果 modified LIBERO 位于
其他位置，请将 ``LIBERO_OOD_ROOT`` 设置为它的根目录。

运行两个核心套件：

.. code-block:: bash

   bash evaluations/run_libero_ood_eval.sh libero_goal_ood_openvlaoft_eval
   bash evaluations/run_libero_ood_eval.sh libero_spatial_ood_openvlaoft_eval

额外的诊断套件应单独运行：

.. code-block:: bash

   bash evaluations/run_libero_ood_eval.sh libero_object_ood_openvlaoft_eval

Base-Lora 发布目录含有完整模型分片，并以 ``is_lora: false`` 加载。不要将
``MODEL_PATH`` 指向其中的 ``lora_adapter`` 子目录。

评测协议
--------

OOD 专用环境使用 10 个并行任务环境和 10 个 ``rollout_epoch``。每个任务环境
只使用 seed 7 初始化一次，之后连续随机 reset 10 次。逻辑 trial ID 只用于
调度和指标统计；评测器不会调用 ``get_task_init_states`` 或
``set_init_state``。

Goal-OOD 与 Spatial-OOD 在 10 个稳定步骤后严格执行 300 个策略动作，
Object-OOD 执行 280 个。OpenVLA-OFT 每次预测 8 个动作；对于 300 步套件，
最后一次预测只执行前 4 个动作，避免把协议静默改成 296 或 304 步。

modified LIBERO 提供 OOD BDDL、放宽后的 ``On(object, object)`` XY 阈值和扩大
后的炉灶 base region。启动脚本在 ``logs`` 下生成私有
``LIBERO_CONFIG_PATH``，不会覆盖标准 LIBERO 路径。

结果
----

``eval/success_once`` 是套件整体成功率。每任务指标格式为：

.. code-block:: text

   eval/task_00/success_once
   eval/task_00/num_trajectories
   ...
   eval/task_09/success_once

Goal-OOD 与 Spatial-OOD 应分别报告。核心 20-task 分数为两个套件的总成功数
除以 200，不应计入 Object-OOD。
