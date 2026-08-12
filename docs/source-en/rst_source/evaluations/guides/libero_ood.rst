LIBERO-OOD Evaluation
=====================

RLinf integrates the LIBERO-OOD task definitions and scorer changes from
`pi0-text-latent <https://github.com/QuanyiLi/pi0-text-latent>`_ into the normal
Ray/Hydra embodied evaluation pipeline. This flow is separate from standard
LIBERO evaluation because the OOD suites do not provide fixed init-state files.

Run
---

Activate an OpenVLA-OFT environment with ``robosuite==1.4.1``, then set:

.. code-block:: bash

   export MODEL_PATH=/path/to/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

The launcher uses ``third_party/modified_libero`` by default. If the modified
LIBERO checkout is stored elsewhere, set ``LIBERO_OOD_ROOT`` to its root.

Run the two core suites:

.. code-block:: bash

   bash evaluations/run_libero_ood_eval.sh libero_goal_ood_openvlaoft_eval
   bash evaluations/run_libero_ood_eval.sh libero_spatial_ood_openvlaoft_eval

Run the optional diagnostic suite separately:

.. code-block:: bash

   bash evaluations/run_libero_ood_eval.sh libero_object_ood_openvlaoft_eval

The Base-Lora release contains full model shards and is loaded with
``is_lora: false``. Do not point ``MODEL_PATH`` at its ``lora_adapter``
subdirectory.

Protocol
--------

The OOD-specific environment uses 10 parallel task environments and 10
``rollout_epoch`` values. Each task environment is seeded once with 7 and then
reset sequentially 10 times. The logical trial IDs are used only for scheduling
and metrics: the evaluator never calls ``get_task_init_states`` or
``set_init_state``.

Goal-OOD and Spatial-OOD run exactly 300 policy steps after 10 stabilization
steps; Object-OOD runs 280. OpenVLA-OFT predicts eight actions per inference.
For the 300-step suites the final prediction executes only its first four
actions, avoiding a silent 296- or 304-step protocol change.

The modified LIBERO package supplies the OOD BDDL files, its relaxed
``On(object, object)`` XY threshold, and the enlarged stove base region. The
launcher generates a private ``LIBERO_CONFIG_PATH`` under ``logs`` so standard
LIBERO paths are not overwritten.

Results
-------

``eval/success_once`` is the suite success rate. Per-task metrics are emitted as:

.. code-block:: text

   eval/task_00/success_once
   eval/task_00/num_trajectories
   ...
   eval/task_09/success_once

Report Goal-OOD and Spatial-OOD separately. The core 20-task score is the total
number of successes from both suites divided by 200; do not include Object-OOD.
