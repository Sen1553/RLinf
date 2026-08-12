OpenVLA-OFT Evaluation on LIBERO-OOD
====================================

This example evaluates an RLinf OpenVLA-OFT checkpoint on the
`LIBERO-OOD benchmark <https://github.com/QuanyiLi/pi0-text-latent>`_. The core
benchmark contains ``libero_goal_ood`` and ``libero_spatial_ood`` (20 tasks in
total); ``libero_object_ood`` is reported separately as a diagnostic suite.

Environment
-----------

- **Simulator**: the modified LIBERO package shipped by ``pi0-text-latent``
- **Model input**: 256×256 agent-view image and task text
- **Action space**: 7-D LIBERO delta action
- **Protocol**: 10 random-reset trials per task, seed 7, and no fixed init-state files

Dependency Installation
-----------------------

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate
   python -m pip install "robosuite==1.4.1"

Place the modified LIBERO source at
``third_party/modified_libero`` in the RLinf repository. Its package root must
contain ``libero/libero/bddl_files/libero_goal_ood``.

Quick Start
-----------

.. code-block:: bash

   export MODEL_PATH=/path/to/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   bash evaluations/run_libero_ood_eval.sh libero_goal_ood_openvlaoft_eval
   bash evaluations/run_libero_ood_eval.sh libero_spatial_ood_openvlaoft_eval

The launcher creates a dedicated LIBERO path configuration under
``logs/libero_ood_config`` and prepends the modified package to ``PYTHONPATH``;
it does not modify ``~/.libero`` or the standard LIBERO installation.
To keep the source elsewhere, set ``LIBERO_OOD_ROOT`` to the modified LIBERO
repository root before launching.
The released Base-Lora directory contains full model shards, so evaluation loads
it as a complete checkpoint with ``is_lora: false`` rather than loading the
bundled ``lora_adapter`` subdirectory separately.

Evaluation
----------

The terminal and configured metric backend report the suite-wide
``eval/success_once`` plus ``eval/task_XX/success_once`` and
``eval/task_XX/num_trajectories`` for every task. Combine Goal-OOD and
Spatial-OOD success counts over their 200 trajectories for the core benchmark
score. See :doc:`../../evaluations/guides/libero_ood` for the exact reset,
horizon, and scorer protocol.
