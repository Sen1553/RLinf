Sequential LoRA Fine-Tuning on LIBERO
=====================================

This example adds single-task and sequential task training to RLinf's existing
embodied pipeline. It follows the sequential LoRA fine-tuning recipe from
`continual-vla-rl <https://github.com/UT-Austin-RobIn/continual-vla-rl>`_: train
one LIBERO task by updating a LoRA adapter, save the resulting policy state, and
use that state to initialize the next task.

The implementation supports standard LIBERO and the modified LIBERO-OOD suites.
It currently implements sequential LoRA fine-tuning, not EWC, ER, DER, weight
merge, or the other baselines from continual-vla-rl.

How It Works
------------

Each requested task is an independent RLinf run:

.. code-block:: text

   frozen base StarVLA checkpoint + new LoRA adapter
      → task 0 LoRA training → task 0 LoRA-wrapped full_weights.pt
      → task 1 LoRA training → task 1 LoRA-wrapped full_weights.pt
      → ...

For every task, the launcher:

1. sets ``env.train.task_id_filter`` to one task ID;
2. forces ``actor.model.is_lora=true`` and creates rank-32 LoRA by default;
3. starts the regular ``train_embodied_agent.py`` pipeline, where PEFT freezes
   the base parameters and the optimizer receives the injected LoRA tensors;
4. saves ``actor/model_state_dict/full_weights.pt`` at the final step;
5. reconstructs the same LoRA topology for the next run and loads that file
   through ``runner.ckpt_path``.

``runner.ckpt_path`` loads the LoRA-wrapped model state into both actor and
rollout copies. A new optimizer, scheduler, and global step are created for each
task. This is intentional; ``runner.resume_dir`` would instead resume the
previous run's full training state.

RLinf Version Difference
------------------------

The upstream continual-vla-rl repository is based on RLinf v0.1. It saves a
PEFT adapter directory and passes the preceding task through
``actor.model.lora_path``. This implementation targets RLinf v0.3. The v0.3
FSDP saver emits a unified ``full_weights.pt`` even when only LoRA parameters
are trainable, and actor and rollout workers both understand
``runner.ckpt_path``. Therefore, the file contains the complete serialized
LoRA-wrapped state, but gradient updates are still LoRA-only.

Do not pass a v0.1 ``actor/`` adapter directory as ``runner.ckpt_path``. The
optional checkpoint argument must be a v0.3 ``full_weights.pt`` produced with
the same model, LoRA rank, and target-module topology.

Installation
------------

Install StarVLA and LIBERO as described in :doc:`StarVLA <starvla>`. Set the
initial Qwen2.5-VL-OFT checkpoint:

.. code-block:: bash

   source .venv-starvla/bin/activate
   export STARVLA_MODEL_PATH=/path/to/Qwen2.5-VL-OFT-LIBERO-4in1

Quick Start
-----------

The default config is
``crl_experiment/libero_spatial_grpo_starvla_qwen25``.

Train only task 0 of LIBERO-Spatial:

.. code-block:: bash

   bash examples/crl_experiment/run_embodiment_sequential.sh 0

Train tasks 0 through 4, inclusive:

.. code-block:: bash

   bash examples/crl_experiment/run_embodiment_sequential.sh "0,4"

The full signature is:

.. code-block:: text

   run_embodiment_sequential.sh \
      TASK_ID_OR_RANGE [INITIAL_LORA_FULL_WEIGHTS] [MAX_EPOCHS] [CONFIG_NAME] [SEED]

``TASK_ID_OR_RANGE`` accepts one integer or an inclusive ``START,END`` range.
A range is sequential, not multitask: at any point, rollout data comes from only
one task.

The default LoRA rank is 32. Override it consistently for every task with:

.. code-block:: bash

   CRL_LORA_RANK=8 \
      bash examples/crl_experiment/run_embodiment_sequential.sh "0,4"

LIBERO-OOD
----------

Train Goal-OOD tasks 0 through 4 for 10 epochs per task:

.. code-block:: bash

   bash examples/crl_experiment/run_embodiment_sequential.sh \
      "0,4" "" 10 libero_goal_ood_grpo_starvla_qwen25 42

Use ``libero_spatial_ood_grpo_starvla_qwen25`` for Spatial-OOD. The OOD
launcher automatically selects ``third_party/modified_libero`` and its random
BDDL reset protocol.

Single-A800 Goal-OOD Recipe
---------------------------

For one A800 80GB GPU, 14 CPU cores, and about 120GB host memory, use the
hardware-tuned config below to train all ten Goal-OOD tasks sequentially:

.. code-block:: bash

   export STARVLA_MODEL_PATH=/path/to/Qwen2.5-VL-OFT-LIBERO-4in1
   bash examples/crl_experiment/run_embodiment_sequential.sh \
      "0,9" "" 10 \
      crl_experiment/libero_goal_ood_grpo_lora_starvla_a800_1gpu \
      42

The config uses one ten-rollout GRPO group, ten parallel environments,
``micro_batch_size=4``, BF16, and keeps actor and rollout resident on the GPU.
It disables in-loop evaluation, video recording, model offload, and gradient
checkpointing to avoid idle GPU time and unnecessary CPU transfers. If the
first real update runs out of GPU memory, reduce ``actor.micro_batch_size`` to
2 first. If memory pressure remains, enable actor and rollout offload and set
``actor.micro_batch_size=1``.

.. important::

   Training on OOD tasks makes them training data. Scores after this procedure
   measure continual target-suite adaptation, not zero-shot OOD generalization.

Starting from Existing Weights
------------------------------

The optional second argument may be a v0.3 LoRA-wrapped ``full_weights.pt`` file
or a directory containing one:

.. code-block:: bash

   bash examples/crl_experiment/run_embodiment_sequential.sh \
      "3,4" \
      /path/to/task_2/actor/model_state_dict/full_weights.pt \
      10 \
      crl_experiment/libero_spatial_grpo_starvla_qwen25 \
      42

For a range, the supplied checkpoint initializes only its first task. Every
later task automatically uses the preceding task's output. A full-parameter
checkpoint or a checkpoint created with another LoRA rank is not compatible
with this strict state-dict load.

Outputs and Failure Handling
----------------------------

By default, runs are written below:

.. code-block:: text

   logs/continual/<config>/<run-id>/
      task_0/
      task_1/
      ...
      latest_checkpoint.txt

The script stops immediately if one task fails or if the expected final v0.3
LoRA ``full_weights.pt`` is missing. It also refuses to reuse an existing
output directory, preventing accidental overwrite. Set ``CRL_RUN_ID`` or
``CRL_OUTPUT_ROOT`` to control the output location.

The sequential launcher disables in-loop validation and writes one final
checkpoint per task. Use the normal standalone LIBERO or LIBERO-OOD evaluator
on intermediate checkpoints when measuring forgetting across previous tasks.
