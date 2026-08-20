import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT_CONFIG_DIR = REPO_ROOT / "examples/embodiment/config"
COMMON_FUNCTIONS = REPO_ROOT / "examples/crl_experiment/common_functions.sh"
SEQUENTIAL_SCRIPT = REPO_ROOT / "examples/crl_experiment/run_embodiment_sequential.sh"
EMBODIMENT_SCRIPT = REPO_ROOT / "examples/embodiment/run_embodiment.sh"


def run_common_function(function_call: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "$1"; {function_call}', "bash", str(COMMON_FUNCTIONS)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_parse_single_task():
    result = run_common_function('parse_task_spec "3"')

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["3"]


def test_parse_inclusive_task_range():
    result = run_common_function('parse_task_spec "0,4"')

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["0", "1", "2", "3", "4"]


def test_parse_task_range_rejects_reverse_order():
    result = run_common_function('parse_task_spec "4,0"')

    assert result.returncode != 0
    assert "START < END" in result.stderr


def test_resolve_full_weights_uses_latest_global_step(tmp_path):
    step_2 = (
        tmp_path / "checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
    )
    step_10 = (
        tmp_path / "checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"
    )
    step_2.parent.mkdir(parents=True)
    step_10.parent.mkdir(parents=True)
    step_2.touch()
    step_10.touch()

    result = run_common_function(
        f'resolve_full_weights_checkpoint "{tmp_path.as_posix()}"'
    )

    assert result.returncode == 0
    assert Path(result.stdout.strip()) == step_10


def test_sequential_script_help_lists_single_and_range_usage():
    result = subprocess.run(
        ["bash", str(SEQUENTIAL_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "TASK_ID_OR_RANGE" in result.stdout
    assert '"0,4"' in result.stdout


def test_sequential_script_chains_v03_lora_full_weights_between_tasks(tmp_path):
    fake_launcher = tmp_path / "fake_launcher.sh"
    fake_launcher.write_text(
        """#!/usr/bin/env bash
set -eu
max_epochs=""
experiment_name=""
for arg in "$@"; do
    case "$arg" in
        runner.max_epochs=*) max_epochs=${arg#*=} ;;
        runner.logger.experiment_name=*) experiment_name=${arg#*=} ;;
    esac
done
checkpoint="${RUN_LOG_DIR}/${experiment_name}/checkpoints/global_step_${max_epochs}/actor/model_state_dict/full_weights.pt"
mkdir -p "$(dirname "$checkpoint")"
touch "$checkpoint"
printf '%s\\n' "$*" >> "$CRL_TEST_TRACE"
""",
        encoding="utf-8",
    )
    trace_path = tmp_path / "trace.txt"
    output_root = tmp_path / "output"
    env = {
        "PATH": "/usr/bin:/bin",
        "CRL_EMBODIMENT_LAUNCHER": str(fake_launcher),
        "CRL_OUTPUT_ROOT": str(output_root),
        "CRL_TEST_TRACE": str(trace_path),
    }

    result = subprocess.run(
        ["bash", str(SEQUENTIAL_SCRIPT), "0,2", "", "3"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 3
    assert all("actor.model.is_lora=true" in line for line in trace_lines)
    assert all("actor.model.lora_rank=32" in line for line in trace_lines)
    assert all("actor.model.lora_path=null" in line for line in trace_lines)
    assert "runner.ckpt_path=null" in trace_lines[0]
    assert "task_0/continual_task_0/checkpoints/global_step_3" in trace_lines[1]
    assert "task_1/continual_task_1/checkpoints/global_step_3" in trace_lines[2]
    final_checkpoint = (output_root / "latest_checkpoint.txt").read_text().strip()
    assert "task_2/continual_task_2/checkpoints/global_step_3" in final_checkpoint


def test_sequential_script_supports_lora_rank_override(tmp_path):
    fake_launcher = tmp_path / "fake_launcher.sh"
    fake_launcher.write_text(
        """#!/usr/bin/env bash
set -eu
max_epochs=""
experiment_name=""
for arg in "$@"; do
    case "$arg" in
        runner.max_epochs=*) max_epochs=${arg#*=} ;;
        runner.logger.experiment_name=*) experiment_name=${arg#*=} ;;
    esac
done
checkpoint="${RUN_LOG_DIR}/${experiment_name}/checkpoints/global_step_${max_epochs}/actor/model_state_dict/full_weights.pt"
mkdir -p "$(dirname "$checkpoint")"
touch "$checkpoint"
printf '%s\\n' "$*" > "$CRL_TEST_TRACE"
""",
        encoding="utf-8",
    )
    trace_path = tmp_path / "trace.txt"
    env = {
        "PATH": "/usr/bin:/bin",
        "CRL_EMBODIMENT_LAUNCHER": str(fake_launcher),
        "CRL_OUTPUT_ROOT": str(tmp_path / "output"),
        "CRL_TEST_TRACE": str(trace_path),
        "CRL_LORA_RANK": "8",
    }

    result = subprocess.run(
        ["bash", str(SEQUENTIAL_SCRIPT), "0", "", "1"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "actor.model.is_lora=true" in trace_path.read_text(encoding="utf-8")
    assert "actor.model.lora_rank=8" in trace_path.read_text(encoding="utf-8")


def test_goal_ood_a800_config_composes_for_single_task(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(EMBODIMENT_CONFIG_DIR), version_base=None
    ):
        cfg = compose(
            config_name=("crl_experiment/libero_goal_ood_grpo_lora_starvla_a800_1gpu"),
            overrides=["++env.train.task_id_filter=[7]"],
        )

    assert cfg.env.train.task_suite_name == "libero_goal_ood"
    assert cfg.env.train.task_id_filter == [7]
    assert cfg.env.train.total_num_envs == 10
    assert cfg.env.train.group_size == 10
    assert cfg.actor.model.is_lora is True
    assert cfg.actor.model.lora_rank == 32
    assert cfg.actor.micro_batch_size == 4
    assert cfg.actor.global_batch_size == 512
    assert cfg.actor.enable_offload is False
    assert cfg.rollout.enable_offload is False
    assert cfg.actor.fsdp_config.gradient_checkpointing is False


def test_embodiment_launcher_forwards_hydra_overrides(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" > "$CRL_TEST_TRACE"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    trace_path = tmp_path / "python_args.txt"
    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "CRL_TEST_TRACE": str(trace_path),
        "RUN_LOG_DIR": str(tmp_path / "logs"),
    }

    result = subprocess.run(
        [
            "bash",
            str(EMBODIMENT_SCRIPT),
            "crl_experiment/libero_spatial_grpo_starvla_qwen25",
            "++env.train.task_id_filter=[2]",
            "runner.max_epochs=3",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    forwarded_args = trace_path.read_text(encoding="utf-8")
    assert "--config-name crl_experiment/libero_spatial_grpo_starvla_qwen25" in (
        forwarded_args
    )
    assert "++env.train.task_id_filter=[2]" in forwarded_args
    assert "runner.max_epochs=3" in forwarded_args
    assert "Using ROBOT_PLATFORM=LIBERO" in result.stdout
