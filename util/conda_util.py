import subprocess


def run_conda_command(
        env_name: str,
        command: list[str],
        env_vars: dict = None,
        check: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a command inside a Conda environment, always setting PYTHONPATH.

    :param env_name: Name of the Conda environment to activate.
    :param command: List of command elements to run.
    :param env_vars: Optional additional environment variables to include.
    :param check: If True, raises CalledProcessError on non-zero exit.
    :return: subprocess.CompletedProcess object.
    """
    # Always include PYTHONPATH
    merged_env = {"PYTHONPATH": "/ngen-app/ngen-forcing"}
    if env_vars:
        merged_env.update(env_vars)

    base_cmd = ["conda", "run", "-n", env_name, "--no-capture-output"]
    env_block = ["env"] + [f"{k}={v}" for k, v in merged_env.items()]
    full_cmd = base_cmd + env_block + command

    return subprocess.run(full_cmd, check=check)
