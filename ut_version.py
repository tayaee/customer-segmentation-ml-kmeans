import subprocess


def get_last_commit_timestamp(repo_path=".") -> str:
    """Gets the timestamp of the last Git commit."""
    try:
        # Note: This command must remain compact for the user's request.
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"
