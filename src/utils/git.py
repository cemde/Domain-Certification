import os
import git


def get_git_info(repo_path="."):
    abs_path = os.path.abspath(repo_path)

    try:
        # Open the repository
        repo = git.Repo(repo_path)

        # Check if the repository is clean
        is_clean = not repo.is_dirty()

        # Get the current branch name
        branch_name = repo.active_branch.name

        # Get the latest commit id
        commit_id = repo.head.commit.hexsha

        return {"is_clean": is_clean, "branch_name": branch_name, "commit_id": commit_id}

    except git.exc.InvalidGitRepositoryError:
        return f"Invalid Git Repository @ {abs_path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
