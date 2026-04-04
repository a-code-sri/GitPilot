import os
import shutil
from backend.github.github_cli import run_git_command, git_commit_push

BASE_REPO_DIR = "repos"

# --- HELPER FUNCTIONS ---
def run_gh_command(command: list):
    """Helper to run generic GH CLI commands."""
    cmd = ["gh"] + command
    out, err = run_git_command(cmd)
    # gh often outputs info to stderr even on success, so we check content
    if err and "error" in err.lower() and "warning" not in err.lower():
        return {"error": err.strip()}
    return {"output": out.strip(), "message": "Command executed successfully"}

def ensure_repo_cloned(repo_name):
    """Ensures the repo is cloned locally."""
    os.makedirs(BASE_REPO_DIR, exist_ok=True)
    
    # Handle owner/repo format for folder naming
    folder_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
    repo_path = os.path.join(BASE_REPO_DIR, folder_name)

    if not os.path.exists(os.path.join(repo_path, ".git")):
        # Clean up empty folder if exists
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        
        # Clone
        print(f"Cloning {repo_name}...")
        clone_cmd = ["gh", "repo", "clone", repo_name, repo_path]
        out, err = run_git_command(clone_cmd)
        
        if not os.path.exists(repo_path):
            raise Exception(f"Failed to clone repo: {err}")
            
        # Configure Identity for the agent inside this repo
        run_git_command(["git", "config", "user.email", "agent@gitpilot.ai"], cwd=repo_path)
        run_git_command(["git", "config", "user.name", "GitPilot Agent"], cwd=repo_path)

    return repo_path

# ----------------------
# REPO MANAGEMENT
# ----------------------
def create_repo(repo_name, private=False):
    visibility = "--private" if private else "--public"
    # Create usually doesn't need the consequence flag, but we add --confirm just in case
    return run_gh_command(["repo", "create", repo_name, visibility, "--confirm", "--add-readme"])

def delete_repo(repo_name):
    return run_gh_command(["repo", "delete", repo_name, "--yes"])

def make_repo_private(repo_name):
    # FIXED: Added --accept-visibility-change-consequences
    return run_gh_command(["repo", "edit", f"a-code-sri/{repo_name}", "--visibility", "private", "--accept-visibility-change-consequences"])

def make_repo_public(repo_name):
    # FIXED: Added --accept-visibility-change-consequences
    return run_gh_command(["repo", "edit", repo_name, "--visibility", "public", "--accept-visibility-change-consequences"])

def fork_repo(repo_name):
    return run_gh_command(["repo", "fork", repo_name, "--clone=false"])

def star_repo(repo_name):
    return run_gh_command(["api", f"user/starred/{repo_name}", "-X", "PUT"])

def watch_repo(repo_name):
    return run_gh_command(["repo", "watch", repo_name])

def unwatch_repo(repo_name):
    return run_gh_command(["api", f"repos/{repo_name}/subscription", "-X", "DELETE"])

# ----------------------
# FILE MANAGEMENT
# ----------------------
def add_file(repo_name, file_path, content, commit_message="Add file via GitPilot"):
    try:
        repo_path = ensure_repo_cloned(repo_name)
        full_path = os.path.join(repo_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content if content else "")
            
        git_commit_push(repo_path, commit_message)
        return {"status": "success", "message": f"File {file_path} created/updated."}
    except Exception as e:
        return {"error": str(e)}

def update_file(repo_name, file_path, content, commit_message="Update file via GitPilot"):
    return add_file(repo_name, file_path, content, commit_message)

def delete_file(repo_name, file_path, commit_message="Delete file via GitPilot"):
    try:
        repo_path = ensure_repo_cloned(repo_name)
        full_path = os.path.join(repo_path, file_path)
        if os.path.exists(full_path):
            os.remove(full_path)
            git_commit_push(repo_path, commit_message)
            return {"status": "success", "message": f"File {file_path} deleted."}
        return {"error": "File not found locally."}
    except Exception as e:
        return {"error": str(e)}

# ----------------------
# BRANCHES
# ----------------------
def create_branch(repo_name, branch_name):
    try:
        repo_path = ensure_repo_cloned(repo_name)
        run_git_command(["git", "checkout", "-b", branch_name], cwd=repo_path)
        run_git_command(["git", "push", "-u", "origin", branch_name], cwd=repo_path)
        return {"status": "success", "message": f"Branch {branch_name} created."}
    except Exception as e:
        return {"error": str(e)}

def delete_branch(repo_name, branch_name):
    try:
        repo_path = ensure_repo_cloned(repo_name)
        # Attempt remote delete
        run_git_command(["git", "push", "origin", "--delete", branch_name], cwd=repo_path)
        return {"status": "success", "message": f"Branch {branch_name} deleted."}
    except Exception as e:
        return {"error": str(e)}

def merge_branch(repo_name, head_branch, base_branch):
    try:
        repo_path = ensure_repo_cloned(repo_name)
        run_git_command(["git", "checkout", base_branch], cwd=repo_path)
        run_git_command(["git", "pull"], cwd=repo_path)
        run_git_command(["git", "merge", head_branch], cwd=repo_path)
        run_git_command(["git", "push"], cwd=repo_path)
        return {"status": "success", "message": f"Merged {head_branch} into {base_branch}."}
    except Exception as e:
        return {"error": str(e)}

# ----------------------
# PRs & ISSUES
# ----------------------
def create_pr(repo_name, title, body, head_branch, base_branch):
    cmd = ["pr", "create", "--repo", repo_name, "--title", title, "--body", body]
    if head_branch: cmd.extend(["--head", head_branch])
    if base_branch: cmd.extend(["--base", base_branch])
    return run_gh_command(cmd)

def merge_pr(repo_name, pr_number):
    return run_gh_command(["pr", "merge", str(pr_number), "--merge", "--repo", repo_name])

def close_pr(repo_name, pr_number):
    return run_gh_command(["pr", "close", str(pr_number), "--repo", repo_name])

def create_issue(repo_name, title, body):
    return run_gh_command(["issue", "create", "--repo", repo_name, "--title", title, "--body", body])

def close_issue(repo_name, issue_number):
    return run_gh_command(["issue", "close", str(issue_number), "--repo", repo_name])

def assign_issue(repo_name, issue_number, assignee):
    return run_gh_command(["issue", "edit", str(issue_number), "--add-assignee", assignee, "--repo", repo_name])

def add_label(repo_name, issue_number, label_name):
    return run_gh_command(["issue", "edit", str(issue_number), "--add-label", label_name, "--repo", repo_name])

def remove_label(repo_name, issue_number, label_name):
    return run_gh_command(["issue", "edit", str(issue_number), "--remove-label", label_name, "--repo", repo_name])

# ----------------------
# COLLABORATORS & MISC
# ----------------------
def add_collaborator(repo_name, username):
    """
    Adds a collaborator using GitHub API.
    PUT /repos/{owner}/{repo}/collaborators/{username}
    """
    if "/" not in repo_name:
        return {"error": "repo_name must be in 'owner/repo' format"}
    
    # Use gh api to add collaborator with push permission
    return run_gh_command([
        "api", 
        f"repos/{repo_name}/collaborators/{username}", 
        "-X", "PUT",
        "-f", "permission=push"
    ])

def remove_collaborator(repo_name, username):
    """
    Removes a collaborator using GitHub API.
    DELETE /repos/{owner}/{repo}/collaborators/{username}
    """
    if "/" not in repo_name:
        return {"error": "repo_name must be in 'owner/repo' format"}
    
    return run_gh_command([
        "api", 
        f"repos/{repo_name}/collaborators/{username}", 
        "-X", "DELETE"
    ])