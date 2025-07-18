def __init__(self, directory_path: Path):
    try:
        self.git_repo = git.Repo(directory_path)
    except git.exc.NoSuchPathError:
        self.git_repo = git.Repo.init(directory_path)
    self.root_dir = directory_path
    self.metadata_path = self.root_dir / "metadata.json"
    if not self.metadata_path.exists():
        self.metadata_path.write_text("{}")
        self.git_repo.index.add([str(self.metadata_path)])
        self.git_repo.index.commit("Initialized data repository")

def _scan_directory(self) -> Dict[str, Dict]:
    changed_files = self.git_repo.git.status(porcelain=True).splitlines()
    metadata = json.loads(self.metadata_path.read_text())
    new_files = {}
    for line in changed_files:
        status, filepath = line.strip().split(maxsplit=1)
        abs_path = self.root_dir / filepath
        if status == "??":  # New file
            new_files[str(abs_path)] = _get_file_metadata(abs_path)
        elif status in ["M", "A"]:  # Modified or added file
            new_files[str(abs_path)] = _get_file_metadata(abs_path)
    metadata.update(new_files)
    self.metadata_path.write_text(json.dumps(metadata))
    self.git_repo.index.add([str(self.metadata_path)])
    self.git_repo.index.add([str(path) for path in new_files.keys()])
    return new_files

def update_entry(self, filepath: Path, **kwargs):
    metadata = json.loads(self.metadata_path.read_text())
    file_str = str(filepath.resolve())
    if file_str in metadata:
        metadata[file_str].update(kwargs)
    else:
        metadata[file_str] = kwargs
    self.metadata_path.write_text(json.dumps(metadata))
    self.git_repo.index.add([str(self.metadata_path)])
    self.git_repo.index.add([str(filepath)])
    self.git_repo.index.commit(f"Updated {filepath.name}")

def save(self):
    self.git_repo.git.add(all=True)
    self.git_repo.index.commit("Repository snapshot")

def get_summary_metadata(self):
    commit_count = len(list(self.git_repo.iter_commits()))
    total_size = sum(blob.size for commit in self.git_repo.iter_commits() for blob in commit.tree.blobs)
    date_range = [commit.committed_datetime.isoformat() for commit in self.git_repo.iter_commits()]
    return {
        "file_count": len(self.git_repo.git.ls_files().splitlines()),
        "total_size": total_size,
        "commit_count": commit_count,
        "date_range": [min(date_range), max(date_range)]
    }

def get_files_by_status(self, status: Union[str, List[str]]) -> List[Path]:
    status_output = self.git_repo.git.status(porcelain=True).splitlines()
    files = []
    for line in status_output:
        status_code, filepath = line.strip().split(maxsplit=1)
        if isinstance(status, str) and status_code == status:
            files.append(self.root_dir / filepath)
        elif isinstance(status, list) and status_code in status:
            files.append(self.root_dir / filepath)
    return files

def scan_and_update(self, base_dir: Path):
    self.git_repo.git.add(str(base_dir))
    self.git_repo.git.commit("-m", "Updated repository")

def get_status(self, filepath: Path) -> Optional[str]:
    status_output = self.git_repo.git.status("--porcelain", str(filepath))
    if status_output:
        return status_output.strip().split(maxsplit=1)[0]
    return None

def get_processed_path(self, source_filepath: Path) -> Optional[Path]:
    metadata = json.loads(self.metadata_path.read_text())
    file_str = str(source_filepath.resolve())
    if file_str in metadata and "processed_path" in metadata[file_str]:
        return Path(metadata[file_str]["processed_path"])
    return None

def save_repo(self):
    self.git_repo.git.add(all=True)
    self.git_repo.index.commit("Saved repository state")

def load_progress(self, process_id: str) -> Optional[Dict[str, Any]]:
    metadata = json.loads(self.metadata_path.read_text())
    if "progress" in metadata and process_id in metadata["progress"]:
        return metadata["progress"][process_id]
    return None

def record_error(self, filepath: Path, error_msg: str):
    metadata = json.loads(self.metadata_path.read_text())
    file_str = str(filepath.resolve())
    if "errors" not in metadata:
        metadata["errors"] = {}
    metadata["errors"][file_str] = error_msg
    self.metadata_path.write_text(json.dumps(metadata))
    self.git_repo.index.add([str(self.metadata_path)])
    self.git_repo.index.commit(f"Recorded error for {filepath.name}")

def _load_repo_dataframe(self) -> pd.DataFrame:
    metadata = json.loads(self.metadata_path.read_text())
    data = []
    for file_str, file_metadata in metadata.items():
        if file_str != "errors" and file_str != "progress":
            data.append({"filepath": file_str, **file_metadata})
    return pd.DataFrame(data)

def _save_repo_dataframe(self, df: pd.DataFrame):
    metadata = {}
    for index, row in df.iterrows():
        metadata[row["filepath"]] = row.to_dict()
    self.metadata_path.write_text(json.dumps(metadata))
    self.git_repo.index.add([str(self.metadata_path)])
    self.git_repo.index.commit("Updated repository metadata")

def _get_target_repo_path(self, filepath: Path) -> Path:
    for submodule in self.git_repo.submodules:
        if filepath.resolve().startswith(submodule.path):
            return Path(submodule.path)
    return self.root_dir

def add_sub_repository(self, sub_dir: Path, url: str):
    self.git_repo.git.submodule("add", url, str(sub_dir))
    self.git_repo.index.commit(f"Added submodule {sub_dir.name}")

def _validate_sub_repository(self, sub_dir: Path) -> bool:
    try:
        git.Repo(sub_dir)
        return True
    except git.exc.NoSuchPathError:
        return False
    except git.exc.InvalidGitRepositoryError:
        return False

def configure_lfs(self):
    self.git_repo.git.lfs("install")
    with (self.root_dir / ".gitattributes").open("a") as f:
        f.write("*.zst filter=lfs diff=lfs merge=lfs\n")
    self.git_repo.git.add(".gitattributes")
    self.git_repo.index.commit("Configured LFS")

def get_file_history(self, filepath: Path) -> List[Dict]:
    log_output = self.git_repo.git.log("--follow", "--format=%H|%ad|%s", str(filepath))
    commits = []
    for line in log_output.splitlines():
        commit_hash, commit_date, commit_message = line.split("|", 2)
        commits.append({
            "commit_hash": commit_hash,
            "commit_date": commit_date,
            "commit_message": commit_message
        })
    return commits

def parallel_scan(self):
    files = self.git_repo.git.ls_files().splitlines()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file) for file in files]
        results = [future.result() for future in futures]
    return results

def create_version(self, version: str, notes: str = ""):
    self.git_repo.git.tag("-a", version, "-m", notes)
    self.git_repo.git.push("origin", version)

def load_version(self, version: str):
    self.git_repo.git.checkout(version)

def add_remote_backup(self, url: str):
    self.git_repo.git.remote("add", "backup", url)
    self.git_repo.git.push("backup", "HEAD")

def verify_backup(self):
    backup_repo = git.Repo(self.root_dir / ".git" / "backup")
    fsck_output = backup_repo.git.fsck()
    if "dangling" in fsck_output or "missing" in fsck_output:
        return False
    return True

def verify_file_integrity(self, filepath: Path):
    expected_hash = self.git_repo.git.hash_object(str(filepath))
    actual_hash = self.git_repo.git.ls_files("-s", str(filepath)).split()[1]
    return expected_hash == actual_hash

def repair_file(self, filepath: Path):
    self.git_repo.git.checkout("HEAD", "--", str(filepath))

def get_status(self, filepath: Path):
    try:
        status_output = self.git_repo.git.status("--porcelain", str(filepath))
        return status_output.strip().split(maxsplit=1)[0]
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def save_progress(self, process_id: str, current_state: Dict[str, Any]):
    progress_file = self.root_dir / f"progress_{process_id}.json"
    with open(progress_file, "w") as f:
        json.dump(current_state, f)
    self.git_repo.git.add(str(progress_file))
    self.git_repo.index.commit(f"Saved progress for {process_id}")

def load_progress(self, process_id: str):
    try:
        progress_file = self.root_dir / f"progress_{process_id}.json"
        with open(progress_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        return f"Error: {e}"
    
def get_files_by_status(self, status: str):
    status_output = self.git_repo.git.status("--porcelain")
    files = []
    for line in status_output.splitlines():
        if line.startswith(status):
            files.append(line.split(maxsplit=2)[1])
    return files

def get_summary_metadata(self):
    metadata = {
        "commit_count": len(list(self.git_repo.iter_commits())),
        "branch_count": len(self.git_repo.branches),
        "tag_count": len(self.git_repo.tags)
    }
    return metadata

def parallel_scan(self):
    files = self.git_repo.git.ls_files().splitlines()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(self.process_file, file) for file in files]
        results = [future.result() for future in futures]
    return results

def process_file(self, file):
    # Process the file here
    pass

def get_file_history(self, filepath: Path):
    log_output = self.git_repo.git.log("--follow", "--format=%H|%an|%ad|%s", str(filepath))
    commits = []
    for line in log_output.splitlines():
        commit_hash, author, date, message = line.split("|", 3)
        commits.append({
            "commit_hash": commit_hash,
            "author": author,
            "date": date,
            "message": message
        })
    return commits

def get_diff(self, commit_hash1: str, commit_hash2: str = None):
    if commit_hash2:
        diff_output = self.git_repo.git.diff(commit_hash1, commit_hash2)
    else:
        diff_output = self.git_repo.git.diff(commit_hash1)
    return diff_output

def get_blame(self, filepath: Path):
    blame_output = self.git_repo.git.blame(str(filepath))
    return blame_output

def get_commit(self, commit_hash: str):
    commit = self.git_repo.commit(commit_hash)
    return {
        "author": commit.author.name,
        "date": commit.authored_datetime,
        "message": commit.message
    }

def merge_commits(self, commit_hash1: str, commit_hash2: str):
    try:
        self.git_repo.git.merge(commit_hash2)
        self.git_repo.git.commit("-m", "Merged commits")
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def revert_commit(self, commit_hash: str):
    try:
        self.git_repo.git.revert(commit_hash)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def reset_repository(self, commit_hash: str):
    try:
        self.git_repo.git.reset("--hard", commit_hash)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def clean_repository(self):
    try:
        self.git_repo.git.clean("-fdx")
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def clean_repository(self):
    try:
        self.git_repo.git.clean("-fdx")
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_repository_status(self):
    try:
        status_output = self.git_repo.git.status("--porcelain")
        return status_output
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_history(self):
    try:
        log_output = self.git_repo.git.log("--format=%H|%an|%ad|%s")
        commits = []
        for line in log_output.splitlines():
            commit_hash, author, date, message = line.split("|", 3)
            commits.append({
                "commit_hash": commit_hash,
                "author": author,
                "date": date,
                "message": message
            })
        return commits
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_branches(self):
    try:
        branch_output = self.git_repo.git.branch()
        branches = [branch.strip() for branch in branch_output.splitlines()]
        return branches
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_tags(self):
    try:
        tag_output = self.git_repo.git.tag()
        tags = [tag.strip() for tag in tag_output.splitlines()]
        return tags
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def create_branch(self, branch_name: str):
    try:
        self.git_repo.git.branch(branch_name)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def delete_branch(self, branch_name: str):
    try:
        self.git_repo.git.branch("-d", branch_name)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def checkout_branch(self, branch_name: str):
    try:
        self.git_repo.git.checkout(branch_name)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def merge_branch(self, branch_name: str):
    try:
        self.git_repo.git.merge(branch_name)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def push_changes(self, remote_name: str = "origin", branch_name: str = "main"):
    try:
        self.git_repo.git.push(remote_name, branch_name)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def pull_changes(self, remote_name: str = "origin", branch_name: str = "main"):
    try:
        self.git_repo.git.pull(remote_name, branch_name)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def fetch_changes(self, remote_name: str = "origin"):
    try:
        self.git_repo.git.fetch(remote_name)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_remote_url(self, remote_name: str = "origin"):
    try:
        remote_url = self.git_repo.git.remote("get-url", remote_name)
        return remote_url
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def add_remote(self, remote_name: str, remote_url: str):
    try:
        self.git_repo.git.remote("add", remote_name, remote_url)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def remove_remote(self, remote_name: str):
    try:
        self.git_repo.git.remote("remove", remote_name)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_diff(self, commit_hash1: str, commit_hash2: str):
    try:
        diff_output = self.git_repo.git.diff(commit_hash1, commit_hash2)
        return diff_output
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_file_diff(self, commit_hash1: str, commit_hash2: str, file_path: str):
    try:
        diff_output = self.git_repo.git.diff(commit_hash1, commit_hash2, file_path)
        return diff_output
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_message(self, commit_hash: str):
    try:
        commit_message = self.git_repo.git.log("-1", "--format=%s", commit_hash)
        return commit_message
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_author(self, commit_hash: str):
    try:
        commit_author = self.git_repo.git.log("-1", "--format=%an", commit_hash)
        return commit_author
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_date(self, commit_hash: str):
    try:
        commit_date = self.git_repo.git.log("-1", "--format=%ad", commit_hash)
        return commit_date
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_repository_root(self):
    try:
        repository_root = self.git_repo.git.rev_parse("--show-toplevel")
        return repository_root
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def is_git_repository(self, directory: str):
    try:
        self.git_repo.git.rev_parse("--is-inside-work-tree", cwd=directory)
        return True
    except git.exc.GitCommandError:
        return False
    
def get_gitignore(self):
    try:
        with open(self.git_repo.working_tree_dir + "/.gitignore", "r") as f:
            return f.read()
    except FileNotFoundError:
        return None
    
def get_gitignore(self):
    try:
        with open(self.git_repo.working_tree_dir + "/.gitignore", "r") as f:
            return f.read()
    except FileNotFoundError:
        return None
    
def remove_from_gitignore(self, pattern: str):
    try:
        with open(self.git_repo.working_tree_dir + "/.gitignore", "r+") as f:
            lines = f.readlines()
            f.seek(0)
            for line in lines:
                if line.strip() != pattern:
                    f.write(line)
            f.truncate()
        return True
    except Exception as e:
        return f"Error: {e}"
    
def get_submodules(self):
    try:
        submodule_output = self.git_repo.git.submodule("status")
        submodules = []
        for line in submodule_output.splitlines():
            submodules.append(line.split()[1])
        return submodules
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def add_submodule(self, submodule_url: str, submodule_path: str):
    try:
        self.git_repo.git.submodule("add", submodule_url, submodule_path)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def update_submodule(self, submodule_path: str):
    try:
        self.git_repo.git.submodule("update", "--remote", submodule_path)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def remove_submodule(self, submodule_path: str):
    try:
        self.git_repo.git.submodule("deinit", submodule_path)
        self.git_repo.git.rm(submodule_path)
        return True
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_tags_with_message(self):
    try:
        tag_output = self.git_repo.git.tag("-n")
        tags = {}
        for line in tag_output.splitlines():
            tag_name, tag_message = line.split(maxsplit=1)
            tags[tag_name] = tag_message.strip()
        return tags
    except git.exc.GitCommandError as e:
        return f"Error: {e}"

def get_commits_between_tags(self, tag1: str, tag2: str):
    try:
        commit_output = self.git_repo.git.log(f"{tag1}..{tag2}", "--format=%H")
        commits = commit_output.splitlines()
        return commits
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_count(self):
    try:
        commit_count = self.git_repo.git.rev_list("--count", "HEAD")
        return int(commit_count)
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_last_commit_timestamp(self):
    try:
        timestamp = self.git_repo.git.log("-1", "--format=%ct")
        return int(timestamp)
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_authors(self):
    try:
        author_output = self.git_repo.git.shortlog("-sne")
        authors = []
        for line in author_output.splitlines():
            author_info = line.split("\t")
            authors.append({"name": author_info[1].split("<")[0].strip(), "email": author_info[1].split("<")[1].strip(">")})
        return authors
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_contributors(self):
    try:
        contributor_output = self.git_repo.git.shortlog("-sne")
        contributors = []
        for line in contributor_output.splitlines():
            contributor_info = line.split("\t")
            contributors.append({"name": contributor_info[1].split("<")[0].strip(), "email": contributor_info[1].split("<")[1].strip(">")})
        return contributors
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_history_by_author(self, author: str):
    try:
        commit_output = self.git_repo.git.log("--author", author, "--format=%H")
        commits = commit_output.splitlines()
        return commits
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_lines_added_by_author(self, author: str):
    try:
        lines_added = 0
        for file in self.git_repo.git.ls_files().splitlines():
            blame_output = self.git_repo.git.blame("-p", "--", file)
            for line in blame_output.splitlines():
                if line.startswith("author "):
                    if line[7:] == author:
                        lines_added += 1
        return lines_added
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_lines_deleted_by_author(self, author: str):
    try:
        lines_deleted = 0
        log_output = self.git_repo.git.log("--author", author, "--numstat", "--format=")
        for line in log_output.splitlines():
            stats = line.split()
            if len(stats) > 1:
                lines_deleted += int(stats[1])
        return lines_deleted
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_files_changed_by_author(self, author: str):
    try:
        files_changed = self.git_repo.git.log("--author", author, "--name-only", "--format=")
        return files_changed.splitlines()
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commits_by_date(self, date: str):
    try:
        commits = self.git_repo.git.log(f"--since={date}", f"--until={date} 23:59:59", "--format=%H")
        return commits.splitlines()
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_tags_by_date(self, date: str):
    try:
        tags = self.git_repo.git.tag("--format=%(creatordate:short) %(refname:short)")
        tags_on_date = []
        for tag in tags.splitlines():
            tag_date, tag_name = tag.split()
            if tag_date == date:
                tags_on_date.append(tag_name)
        return tags_on_date
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_branches_by_commit(self, commit_hash: str):
    try:
        branches = self.git_repo.git.branch("--contains", commit_hash)
        return [branch.strip() for branch in branches.splitlines()]
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commits_by_branch(self, branch_name: str):
    try:
        commits = self.git_repo.git.log(branch_name, "--format=%H")
        return commits.splitlines()
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_merge_commits(self):
    try:
        merge_commits = self.git_repo.git.log("--merges", "--format=%H")
        return merge_commits.splitlines()
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_count_by_author(self, author: str):
    try:
        commit_count = self.git_repo.git.shortlog("-s", "--author", author)
        return int(commit_count.split()[0])
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_count_by_date(self, date: str):
    try:
        commit_count = self.git_repo.git.rev_list(f"--since={date}", f"--until={date} 23:59:59", "--count", "HEAD")
        return int(commit_count)
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_authors_by_commit(self, commit_hash: str):
    try:
        authors = self.git_repo.git.log(commit_hash, "-1", "--format=%an")
        return authors
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_committers_by_commit(self, commit_hash: str):
    try:
        committers = self.git_repo.git.log(commit_hash, "-1", "--format=%cn")
        return committers
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_message_by_hash(self, commit_hash: str):
    try:
        commit_message = self.git_repo.git.log(commit_hash, "-1", "--format=%s")
        return commit_message
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_diff_by_hash(self, commit_hash: str):
    try:
        commit_diff = self.git_repo.git.show(commit_hash)
        return commit_diff
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_stats_by_hash(self, commit_hash: str):
    try:
        commit_stats = self.git_repo.git.show(commit_hash, "--stat")
        return commit_stats
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_files_by_hash(self, commit_hash: str):
    try:
        commit_files = self.git_repo.git.show(commit_hash, "--name-only")
        return commit_files.splitlines()
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_last_commit_hash(self):
    try:
        last_commit_hash = self.git_repo.git.log("-1", "--format=%H")
        return last_commit_hash
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_count_between_commits(self, start_commit: str, end_commit: str):
    try:
        commit_count = self.git_repo.git.rev_list(f"{start_commit}..{end_commit}", "--count")
        return int(commit_count)
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commits_between_commits(self, start_commit: str, end_commit: str):
    try:
        commits = self.git_repo.git.rev_list(f"{start_commit}..{end_commit}")
        return commits.splitlines()
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_diff_between_commits(self, start_commit: str, end_commit: str):
    try:
        commit_diff = self.git_repo.git.diff(start_commit, end_commit)
        return commit_diff
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_stats_between_commits(self, start_commit: str, end_commit: str):
    try:
        commit_stats = self.git_repo.git.diff(start_commit, end_commit, "--stat")
        return commit_stats
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_files_changed_between_commits(self, start_commit: str, end_commit: str):
    try:
        files_changed = self.git_repo.git.diff(start_commit, end_commit, "--name-only")
        return files_changed.splitlines()
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_authors_between_commits(self, start_commit: str, end_commit: str):
    try:
        authors = self.git_repo.git.shortlog(f"{start_commit}..{end_commit}", "-sne")
        return authors.splitlines()
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_count_by_author_between_commits(self, author: str, start_commit: str, end_commit: str):
    try:
        commit_count = self.git_repo.git.rev_list(f"{start_commit}..{end_commit}", "--author", author, "--count")
        return int(commit_count)
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commits_by_author_between_commits(self, author: str, start_commit: str, end_commit: str):
    try:
        commits = self.git_repo.git.rev_list(f"{start_commit}..{end_commit}", "--author", author)
        return commits.splitlines()
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_diff_by_author_between_commits(self, author: str, start_commit: str, end_commit: str):
    try:
        commit_diff = self.git_repo.git.log(f"{start_commit}..{end_commit}", "--author", author, "-p")
        return commit_diff
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_stats_by_author_between_commits(self, author: str, start_commit: str, end_commit: str):
    try:
        commit_stats = self.git_repo.git.log(f"{start_commit}..{end_commit}", "--author", author, "--stat")
        return commit_stats
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_files_changed_by_author_between_commits(self, author: str, start_commit: str, end_commit: str):
    try:
        files_changed = self.git_repo.git.log(f"{start_commit}..{end_commit}", "--author", author, "--name-only")
        return files_changed.splitlines()
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_last_commit_hash_by_author(self, author: str):
    try:
        last_commit_hash = self.git_repo.git.log("--author", author, "-1", "--format=%H")
        return last_commit_hash
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_count_by_committer(self, committer: str):
    try:
        commit_count = self.git_repo.git.log("--committer", committer, "--count")
        return int(commit_count)
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commits_by_committer(self, committer: str):
    try:
        commits = self.git_repo.git.log("--committer", committer, "--format=%H")
        return commits.splitlines()
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_diff_by_committer(self, committer: str):
    try:
        commit_diff = self.git_repo.git.log("--committer", committer, "-p")
        return commit_diff
    except git.exc.GitCommandError as e:
        return f"Error: {e}"
    
def get_commit_stats_by_committer(self, committer: str):
    try:
        commit_stats = self.git_repo.git.log("--committer", committer, "--stat")
        return commit_stats
    except git.exc.GitCommandError as e:
        return f"Error: {e}"