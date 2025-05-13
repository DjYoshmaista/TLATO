**Refined Proposals for `src/core/repo_handler.py`**

**I. Core Repository Setup and `metadata.json` Management**

* **1. Standardize `metadata.json` Structure (Refined):**
    * The `metadata.json` file, managed by `MetadataFileHandler`, will store a dictionary where keys are file paths relative to the repository root. Each value will be a dictionary with the following *standardized fields* (extensible as needed):
        ```json
        {
            "filepath_relative": "str, path relative to repo root",
            "filename": "str, file name with extension",
            "extension": "str, file extension",
            "size_bytes": "int, size in bytes",
            "os_last_modified_utc": "str, ISO 8601 timestamp of OS mtime",
            "git_object_hash_current": "str, Git blob hash of the current version in HEAD (if tracked and committed)",
            "custom_hashes": { // Calculated on demand or when 'added' via a specific method
                "md5": "str, MD5 hash of the file content",
                "sha256": "str, SHA256 hash of the file content"
            },
            "date_added_to_metadata_utc": "str, ISO 8601 timestamp when entry first added to metadata.json",
            "last_metadata_update_utc": "str, ISO 8601 timestamp of last update to this metadata entry",
            "application_status": "str, e.g., 'new', 'active', 'processed', 'archived', 'error'",
            "user_metadata": {}, // User-defined key-value pairs
            "version_history_app": [ // Optional: for app-level change logging to this metadata entry
                // { "timestamp_utc": "...", "change_description": "...", "details": {...} }
            ]
        }
        ```
    * **Rationale:** Using relative paths as keys is crucial for portability. `git_object_hash_current` links to Git's own versioning. `custom_hashes` provides the MD5/SHA256. `application_status` allows for workflow management. `version_history_app` can track changes *to the metadata entry itself* if needed, distinct from Git's file content history.

* **2. Concurrency for `metadata.json` (Refined):**
    * Given that `MetadataFileHandler.write_metadata` can be called before a commit, and multiple operations might try to modify it, a `threading.Lock` should be added to `MetadataFileHandler` to protect read-modify-write sequences on the `self.metadata_path` file *before* it's committed by Git.
        ```python
        # In MetadataFileHandler.__init__
        self.file_lock = threading.Lock()

        # In MetadataFileHandler.read_metadata
        # with self.file_lock: # Consider if lock needed for read if writes are locked
        #     ... read logic ...

        # In MetadataFileHandler.write_metadata (before Git commit part)
        with self.file_lock:
            # ... ensure parent dir exists ...
            # ... open and json.dump ...
        # ... then proceed to commit if requested ...
        ```
    * The `RepoHandler` methods orchestrating these should be designed knowing this. For operations that involve multiple steps on `metadata.json` before a final commit, the lock should be acquired at the beginning of the sequence in the `RepoHandler` method and released at the end. However, more granular locks within `MetadataFileHandler` for its read/write might be simpler. The critical section is the actual `open()`, `json.load()`, modification of the Python dict, `open()`, and `json.dump()`.

**II. File Addition and Tracking (The `add_data_file` Equivalent)**

* **1. Enhance `_get_file_metadata` (Placeholder or Actual):**
    * **Location:** Ideally, the actual implementation in `src.utils.helpers._get_file_metadata` should be enhanced. If not, the placeholder in `src/core/repo_handler.py` needs to be made fully functional.
    * **Functionality:**
        * Accept `file_path: Path`.
        * Return a dictionary including: `filename`, `extension`, `size_bytes`, `os_last_modified_utc`.
        * **Crucially, integrate hashing:** Use `src.utils.hashing.hash_file` (assuming it's available and provides MD5/SHA256) to calculate and include `md5` and `sha256` in the returned dict.
            ```python
            # In the enhanced _get_file_metadata or its helper
            # from src.utils.hashing import hash_file # Assuming this function exists and works
            # calculated_hashes = hash_file(filepath=str(file_path), current_size=file_path.stat().st_size)
            # metadata_dict["custom_hashes"] = {"md5": calculated_hashes.get("md5"), "sha256": calculated_hashes.get("sha256")}
            ```
        * The `LOG_INS` generation should be consistent with the project's standard.

* **2. Add `RepoHandler.add_file_to_tracking(self, file_to_add: Path, user_metadata: Optional[Dict[str, Any]] = None, application_status: str = "new", commit: bool = True, commit_message: Optional[str] = None) -> bool` (Refined `add_data_file`):**
    * **Method in `RepoHandler` class.**
    * **Logic:**
        1.  `_log_ins_val = _get_log_ins(inspect.currentframe(), self.__class__.__name__)`
        2.  Resolve `file_to_add` to an absolute path. Check if it's within `self.root_dir`. If not, log error and return `False`.
        3.  If `not file_to_add.exists() or not file_to_add.is_file()`, log error and return `False`.
        4.  Call the *enhanced* `_get_file_metadata(file_to_add)` (from `src.utils.helpers` or the functional placeholder) to get OS stats and custom hashes.
        5.  `current_time_utc_iso = dt.now(timezone.utc).isoformat()`
        6.  `relative_file_path_str = str(file_to_add.relative_to(self.root_dir))`
        7.  Prepare `new_entry_data` based on the standardized `metadata.json` structure:
            ```python
            entry_data = {
                "filepath_relative": relative_file_path_str,
                "filename": file_to_add.name,
                "extension": file_to_add.suffix,
                "size_bytes": file_metadata_from_helper.get("size_bytes"),
                "os_last_modified_utc": file_metadata_from_helper.get("os_last_modified_utc"),
                "custom_hashes": file_metadata_from_helper.get("custom_hashes", {}),
                "date_added_to_metadata_utc": current_time_utc_iso,
                "last_metadata_update_utc": current_time_utc_iso,
                "application_status": application_status,
                "user_metadata": user_metadata or {},
                "version_history_app": [{"timestamp_utc": current_time_utc_iso, "change_description": "Initial addition to tracking"}]
            }
            ```
        8.  Read current `metadata.json` using `self.metadata_handler.read_metadata()`.
        9.  Update the `metadata` dict: `metadata[relative_file_path_str] = entry_data`.
        10. Write updated `metadata` back using `self.metadata_handler.write_metadata(metadata, commit_message=None)` (write only, no commit from handler yet). If write fails, log error and return `False`.
        11. If `commit` is `True`:
            * `files_to_commit = [file_to_add, self.metadata_handler.metadata_path]`
            * `msg = commit_message or f"Added file {file_to_add.name} and updated metadata"`
            * `success = self.modifier._commit_changes(files_to_commit, msg)`
            * Return `success`.
        12. Return `True` if `commit` is `False` and metadata write was successful.
    * **Logging:** Use `actual_log_statement` with `LOG_INS` for all significant steps.

* **3. Add `RepoHandler.add_files_to_tracking_threaded(...)` (Refined `add_data_files_threaded`):**
    * **Strategy:** Parallel metadata gathering, single metadata update, single commit.
    * **Logic:**
        1.  Define a helper function `_prepare_file_metadata_for_add(file_path, user_meta, app_status)` that performs steps 2-7 from `add_file_to_tracking` (path validation, call enhanced `_get_file_metadata`, prepare `entry_data`) and returns `(relative_path_str, entry_data)` or `None` on error.
        2.  Use `ThreadPoolExecutor` to call `_prepare_file_metadata_for_add` for all `file_paths`.
        3.  Collect valid results.
        4.  If no valid results, return empty success map or `False`.
        5.  In main thread:
            * Acquire lock for `metadata.json` if deemed necessary (see I.2 refinement).
            * `metadata = self.metadata_handler.read_metadata()`.
            * For each `(rel_path, entry)` from parallel results, update `metadata[rel_path] = entry`.
            * `self.metadata_handler.write_metadata(metadata, commit_message=None)`.
            * Release lock.
            * If write fails, log and return failure status for all.
        6.  If `commit` is `True`:
            * Collect all original `file_paths` (that were successfully processed) and `self.metadata_handler.metadata_path`.
            * `self.modifier._commit_changes(all_files_to_commit, "Batch added files and updated metadata")`.
        7.  Return dictionary `Dict[Path, bool]` indicating success/failure for each input file.

**III. Application Status Management**

* **1. `RepoHandler.update_file_application_status(...)` (Refined):**
    * **Method in `RepoHandler` class.**
    * **Logic:**
        1.  `_log_ins_val = _get_log_ins(inspect.currentframe(), self.__class__.__name__)`
        2.  `resolved_path = (self.root_dir / file_path).resolve()` (if `file_path` can be relative).
        3.  `file_key = str(resolved_path.relative_to(self.root_dir))`.
        4.  Read metadata: `metadata = self.metadata_handler.read_metadata()`.
        5.  If `file_key not in metadata`, log warning "File not tracked in metadata" and return `False`.
        6.  Prepare update: `updates_for_metadata = {"application_status": new_status, "last_metadata_update_utc": dt.now(timezone.utc).isoformat()}`.
        7.  Optionally add to `version_history_app` in `metadata[file_key]`.
        8.  `metadata[file_key].update(updates_for_metadata)`.
        9.  `final_commit_message = commit_message or f"Updated application status for {resolved_path.name} to {new_status}"`.
        10. `write_success = self.metadata_handler.write_metadata(metadata, commit_message=final_commit_message if commit else None)`.
            * Note: `MetadataFileHandler.write_metadata` itself handles the commit if `commit_message` is passed to it. So this call is sufficient if we want `MetadataFileHandler` to do the commit.
            * Alternatively, `self.metadata_handler.write_metadata(metadata)` then `self.modifier._commit_changes([self.metadata_handler.metadata_path], final_commit_message)`.
        11. Return `write_success`.

* **2. `RepoHandler.get_files_by_application_status(...)` (Refined):**
    * **Method in `RepoHandler` class.**
    * **Logic using DataFrame for potentially better performance/ease:**
        1.  `df = self.load_repository_as_dataframe()`.
        2.  If `df is None or df.empty or "application_status" not in df.columns`, return `[]`.
        3.  `target_statuses = [status_filter] if isinstance(status_filter, str) else status_filter`.
        4.  `filtered_df = df[df["application_status"].isin(target_statuses)]`.
        5.  Identify the correct column for file paths (e.g., `filepath_original_key` or a standardized `filepath_relative` if ensured by `load_repository_as_dataframe`).
        6.  Return `[self.root_dir / rel_path for rel_path in filtered_df[path_column_name]]`.
    * Ensure `load_repository_as_dataframe` correctly exposes the relative path for reconstruction.

**IV. Retrieving All Tracked Files (`get_all_managed_files`)**

* **`RepoHandler.get_all_managed_files(self) -> List[Path]` (Refined):**
    * **Method in `RepoHandler` class.**
    * **Logic:**
        1.  `metadata = self.metadata_handler.read_metadata()`.
        2.  `managed_files = []`.
        3.  Iterate `metadata.keys()`. For each key:
            * Assume key is a relative path string. (Need to filter out non-file-entry keys if `metadata.json` might contain other top-level data).
            * `managed_files.append(self.root_dir / key_as_relative_path)`.
        4.  Return `managed_files`.
        * To be more robust, ensure the `metadata.json` keys used are indeed file entries, e.g., by checking if their corresponding value is a dictionary and contains a known field like `"filepath_relative"`.

**V. File Removal (`remove_tracked_file`)**

* **`RepoHandler.remove_tracked_file(self, file_path: Path, permanent_from_git_and_disk: bool = True, metadata_action: str = "remove_entry") -> bool` (Refined):**
    * `file_path` should be relative to repo root or absolute.
    * `metadata_action` can be `"remove_entry"` or `"mark_removed"`.
    * **Logic:**
        1.  Resolve `file_path` to absolute `abs_file_path` and relative `rel_file_path_str`.
        2.  `git_removal_staged = False`
        3.  If `permanent_from_git_and_disk`:
            * If `abs_file_path.exists()`:
                * `self.git_ops_helper.git_repo.index.remove([str(abs_file_path)], working_tree=True)` (stages removal, deletes from disk).
                * `git_removal_staged = True`
            * Else (file doesn't exist on disk, but maybe Git tracks it):
                * Try `self.git_ops_helper.git_repo.index.remove([str(abs_file_path)], working_tree=False)` to remove from index if tracked.
                * `git_removal_staged = True`
        4.  Update `metadata.json`:
            * Read metadata.
            * If `metadata_action == "remove_entry"` and `rel_file_path_str in metadata`: `del metadata[rel_file_path_str]`.
            * If `metadata_action == "mark_removed"` and `rel_file_path_str in metadata`:
                * `metadata[rel_file_path_str]['application_status'] = 'removed'`
                * `metadata[rel_file_path_str]['last_metadata_update_utc'] = now_utc_iso()`
            * Else (if file not in metadata), log info, skip metadata write.
            * `write_meta_success = self.metadata_handler.write_metadata(metadata, commit_message=None)` (no commit yet).
            * If not `write_meta_success`, log error, potentially return `False` or attempt to revert Git staging if possible (complex).
        5.  Commit:
            * `commit_msg = f"Processed removal of {file_path.name}; Git permanent: {permanent_from_git_and_disk}, Metadata: {metadata_action}"`
            * `files_to_explicitly_add_for_commit = [self.metadata_handler.metadata_path]` (Git commit will pick up staged file removals automatically).
            * `commit_success = self.modifier._commit_changes(files_to_explicitly_add_for_commit, commit_msg)`
        6.  Return `commit_success`.

**VI. Checksum Verification (`verify_file_integrity`)**

* **`RepoHandler.verify_file_integrity(self, file_to_check: Optional[Path] = None) -> Dict[str, str]` (Refined):**
    * **Logic:**
        1.  `metadata = self.metadata_handler.read_metadata()`.
        2.  `verification_results = {}`.
        3.  Determine `files_to_iterate`: If `file_to_check` provided, resolve it to relative path and check that key. Else, iterate `metadata.keys()`.
        4.  For each `relative_path_str` in `files_to_iterate`:
            * `entry = metadata.get(relative_path_str)`.
            * `abs_path = self.root_dir / relative_path_str`.
            * If not `entry` or not `entry.get("custom_hashes")` or not `entry["custom_hashes"].get("sha256")`:
                * `verification_results[relative_path_str] = "NO_STORED_HASH_IN_METADATA"`, continue.
            * `stored_sha256 = entry["custom_hashes"]["sha256"]`.
            * If not `abs_path.is_file()`:
                * `verification_results[relative_path_str] = "FILE_MISSING_IN_WORKING_TREE"`, continue.
            * Use the enhanced `_get_file_metadata`'s hashing capability or call `src.utils.hashing.hash_file` directly to get current SHA256.
                ```python
                # from src.utils.hashing import hash_file
                # current_hashes = hash_file(filepath=str(abs_path), current_size=abs_path.stat().st_size)
                # current_sha256 = current_hashes.get("sha256")
                ```
            * If `current_sha256 == stored_sha256`: `verification_results[relative_path_str] = "OK"`.
            * Else: `verification_results[relative_path_str] = "MISMATCH"`.
        5.  Return `verification_results`.

**VII. Compression Utilities**

* **Refinement:**
    1.  **Location:** Given `src/utils/compression.py` exists (from uploaded file list), these utilities should ideally reside there and be imported. If that file is not suitable or feature-complete for Zstandard, then methods can be added to `RepoHandler` or a `FileUtilHelper`.
    2.  **Integration with `RepoHandler` (if methods are external):**
        * `RepoHandler` can have methods like `RepoHandler.compress_tracked_file(self, relative_path: Path, ...)` and `RepoHandler.decompress_tracked_file(self, relative_path_to_zst: Path, ...)`.
        * These methods would:
            * Call the external compression/decompression utility.
            * Use `add_file_to_tracking` to add the new (compressed/decompressed) file.
            * Update `metadata.json` for both files (original and new) to link them (e.g., `original_of: path/to/file.zst` or `compressed_as: path/to/file.txt`).
            * Optionally use `remove_tracked_file` for the original after compression if requested.
            * Commit all changes (new file, metadata updates, old file removal).

**VIII. Duplicate File Detection**

* **`RepoHandler.find_duplicate_files_in_metadata(self, hash_type: str = 'sha256') -> Dict[str, List[str]]` (Refined):**
    * The previous proposal is largely sound. Ensure `metadata.json` keys are consistently relative paths.
    * **Return Value:** The list of strings should be relative file paths, which can then be resolved by the caller if needed.
    * This method is fully dependent on `custom_hashes` being populated in `metadata.json` by the `add_file_to_tracking` (or equivalent) process.

**IX. Path Handling for Processed Outputs**

* **Refinement:**
    1.  **Configuration Access:** `RepoHandler.__init__` should probably accept an optional `config` object (e.g., the one loaded by `src.utils.config.load_config()`). If not provided, methods needing it can try to load it themselves or operate without it if possible.
        ```python
        # In RepoHandler.__init__
        # self.config = config or load_config() # If load_config() is idempotent
        ```
    2.  **Method Signature:**
        `RepoHandler.determine_output_path(self, source_file_rel_path: Path, processing_step_name: str, new_extension: Optional[str] = None) -> Optional[Path]`
        * `source_file_rel_path`: Relative path of the source file within the repo.
        * `processing_step_name`: e.g., "cleaned", "normalized", "features". Used to create subdirectories in the output location.
        * `config`: Use `self.config` (if initialized) or a passed `config` to get a base output directory (e.g., `self.config['DataProcessingConfig']['output_directory']`).
        * Logic similar to `repo_handlerORIG.py`'s `_determine_processed_path`, creating `output_base_dir / processing_step_name / new_filename`.

**X. Logging Consistency (Refined Example)**

* Within any method of `RepoHandler` or its helpers:
    ```python
    # Example in RepoHandler.some_method(self, ...)
    # LOG_INS_PREFIX is self.LOG_INS_PREFIX, set in __init__
    method_name = inspect.currentframe().f_code.co_name
    _log_ins_val = f"{self.LOG_INS_PREFIX}::{method_name}::{inspect.currentframe().f_lineno}"
    
    log_level_str = "info" # or "debug", "error"
    action_description = "File processed successfully."
    
    # Construct the log statement string as per the specific f"{LOG_INS}:LOGLEVEL>>logstatement" format
    log_message_for_statement = f"{_log_ins_val}:{log_level_str.upper()}>>{action_description}"
    
    actual_log_statement( # Assuming this is the imported or placeholder function
        loglevel=log_level_str,
        logstatement=log_message_for_statement,
        main_logger_name=__file__, # Or a more specific logger name if available/configured
        exc_info=False
    )
    ```
* This ensures the `logstatement` string itself contains the `LOG_INS` and the uppercase `LOGLEVEL` string, matching the user's specified format, while still passing the `loglevel` string as the first argument to `actual_log_statement` for level-based filtering.

**XI. General Considerations (Refined)**

* **Thread Safety for `metadata.json`:**
    * For `RepoHandler` methods that perform a sequence of read `metadata.json` -> modify dict -> write `metadata.json` -> commit, if these high-level methods can be called concurrently by the application, a lock within `RepoHandler` for these compound operations might be necessary to prevent one thread from overwriting changes made by another before commit.
    * Example: `self.metadata_modification_lock = threading.Lock()` in `RepoHandler.__init__`.
        ```python
        # In a RepoHandler method like add_file_to_tracking
        # with self.metadata_modification_lock:
        #     metadata = self.metadata_handler.read_metadata()
        #     # ... modify metadata dict ...
        #     write_ok = self.metadata_handler.write_metadata(metadata, commit_message=None) # Handler's internal lock protects this single write
        #     if write_ok and commit_flag:
        #          self.modifier._commit_changes(...)
        ```
    * The `MetadataFileHandler`'s own lock protects the atomicity of a single `read_metadata` or `write_metadata` call against interleaved file access, but not necessarily logical transactionality across multiple such calls orchestrated by `RepoHandler`.

* **Transactionality:**
    * Git commits are atomic. The goal is to make the changes to `metadata.json` and the corresponding data files appear as a single atomic change in Git history. The proposed methods generally try to achieve this by committing `metadata.json` alongside the data files it describes.
    * For complex operations (e.g., batch adding files), if any part fails (like hashing one file), the strategy should define whether to roll back all metadata changes for that batch or commit partial successes. Current proposals lean towards committing what was successful or logging errors. Explicit rollback logic (e.g., resetting `metadata.json` from HEAD if a subsequent step fails before commit) would add complexity but improve transactionality.

These refined proposals provide a more detailed blueprint. The next logical step would be to start outlining or generating the Python code for these new/modified methods within `src/core/repo_handler.py` and its helper classes, focusing on one functional area at a time.