#!/usr/bin/env python3
"""
Package AF3 results with a secret token for private access.

This script handles multi-problem submissions:
1. Finds AF3 output files in each problem subdirectory
2. Generates a unique secret token for the entire submission
3. Copies relevant files to public_results/<token>/
4. Renames files to include problem ID for clarity
5. Updates status.json with the token
"""

import argparse
import json
import secrets
import shutil
import sys
from pathlib import Path


def generate_token() -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(24)


def find_af3_outputs(problem_dir: Path) -> dict:
    """Find AF3 output files in a problem directory."""
    outputs = {}

    # Look for main output files
    for f in problem_dir.glob("*_model.cif"):
        outputs["model_cif"] = f
        break

    for f in problem_dir.glob("*_confidences.json"):
        outputs["confidences"] = f
        break

    for f in problem_dir.glob("*_summary_confidences.json"):
        outputs["summary"] = f
        break

    for f in problem_dir.glob("*_ranking_scores.csv"):
        outputs["ranking"] = f
        break

    # Also check one level deeper (AF3 sometimes nests outputs)
    if "model_cif" not in outputs:
        for f in problem_dir.glob("*/*_model.cif"):
            outputs["model_cif"] = f
            # Also get other files from same dir
            parent = f.parent
            for conf in parent.glob("*_confidences.json"):
                outputs["confidences"] = conf
                break
            for summ in parent.glob("*_summary_confidences.json"):
                outputs["summary"] = summ
                break
            for rank in parent.glob("*_ranking_scores.csv"):
                outputs["ranking"] = rank
                break
            break

    return outputs


def package_multi_results(submission_dir: Path, output_dir: Path, status_file: Path, incremental: bool = True):
    """
    Package results from multi-problem submission.

    Args:
        submission_dir: Directory containing problem subdirectories
        output_dir: Base directory for packaged results
        status_file: Path to status.json
        incremental: If True, reuse existing token and add new results

    Returns the token and list of newly packaged problems if successful, (None, []) otherwise.
    """
    # Check if this is a multi-problem submission
    problem_dirs = sorted(submission_dir.glob("problem_*"))

    if not problem_dirs:
        # Fall back to single-problem behavior for backward compatibility
        token = package_single_result(submission_dir, output_dir, status_file)
        return token, ["single"] if token else []

    # Check which problems are complete
    completed_problems = {}
    for problem_dir in problem_dirs:
        if not problem_dir.is_dir():
            continue
        outputs = find_af3_outputs(problem_dir)
        if "model_cif" in outputs:
            completed_problems[problem_dir.name] = outputs

    if not completed_problems:
        print(f"No completed problems found in {submission_dir}", file=sys.stderr)
        return None, []

    # Read main submission.json
    main_submission = submission_dir / "submission.json"
    if main_submission.exists():
        with open(main_submission) as f:
            submission_data = json.load(f)
    else:
        submission_data = {}

    participant_id = submission_data.get("participant_id", "unknown")
    total_problems = len(submission_data.get("sequences", {}))

    # Check if all problems are complete
    all_complete = len(completed_problems) >= total_problems

    # Check for existing token (incremental mode)
    existing_token = None
    already_packaged = set()
    if status_file.exists():
        with open(status_file) as f:
            existing_status = json.load(f)
        existing_token = existing_status.get("result_token")
        already_packaged = set(existing_status.get("packaged_problems", []))

    # Determine which problems are newly completed
    newly_completed = {k: v for k, v in completed_problems.items() if k not in already_packaged}

    if not newly_completed and existing_token:
        print(f"No new problems to package (already packaged: {already_packaged})")
        return existing_token, []

    # Use existing token or generate new one
    if incremental and existing_token:
        token = existing_token
        token_dir = output_dir / token
    else:
        token = generate_token()
        token_dir = output_dir / token
        newly_completed = completed_problems  # Package all if not incremental

    token_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    problem_results = {}
    newly_packaged = []

    # Copy files from newly completed problems only
    for problem_id, outputs in newly_completed.items():
        problem_results[problem_id] = {"files": []}
        newly_packaged.append(problem_id)

        for file_type, filepath in outputs.items():
            if filepath and filepath.exists():
                # Rename file to include participant and problem ID
                # e.g., team_alpha_problem_1_model.cif
                suffix = filepath.suffix
                stem = filepath.stem

                # Extract the file type (model, confidences, etc.)
                if "_model" in stem:
                    new_name = f"{participant_id}_{problem_id}_model{suffix}"
                elif "_summary_confidences" in stem:
                    new_name = f"{participant_id}_{problem_id}_summary_confidences{suffix}"
                elif "_confidences" in stem:
                    new_name = f"{participant_id}_{problem_id}_confidences{suffix}"
                elif "_ranking_scores" in stem:
                    new_name = f"{participant_id}_{problem_id}_ranking_scores{suffix}"
                else:
                    new_name = f"{participant_id}_{problem_id}_{filepath.name}"

                dest = token_dir / new_name
                shutil.copy2(filepath, dest)
                copied_files.append(new_name)
                problem_results[problem_id]["files"].append(new_name)
                print(f"Copied {filepath.name} -> {new_name}")

    # Copy main submission.json
    if main_submission.exists():
        shutil.copy2(main_submission, token_dir / "submission.json")
        copied_files.append("submission.json")

    # Load existing metadata if incremental
    existing_metadata = {}
    metadata_file = token_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            existing_metadata = json.load(f)

    # Merge with existing files and problems
    all_files = list(set(existing_metadata.get("files", []) + copied_files))
    all_problem_results = existing_metadata.get("problems", {})
    all_problem_results.update(problem_results)

    # Create/update metadata file
    metadata = {
        "token": token,
        "participant_id": participant_id,
        "total_problems": total_problems,
        "completed_problems": len(completed_problems),
        "all_complete": all_complete,
        "files": all_files,
        "problems": all_problem_results,
        "source_dir": str(submission_dir),
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Update main status.json
    if status_file.exists():
        with open(status_file) as f:
            status = json.load(f)
    else:
        status = {}

    status["status"] = "completed" if all_complete else "partial"
    status["result_token"] = token
    status["completed_problems"] = len(completed_problems)
    status["total_problems"] = total_problems
    status["packaged_problems"] = list(already_packaged | set(newly_packaged))

    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)

    print(f"Results packaged to {token_dir}")
    print(f"Completed: {len(completed_problems)}/{total_problems} problems")
    print(f"Newly packaged: {newly_packaged}")
    return token, newly_packaged


def package_single_result(submission_dir: Path, output_dir: Path, status_file: Path):
    """
    Package results from single-problem submission (backward compatibility).
    """
    outputs = find_af3_outputs(submission_dir)

    if "model_cif" not in outputs:
        print(f"No model.cif found in {submission_dir}", file=sys.stderr)
        return None

    # Generate token and create output directory
    token = generate_token()
    token_dir = output_dir / token
    token_dir.mkdir(parents=True, exist_ok=True)

    # Copy output files
    copied = []
    for name, filepath in outputs.items():
        if filepath and filepath.exists():
            dest = token_dir / filepath.name
            shutil.copy2(filepath, dest)
            copied.append(filepath.name)
            print(f"Copied {filepath.name}")

    # Copy submission.json for metadata
    submission_json = submission_dir / "submission.json"
    if submission_json.exists():
        shutil.copy2(submission_json, token_dir / "submission.json")
        copied.append("submission.json")

    # Create metadata file
    metadata = {
        "token": token,
        "files": copied,
        "source_dir": str(submission_dir),
    }
    with open(token_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Update status.json
    if status_file.exists():
        with open(status_file) as f:
            status = json.load(f)
    else:
        status = {}

    status["status"] = "completed"
    status["result_token"] = token

    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)

    print(f"Results packaged to {token_dir}")
    return token


def main():
    parser = argparse.ArgumentParser(description="Package AF3 results with secret token")
    parser.add_argument("--submission-dir", required=True, type=Path,
                        help="Directory containing AF3 outputs")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Base directory for packaged results")
    parser.add_argument("--status-file", required=True, type=Path,
                        help="Path to status.json file")
    parser.add_argument("--incremental", action="store_true", default=True,
                        help="Reuse existing token and add new results (default: True)")

    args = parser.parse_args()

    token, newly_packaged = package_multi_results(
        args.submission_dir, args.output_dir, args.status_file, args.incremental
    )

    if token:
        print(f"Token: {token}")
        print(f"Newly packaged: {','.join(newly_packaged) if newly_packaged else 'none'}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
