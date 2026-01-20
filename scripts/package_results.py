#!/usr/bin/env python3
"""
Package AF3 results with a secret token for private access.

This script:
1. Finds AF3 output files in the submission directory
2. Generates a unique secret token
3. Copies relevant files to public_results/<token>/
4. Updates status.json with the token
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


def find_af3_outputs(submission_dir: Path) -> dict:
    """Find AF3 output files."""
    outputs = {}

    # Look for main output files
    for f in submission_dir.glob("*_model.cif"):
        outputs["model_cif"] = f
        break

    for f in submission_dir.glob("*_confidences.json"):
        outputs["confidences"] = f
        break

    for f in submission_dir.glob("*_summary_confidences.json"):
        outputs["summary"] = f
        break

    for f in submission_dir.glob("*_ranking_scores.csv"):
        outputs["ranking"] = f
        break

    # Also check one level deeper (AF3 sometimes nests outputs)
    if "model_cif" not in outputs:
        for f in submission_dir.glob("*/*_model.cif"):
            outputs["model_cif"] = f
            break

    return outputs


def package_results(submission_dir: Path, output_dir: Path, status_file: Path) -> str | None:
    """
    Package results with a secret token.

    Returns the token if successful, None otherwise.
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

    args = parser.parse_args()

    token = package_results(args.submission_dir, args.output_dir, args.status_file)

    if token:
        print(f"Token: {token}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
