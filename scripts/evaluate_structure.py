#!/usr/bin/env python3
"""
Evaluate predicted protein structures against reference structures.

Uses:
- TMalign/USalign for TM-score and RMSD (standalone binaries)
- Pure Python implementation for lDDT (backbone)

Usage:
    python evaluate_structure.py --model path/to/model.cif --reference path/to/reference.pdb \
        --problem-id problem_1 --problem-type monomer --output evaluation.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


def parse_pdb_ca_coords(pdb_path: str) -> tuple[np.ndarray, list[str]]:
    """
    Parse CA atom coordinates from PDB file.

    Returns:
        coords: Nx3 array of CA coordinates
        residue_ids: List of residue identifiers (chain_resnum)
    """
    coords = []
    residue_ids = []

    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    chain = line[21]
                    resnum = line[22:26].strip()
                    coords.append([x, y, z])
                    residue_ids.append(f"{chain}_{resnum}")

    return np.array(coords), residue_ids


def parse_cif_ca_coords(cif_path: str) -> tuple[np.ndarray, list[str]]:
    """
    Parse CA atom coordinates from mmCIF file.

    Returns:
        coords: Nx3 array of CA coordinates
        residue_ids: List of residue identifiers (chain_resnum)
    """
    coords = []
    residue_ids = []

    with open(cif_path) as f:
        in_atom_site = False
        col_indices = {}

        for line in f:
            line = line.strip()

            if line.startswith("_atom_site."):
                in_atom_site = True
                col_name = line.split(".")[1].split()[0]
                col_indices[col_name] = len(col_indices)
                continue

            if in_atom_site and (line.startswith("_") or line.startswith("#") or line == ""):
                in_atom_site = False
                continue

            if in_atom_site and line.startswith(("ATOM", "HETATM")):
                parts = line.split()

                # Get column indices
                atom_col = col_indices.get("label_atom_id", col_indices.get("auth_atom_id", 3))
                x_col = col_indices.get("Cartn_x", 10)
                y_col = col_indices.get("Cartn_y", 11)
                z_col = col_indices.get("Cartn_z", 12)
                chain_col = col_indices.get("label_asym_id", col_indices.get("auth_asym_id", 6))
                resnum_col = col_indices.get("label_seq_id", col_indices.get("auth_seq_id", 8))

                if len(parts) > max(atom_col, x_col, y_col, z_col, chain_col, resnum_col):
                    atom_name = parts[atom_col]
                    if atom_name == "CA":
                        x = float(parts[x_col])
                        y = float(parts[y_col])
                        z = float(parts[z_col])
                        chain = parts[chain_col]
                        resnum = parts[resnum_col]
                        coords.append([x, y, z])
                        residue_ids.append(f"{chain}_{resnum}")

    return np.array(coords), residue_ids


def parse_structure_ca(file_path: str, chain: str = None) -> tuple[np.ndarray, list[str]]:
    """Parse CA coordinates from PDB or CIF file, optionally filtering by chain."""
    if file_path.endswith(".cif"):
        coords, res_ids = parse_cif_ca_coords(file_path)
    else:
        coords, res_ids = parse_pdb_ca_coords(file_path)

    if chain is not None and len(coords) > 0:
        # Filter by chain
        mask = [rid.startswith(f"{chain}_") for rid in res_ids]
        coords = coords[mask]
        res_ids = [r for r, m in zip(res_ids, mask) if m]

    return coords, res_ids


def extract_chain_to_pdb(input_path: str, output_path: str, chain: str) -> bool:
    """Extract a specific chain from PDB/CIF to a new PDB file."""
    try:
        if input_path.endswith(".cif"):
            # Parse CIF and write only the specified chain
            atoms = []
            with open(input_path) as f:
                in_atom_site = False
                col_indices = {}

                for line in f:
                    line_stripped = line.strip()

                    if line_stripped.startswith("_atom_site."):
                        in_atom_site = True
                        col_name = line_stripped.split(".")[1].split()[0]
                        col_indices[col_name] = len(col_indices)
                        continue

                    if in_atom_site and (line_stripped.startswith("_") or
                                          line_stripped.startswith("#") or
                                          line_stripped == "" or
                                          line_stripped.startswith("loop_")):
                        if atoms:
                            break
                        in_atom_site = False
                        continue

                    if in_atom_site and line_stripped.startswith(("ATOM", "HETATM")):
                        parts = line_stripped.split()
                        chain_col = col_indices.get("label_asym_id", 6)
                        if len(parts) > chain_col and parts[chain_col] == chain:
                            atoms.append(parts)

            if not atoms:
                return False

            # Write PDB
            with open(output_path, "w") as f:
                for i, parts in enumerate(atoms):
                    record = parts[0]
                    atom_id = i + 1
                    atom_name = parts[col_indices.get("label_atom_id", 3)]
                    res_name = parts[col_indices.get("label_comp_id", 5)]
                    res_num = parts[col_indices.get("label_seq_id", 8)]
                    x = float(parts[col_indices.get("Cartn_x", 10)])
                    y = float(parts[col_indices.get("Cartn_y", 11)])
                    z = float(parts[col_indices.get("Cartn_z", 12)])
                    element = parts[col_indices.get("type_symbol", 2)] if "type_symbol" in col_indices else atom_name[0]

                    if len(atom_name) < 4:
                        atom_name_fmt = f" {atom_name:<3}"
                    else:
                        atom_name_fmt = atom_name[:4]

                    f.write(f"{record:<6}{atom_id:>5} {atom_name_fmt}{res_name:>3} {chain[0]}{int(res_num):>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}\n")
                f.write("END\n")
            return True
        else:
            # Parse PDB and filter by chain
            with open(input_path) as f_in, open(output_path, "w") as f_out:
                for line in f_in:
                    if line.startswith(("ATOM", "HETATM")):
                        if line[21] == chain:
                            f_out.write(line)
                    elif line.startswith("END"):
                        f_out.write(line)
            return True
    except Exception as e:
        print(f"Chain extraction failed: {e}", file=sys.stderr)
        return False


def cif_to_pdb(cif_path: str, pdb_path: str) -> bool:
    """
    Convert CIF to PDB format for TMalign compatibility.
    Simple conversion focusing on ATOM records.
    """
    try:
        coords, _ = parse_cif_ca_coords(cif_path)

        # Re-read to get all atoms
        with open(cif_path) as f:
            content = f.read()

        # Try using gemmi if available
        try:
            import gemmi
            structure = gemmi.read_structure(cif_path)
            structure.write_pdb(pdb_path)
            return True
        except ImportError:
            pass

        # Fallback: manual conversion
        atoms = []
        with open(cif_path) as f:
            in_atom_site = False
            col_indices = {}

            for line in f:
                line_stripped = line.strip()

                if line_stripped.startswith("_atom_site."):
                    in_atom_site = True
                    col_name = line_stripped.split(".")[1].split()[0]
                    col_indices[col_name] = len(col_indices)
                    continue

                if in_atom_site and (line_stripped.startswith("_") or line_stripped.startswith("#") or line_stripped == "" or line_stripped.startswith("loop_")):
                    if atoms:  # We've collected atoms, stop
                        break
                    in_atom_site = False
                    continue

                if in_atom_site and line_stripped.startswith(("ATOM", "HETATM")):
                    parts = line_stripped.split()
                    atoms.append(parts)

        if not atoms or not col_indices:
            return False

        # Write PDB
        with open(pdb_path, "w") as f:
            for i, parts in enumerate(atoms):
                record = parts[0]
                atom_id = i + 1
                atom_name = parts[col_indices.get("label_atom_id", 3)]
                res_name = parts[col_indices.get("label_comp_id", 5)]
                chain = parts[col_indices.get("label_asym_id", 6)]
                res_num = parts[col_indices.get("label_seq_id", 8)]
                x = float(parts[col_indices.get("Cartn_x", 10)])
                y = float(parts[col_indices.get("Cartn_y", 11)])
                z = float(parts[col_indices.get("Cartn_z", 12)])
                element = parts[col_indices.get("type_symbol", 2)] if "type_symbol" in col_indices else atom_name[0]

                # Format atom name (4 chars, right-justified for 1-char elements)
                if len(atom_name) < 4:
                    atom_name_fmt = f" {atom_name:<3}"
                else:
                    atom_name_fmt = atom_name[:4]

                f.write(f"{record:<6}{atom_id:>5} {atom_name_fmt}{res_name:>3} {chain[0]}{int(res_num):>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}\n")
            f.write("END\n")

        return True
    except Exception as e:
        print(f"CIF to PDB conversion failed: {e}", file=sys.stderr)
        return False


def run_tmalign(model_path: str, reference_path: str) -> dict:
    """
    Run TMalign to compute TM-score and RMSD.

    Returns dict with tm_score, rmsd, aligned_length, seq_identity
    """
    result = {
        "tm_score": None,
        "tm_score_ref": None,
        "rmsd": None,
        "aligned_length": None,
        "seq_identity": None
    }

    # Convert CIF to PDB if needed
    temp_files = []

    try:
        if model_path.endswith(".cif"):
            model_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
            temp_files.append(model_pdb.name)
            model_pdb.close()
            if not cif_to_pdb(model_path, model_pdb.name):
                return {"error": "Failed to convert model CIF to PDB"}
            model_path = model_pdb.name

        if reference_path.endswith(".cif"):
            ref_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
            temp_files.append(ref_pdb.name)
            ref_pdb.close()
            if not cif_to_pdb(reference_path, ref_pdb.name):
                return {"error": "Failed to convert reference CIF to PDB"}
            reference_path = ref_pdb.name

        # Try USalign first (newer), fall back to TMalign
        for cmd in ["/applic/bin/USalign", "/applic/bin/TMalign", "USalign", "TMalign"]:
            try:
                proc = subprocess.run(
                    [cmd, model_path, reference_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if proc.returncode == 0:
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        else:
            return {"error": "TMalign/USalign not found or failed"}

        # Parse output
        for line in proc.stdout.split("\n"):
            line = line.strip()

            if line.startswith("Aligned length="):
                # Aligned length= 123, RMSD= 1.23, Seq_ID=n_identical/n_aligned= 0.456
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    if part.startswith("Aligned length="):
                        result["aligned_length"] = int(part.split("=")[1].strip())
                    elif part.startswith("RMSD="):
                        result["rmsd"] = float(part.split("=")[1].strip())
                    elif "Seq_ID" in part and "=" in part:
                        # Get the last number after =
                        val = part.split("=")[-1].strip()
                        try:
                            result["seq_identity"] = float(val)
                        except ValueError:
                            pass

            elif line.startswith("TM-score="):
                # TM-score= 0.12345 (normalized by length of Chain_1)
                parts = line.split()
                if len(parts) >= 2:
                    tm = float(parts[1])
                    if "Chain_1" in line or "first" in line.lower():
                        result["tm_score"] = tm
                    elif "Chain_2" in line or "second" in line.lower():
                        result["tm_score_ref"] = tm
                    elif result["tm_score"] is None:
                        result["tm_score"] = tm

        return result

    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


def compute_lddt(model_coords: np.ndarray, ref_coords: np.ndarray,
                 cutoff: float = 15.0, thresholds: tuple = (0.5, 1.0, 2.0, 4.0)) -> float:
    """
    Compute lDDT (Local Distance Difference Test) score.

    This is a backbone-only lDDT using CA atoms.

    Args:
        model_coords: Nx3 array of model CA coordinates
        ref_coords: Nx3 array of reference CA coordinates (must be same length)
        cutoff: Distance cutoff for considering residue pairs (default 15A)
        thresholds: Distance difference thresholds (default 0.5, 1, 2, 4 A)

    Returns:
        lDDT score between 0 and 1
    """
    if len(model_coords) != len(ref_coords):
        # Try to align by length - take minimum
        min_len = min(len(model_coords), len(ref_coords))
        model_coords = model_coords[:min_len]
        ref_coords = ref_coords[:min_len]

    n_residues = len(ref_coords)
    if n_residues < 2:
        return 0.0

    # Compute distance matrices
    def pairwise_distances(coords):
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    ref_dists = pairwise_distances(ref_coords)
    model_dists = pairwise_distances(model_coords)

    # Find pairs within cutoff in reference (excluding self and immediate neighbors)
    mask = (ref_dists < cutoff) & (ref_dists > 0)
    # Exclude immediate sequence neighbors (i, i+1)
    for i in range(n_residues - 1):
        mask[i, i + 1] = False
        mask[i + 1, i] = False

    if not np.any(mask):
        return 0.0

    # Compute distance differences for valid pairs
    ref_valid = ref_dists[mask]
    model_valid = model_dists[mask]

    diff = np.abs(model_valid - ref_valid)

    # Count preserved distances for each threshold
    preserved = sum(np.mean(diff < t) for t in thresholds) / len(thresholds)

    return float(preserved)


def compute_lddt_per_residue(model_coords: np.ndarray, ref_coords: np.ndarray,
                              cutoff: float = 15.0, thresholds: tuple = (0.5, 1.0, 2.0, 4.0)) -> np.ndarray:
    """
    Compute per-residue lDDT scores.

    Returns array of per-residue lDDT scores.
    """
    if len(model_coords) != len(ref_coords):
        min_len = min(len(model_coords), len(ref_coords))
        model_coords = model_coords[:min_len]
        ref_coords = ref_coords[:min_len]

    n_residues = len(ref_coords)
    if n_residues < 2:
        return np.zeros(n_residues)

    # Compute distance matrices
    def pairwise_distances(coords):
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    ref_dists = pairwise_distances(ref_coords)
    model_dists = pairwise_distances(model_coords)

    per_residue_lddt = np.zeros(n_residues)

    for i in range(n_residues):
        # Find neighbors within cutoff (excluding self and immediate neighbor)
        neighbors = (ref_dists[i] < cutoff) & (ref_dists[i] > 0)
        if i > 0:
            neighbors[i - 1] = False
        if i < n_residues - 1:
            neighbors[i + 1] = False

        if not np.any(neighbors):
            per_residue_lddt[i] = 0.0
            continue

        ref_neighbor_dists = ref_dists[i, neighbors]
        model_neighbor_dists = model_dists[i, neighbors]

        diff = np.abs(model_neighbor_dists - ref_neighbor_dists)
        preserved = sum(np.mean(diff < t) for t in thresholds) / len(thresholds)
        per_residue_lddt[i] = preserved

    return per_residue_lddt


def get_af3_metrics(result_dir: str, problem_id: str, participant_id: str) -> dict:
    """
    Extract AF3 confidence metrics from summary_confidences.json.
    """
    patterns = [
        f"{participant_id}_{problem_id}_summary_confidences.json",
        f"*_{problem_id}_summary_confidences.json",
        f"*{problem_id}_summary_confidences.json"
    ]

    for pattern in patterns:
        for f in Path(result_dir).glob(pattern):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    return {
                        "ptm": data.get("ptm"),
                        "iptm": data.get("iptm"),
                        "ranking_score": data.get("ranking_score"),
                        "chain_pair_iptm": data.get("chain_pair_iptm"),
                        "fraction_disordered": data.get("fraction_disordered")
                    }
            except Exception:
                continue

    return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate protein structure predictions")
    parser.add_argument("--model", required=True, help="Path to predicted model (CIF/PDB)")
    parser.add_argument("--reference", required=True, help="Path to reference structure (PDB)")
    parser.add_argument("--problem-id", required=True, help="Problem ID (e.g., problem_1)")
    parser.add_argument("--problem-type", default="monomer", choices=["monomer", "binder"],
                        help="Problem type")
    parser.add_argument("--participant-id", required=True, help="Participant ID")
    parser.add_argument("--token", required=True, help="Result token")
    parser.add_argument("--result-dir", help="Directory containing AF3 result files")
    parser.add_argument("--output", required=True, help="Output JSON path")

    args = parser.parse_args()

    # Initialize result
    result = {
        "problem_id": args.problem_id,
        "problem_type": args.problem_type,
        "participant_id": args.participant_id,
        "token": args.token,
        "model_file": os.path.basename(args.model),
        "reference_file": os.path.basename(args.reference),
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metrics": {},
        "af3_metrics": {}
    }

    # Get AF3 metrics
    if args.result_dir:
        result["af3_metrics"] = get_af3_metrics(
            args.result_dir, args.problem_id, args.participant_id
        )

    # Run TMalign for TM-score and RMSD
    print(f"Running TMalign on {args.model} vs {args.reference}...")
    tm_result = run_tmalign(args.model, args.reference)

    if "error" not in tm_result:
        result["metrics"]["tm_score"] = tm_result.get("tm_score")
        result["metrics"]["tm_score_ref"] = tm_result.get("tm_score_ref")
        result["metrics"]["rmsd"] = tm_result.get("rmsd")
        result["metrics"]["aligned_length"] = tm_result.get("aligned_length")
        result["metrics"]["seq_identity"] = tm_result.get("seq_identity")
    else:
        print(f"TMalign error: {tm_result['error']}", file=sys.stderr)
        result["metrics"]["tm_error"] = tm_result["error"]

    # Compute lDDT
    print("Computing backbone lDDT...")
    try:
        model_coords, model_res = parse_structure_ca(args.model)
        ref_coords, ref_res = parse_structure_ca(args.reference)

        if len(model_coords) > 0 and len(ref_coords) > 0:
            lddt_score = compute_lddt(model_coords, ref_coords)
            result["metrics"]["bb_lddt"] = round(lddt_score, 4)
            result["metrics"]["model_ca_count"] = len(model_coords)
            result["metrics"]["ref_ca_count"] = len(ref_coords)
            print(f"  bb-lDDT: {lddt_score:.4f} (model: {len(model_coords)} CA, ref: {len(ref_coords)} CA)")
        else:
            result["metrics"]["lddt_error"] = "Could not extract CA coordinates"
    except Exception as e:
        print(f"lDDT error: {e}", file=sys.stderr)
        result["metrics"]["lddt_error"] = str(e)

    # For binder problems, also compute binder-only metrics (chain A vs chain A)
    if args.problem_type == "binder":
        print("\nComputing binder-only metrics (chain A vs chain A)...")
        result["binder_metrics"] = {}

        try:
            # Extract chain A from model and reference
            model_chain_a = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
            ref_chain_a = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
            model_chain_a.close()
            ref_chain_a.close()

            model_extracted = extract_chain_to_pdb(args.model, model_chain_a.name, "A")
            ref_extracted = extract_chain_to_pdb(args.reference, ref_chain_a.name, "A")

            if model_extracted and ref_extracted:
                # TMalign on binder chain only
                binder_tm = run_tmalign(model_chain_a.name, ref_chain_a.name)
                if "error" not in binder_tm:
                    result["binder_metrics"]["binder_tm_score"] = binder_tm.get("tm_score")
                    result["binder_metrics"]["binder_rmsd"] = binder_tm.get("rmsd")
                    result["binder_metrics"]["binder_aligned_length"] = binder_tm.get("aligned_length")
                    print(f"  Binder TM-score: {binder_tm.get('tm_score')}")
                    print(f"  Binder RMSD: {binder_tm.get('rmsd')}")

                # lDDT on binder chain only
                model_a_coords, _ = parse_structure_ca(args.model, chain="A")
                ref_a_coords, _ = parse_structure_ca(args.reference, chain="A")

                if len(model_a_coords) > 0 and len(ref_a_coords) > 0:
                    binder_lddt = compute_lddt(model_a_coords, ref_a_coords)
                    result["binder_metrics"]["binder_lddt"] = round(binder_lddt, 4)
                    result["binder_metrics"]["binder_model_ca"] = len(model_a_coords)
                    result["binder_metrics"]["binder_ref_ca"] = len(ref_a_coords)
                    print(f"  Binder lDDT: {binder_lddt:.4f}")
            else:
                result["binder_metrics"]["error"] = "Could not extract chain A"
                print("  Warning: Could not extract chain A for binder-only evaluation")

            # Cleanup temp files
            for f in [model_chain_a.name, ref_chain_a.name]:
                if os.path.exists(f):
                    os.unlink(f)

        except Exception as e:
            print(f"Binder-only evaluation error: {e}", file=sys.stderr)
            result["binder_metrics"]["error"] = str(e)

    # Determine primary score based on problem type
    metrics = result.get("metrics", {})
    af3 = result.get("af3_metrics", {})

    if args.problem_type == "binder":
        # For binders, use ipTM from AF3 as primary
        chain_iptm = af3.get("chain_pair_iptm")
        if chain_iptm and isinstance(chain_iptm, list) and len(chain_iptm) > 0:
            if isinstance(chain_iptm[0], list) and len(chain_iptm[0]) > 0:
                result["primary_score"] = chain_iptm[0][0]
                result["primary_metric"] = "iptm"
            else:
                result["primary_score"] = af3.get("ranking_score")
                result["primary_metric"] = "ranking_score"
        else:
            result["primary_score"] = af3.get("ranking_score")
            result["primary_metric"] = "ranking_score"
    else:
        # For monomers, use bb-lDDT as primary if available
        if metrics.get("bb_lddt") is not None:
            result["primary_score"] = metrics["bb_lddt"]
            result["primary_metric"] = "bb_lddt"
        elif metrics.get("tm_score") is not None:
            result["primary_score"] = metrics["tm_score"]
            result["primary_metric"] = "tm_score"
        else:
            result["primary_score"] = af3.get("ptm")
            result["primary_metric"] = "ptm"

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nEvaluation saved to {args.output}")
    print(f"Primary metric: {result.get('primary_metric')} = {result.get('primary_score')}")

    # Print summary
    print("\n=== Metrics Summary ===")
    for k, v in result["metrics"].items():
        if not k.endswith("_error") and not k.endswith("_count"):
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
