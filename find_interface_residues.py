#!/usr/bin/env python3
import argparse
import os
import numpy as np

from Bio.PDB import MMCIFParser, NeighborSearch, is_aa
import pydssp   


def get_chain(structure, chain_id, model_index=0):
    model = list(structure)[model_index]
    try:
        return model[chain_id]
    except KeyError:
        raise ValueError(f"Chain {chain_id} not found in model {model_index}.")


def residue_plddt(residue):
    """Mean B-factor = AF pLDDT."""
    vals = [atom.get_bfactor() for atom in residue.get_atoms()]
    return sum(vals) / len(vals) if vals else float("nan")


def build_backbone_coords(chain):
    """
    Build an array of shape (L, 4, 3) in residue order:
    atoms = (N, CA, C, O).
    Returns coords (np.ndarray) and list of residues in same order.
    """
    backbone_names = ("N", "CA", "C", "O")
    residues = [r for r in chain if is_aa(r, standard=False)]
    L = len(residues)
    coords = np.zeros((L, 4, 3), dtype=float)

    for i, res in enumerate(residues):
        for j, atom_name in enumerate(backbone_names):
            atom = res[atom_name] if atom_name in res else None
            if atom is None:
                # If backbone incomplete, you may want to skip or mask
                coords[i, j, :] = np.nan
            else:
                coords[i, j, :] = atom.coord

    return coords, residues


def run_pydssp(chain):
    """
    Run PyDSSP on a single chain and return a dict:
        (chain_id, residue.id) -> 'Loop'/'Alpha'/'Beta'
    """
    coords, residues = build_backbone_coords(chain)

    # PyDSSP expects shape (L, 4, 3) or (B, L, 4, 3); we'll give it (1, L, 4, 3)
    coord_batch = np.expand_dims(coords, axis=0)  # (1, L, 4, 3)

    # out_type='index': 0 loop, 1 alpha-helix, 2 beta-strand
    ss_index = pydssp.assign(coord_batch, out_type="index")[0]  # (L,)

    index_to_label = {0: "Loop", 1: "Alpha", 2: "Beta"}

    ss_dict = {}
    for res, idx in zip(residues, ss_index):
        label = index_to_label.get(int(idx), "Loop")
        key = (res.parent.id, res.id)   # (chain_id, residue.id tuple)
        ss_dict[key] = label

    return ss_dict


def find_contact_pairs_with_ss(
    cif_path, chain1_id, chain2_id, cutoff=3.0, model_index=0
):
    """
    Return list of contact pairs with:
    resA, resB, pLDDT_A, pLDDT_B, ss_A, ss_B
    """
    print('> Loading .cif')
    parser = MMCIFParser(QUIET=True)
    structure_id = os.path.basename(cif_path)
    structure = parser.get_structure(structure_id, cif_path)
    model = list(structure)[model_index]

    chain1 = model[chain1_id]
    chain2 = model[chain2_id]

    # Secondary structure via PyDSSP (per chain)
    print('> Computing Secondary Structure')
    ss_chain1 = run_pydssp(chain1)
    ss_chain2 = run_pydssp(chain2)

    print('> Finding Pairs')
    atoms_chain2 = [atom for atom in chain2.get_atoms()]
    ns = NeighborSearch(atoms_chain2)

    pairs = set()
    
    for resA in chain1:
        if not is_aa(resA, standard=False):
            continue
        for atom in resA:
            neighbors = ns.search(atom.coord, cutoff)
            for atomB in neighbors:
                resB = atomB.get_parent()
                if not is_aa(resB, standard=False):
                    continue
                pairs.add((resA, resB))

    print('> Compiling Info')
    results = []
    for resA, resB in sorted(pairs, key=lambda x: (x[0].id[1], x[1].id[1])):
        chainA = resA.parent.id
        chainB = resB.parent.id

        (_, resseqA, icodeA) = resA.id
        (_, resseqB, icodeB) = resB.id

        resnameA = resA.get_resname()
        resnameB = resB.get_resname()

        plddtA = residue_plddt(resA)
        plddtB = residue_plddt(resB)

        ssA = ss_chain1.get((chainA, resA.id), "Loop")
        ssB = ss_chain2.get((chainB, resB.id), "Loop")

        results.append(
            {
                "file": os.path.basename(cif_path),
                "chainA": chainA,
                "resnameA": resnameA,
                "resnumA": resseqA,
                "icodeA": (icodeA or "").strip(),
                "pLDDT_A": plddtA,
                "ss_A": ssA,
                "chainB": chainB,
                "resnameB": resnameB,
                "resnumB": resseqB,
                "icodeB": (icodeB or "").strip(),
                "pLDDT_B": plddtB,
                "ss_B": ssB,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Residue-residue contacts with pLDDT and PyDSSP secondary structure."
    )
    parser.add_argument("cif", nargs="+", help="Path(s) to .cif file(s)")
    parser.add_argument("--chain1", required=True, help="Chain ID of first protein (e.g. A)")
    parser.add_argument("--chain2", required=True, help="Chain ID of partner protein (e.g. B)")
    parser.add_argument("--cutoff", type=float, default=3.0, help="Distance cutoff in Å (default 3.0)")
    parser.add_argument("--out", type=str, default=None, help="Optional TSV output file")
    parser.add_argument("--model-index", type=int, default=0, help="Model index (default 0)")

    args = parser.parse_args()
    print(f'> Parsing {args.cif}')
    all_rows = []
    for cif_path in args.cif:
        rows = find_contact_pairs_with_ss(
            cif_path,
            args.chain1,
            args.chain2,
            cutoff=args.cutoff,
            model_index=args.model_index,
        )

        print(f"\nFile: {cif_path}")
        if not rows:
            print(f"  No contacts between chain {args.chain1} and {args.chain2} within {args.cutoff} Å.")
        else:
            for r in rows:
                print(
                    f"A {r['resnameA']:>3} {r['resnumA']:>4} "
                    f"({r['ss_A']}, pLDDT {r['pLDDT_A']:.1f})  <->  "
                    f"B {r['resnameB']:>3} {r['resnumB']:>4} "
                    f"({r['ss_B']}, pLDDT {r['pLDDT_B']:.1f})"
                )
        all_rows.extend(rows)

    if args.out and all_rows:
        import csv

        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file",
                    "chainA", "resnameA", "resnumA", "icodeA", "pLDDT_A", "ss_A",
                    "chainB", "resnameB", "resnumB", "icodeB", "pLDDT_B", "ss_B",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} contact pairs to {args.out}")


if __name__ == "__main__":
    main()
