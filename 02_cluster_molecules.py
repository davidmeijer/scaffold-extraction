#!/usr/bin/env python3
"""
Author:         David Meijer
Description:    Cluster molecules based on their molecular fingerprint Jaccard
                index.
Usage:          For help use './01_parse_mibig.py -h'.
"""
import typing as ty
import argparse

import dill
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from pvclust import PvClust


def cli() -> argparse.Namespace:
    """
    Command line interface for script.

    Returns
    -------
    argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to csv file containing molecules with IDs in first column "
             "and molecule SMILES in second column."
    )
    parser.add_argument(
        "-oraw",
        "--output_raw",
        type=str,
        required=True,
        help="Path to output file containing raw data."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output dill file containing clustering results."
    )
    return parser.parse_args()


def parse_input_file(input_file: str) -> ty.List[ty.Tuple[str, str]]:
    """
    Parse input file.

    Arguments
    ---------
    input_file (str): path to input file.

    Returns
    -------
    ty.List[ty.Tuple[str, str]]: list of tuples containing molecule IDs and
        SMILES.
    """
    with open(input_file, "r") as handle:
        handle.readline()  # Header.
        return [
            (line.split(",")[0], line.split(",")[1])
            for line in handle.readlines()
        ]


def mol_to_fingerprint(mol: Chem.Mol, n: int) -> np.array:
    """
    Convert a molecule to a fingerprint.

    Arguments
    ---------
    mol (Chem.Mol): molecule to convert.
    n (int): number of features.

    Returns
    -------
    np.array: fingerprint of shape (n,).
    """
    fingerprint = np.zeros((0,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=n
        ),
        fingerprint
    )
    return fingerprint


def main() -> None:
    """
    Driver code.
    """
    args = cli()

    # Parse molecules from input file.
    labels, smiles = zip(*parse_input_file(args.input))

    # Convert SMILES to RDKit molecules.
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    items = zip(labels, mols)
    filtered_items = filter(lambda x: x[1] is not None, items)
    labels, mols = zip(*filtered_items)
    print(f"Molecules parsed from input file: {len(mols)} / {len(smiles)}")

    # Save raw data.
    with open(args.output_raw, "wb") as handle:
        dill.dump(list(zip(labels, mols)), handle)

    # Convert molecules to fingerprints.
    fingerprints = np.array([mol_to_fingerprint(mol, 512) for mol in mols])
    fingerprints = np.transpose(fingerprints)
    print(f"Featurized molecules with shape {fingerprints.shape}")
    fingerprints = pd.DataFrame(fingerprints, columns=labels)
    print(fingerprints)

    # Cluster molecules.
    pv = PvClust(fingerprints, method="ward", metric="euclidean", nboot=1000)
    print(pv.result)

    # Save clustering results.
    with open(args.output, "wb") as handle:
        dill.dump(pv, handle)


if __name__ == "__main__":
    main()
