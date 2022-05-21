#!/usr/bin/env python3
"""
Author:         David Meijer
Description:    Script for parsing molecule SMILES for a specific subclass of
                biosynthetic gene clusters (BGCs) from MIBiG.
                Download the MIBiG database from:
                https://mibig.secondarymetabolites.org/download
Usage:          For help use './01_parse_mibig.py -h'.
"""
import typing as ty
import argparse
import os
import json


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
        help="Path to dir containing MIBiG records as JSON."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output file."
    )
    parser.add_argument(
        "-c",
        "--biosyn_class",
        type=str,
        required=True,
        default="polyketide",
        help="Class of BGCs to parse."
    )
    parser.add_argument(
        "-tbd",
        "--tbd",
        action="store_true",
        help="Parse data as TBD."
    )
    return parser.parse_args()


def read_files(dir_path: str) -> ty.List[str]:
    """
    Read all files in a directory.

    Parameters
    ----------
    dir_path (str): Path to directory containing files.

    Returns
    -------
    ty.List[str]: List of paths to files in directory.
    """
    return [
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]


def get_file_name_from_path(file_path: str) -> str:
    """
    Get file name from path.

    Parameters
    ----------
    file_path (str): Path to file.

    Returns
    -------
    str: File name.
    """
    file_name_items = os.path.basename(file_path).split(".")
    if len(file_name_items) == 2:
        name, _ = file_name_items
        return name
    else:
        raise ValueError("File name must contain a single dot.")


def read_json(file_path: str) -> ty.Dict:
    """
    Read JSON file.

    Parameters
    ----------
    file_path (str): Path to JSON file.

    Returns
    -------
    ty.Dict: JSON data.
    """
    if file_path.endswith(".json"):
        with open(file_path, "r") as handle:
            return json.load(handle)
    else:
        raise ValueError("File must be a JSON file.")


class Record:
    def __init__(self, record_id: str, record: ty.Dict) -> None:
        """
        Parameters
        ----------
        record_id (str): MIBiG ID.
        record (ty.Dict): MIBiG record.
        """
        self.id = record_id
        self.biosyn_class = list(map(
            lambda s: s.lower(),
            record["cluster"]["biosyn_class"]
        ))
        self.smiles = []
        for compound in record["cluster"]["compounds"]:
            try:
                self.smiles.append(compound["chem_struct"])
            except KeyError:
                pass

    def to_csv(self) -> ty.List[str]:
        """
        Returns
        -------
        str: CSV line.
        """
        return [
            f"{self.id}_{idx + 1},{smi}\n"
            for idx, smi in enumerate(self.smiles)
        ]
    
    def to_tbd(self) -> ty.List[str]:
        """
        Returns
        -------
        str: TBD line.
        """
        return [
            f"{self.id}_{idx + 1}\t{smi}\n"
            for idx, smi in enumerate(self.smiles)
        ]

    @classmethod
    def csv_header(cls) -> str:
        """
        Returns
        -------
        str: CSV header.
        """
        return "id,smiles\n"

    @classmethod
    def tbd_header(cls) -> str:
        """
        Returns
        -------
        str: TBD header.
        """
        return "id\tsmiles\n"


def parse_mibig_json_record(file_path: str) -> Record:
    """
    Parse MIBiG JSON record.

    Parameters
    ----------
    file_path (str): Path to JSON file.

    Returns
    -------
    Record: Parsed record.
    """
    record = read_json(file_path)
    return Record(get_file_name_from_path(file_path), record)


def main() -> None:
    """
    Driver code.
    """
    args = cli()
    records = [
        parse_mibig_json_record(file_path)
        for file_path in read_files(args.input)
    ]
    records = filter(lambda r: args.biosyn_class in r.biosyn_class, records)
    with open(args.output, "w") as handle:
        if args.tbd:
            handle.write(Record.tbd_header())
            for record in records:
                for line in record.to_tbd():
                    handle.write(line)
        else:
            handle.write(Record.csv_header())
            for record in records:
                for line in record.to_csv():
                    handle.write(line)


if __name__ == "__main__":
    main()
