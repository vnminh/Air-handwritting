import argparse
from pathlib import Path

from airwriting.manifest import create_label_disjoint_manifest, create_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", choices=("source", "label"), default="source",
        help="split repeated sources or complete lexical labels",
    )
    args = parser.parse_args()
    root = Path("Vni_air_writing/VNI_airwriting")
    if args.protocol == "label":
        output = Path("manifests/label_disjoint_seed_20260908.json")
        manifest = create_label_disjoint_manifest(root, output)
    else:
        output = Path("manifests/split_seed_20260908.json")
        manifest = create_manifest(root, output)
    print(manifest["statistics"])
