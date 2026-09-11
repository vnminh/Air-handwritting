from pathlib import Path

from airwriting.manifest import create_manifest


if __name__ == "__main__":
    root = Path("Vni_air_writing/VNI_airwriting")
    output = Path("manifests/split_seed_20260908.json")
    manifest = create_manifest(root, output)
    print(manifest["statistics"])
