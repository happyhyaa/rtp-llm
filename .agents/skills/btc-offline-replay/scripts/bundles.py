"""Store native 100k replay chunks in larger sequential tar files."""
import shutil
import tarfile
from pathlib import Path


def bundle(directory, chunks, per_file):
    for start in range(0, len(chunks), per_file):
        group = chunks[start:start + per_file]
        name = f"bundle-{start // per_file:06}.tar"
        with tarfile.open(directory / name, "w") as archive:
            for row in group:
                archive.add(directory / row["file"], arcname=row["file"], recursive=False)
        for row in group:
            row["bundle"] = name
            (directory / row["file"]).unlink()


def unpack(directory, chunks, dest):
    from itertools import groupby
    for name, group in groupby(chunks, key=lambda row: row["bundle"]):
        if Path(name).name != name:
            raise ValueError("invalid bundle name")
        with tarfile.open(directory / name, "r|") as archive:
            for row in group:
                member = archive.next()
                if member is None or member.name != row["file"] or not member.isfile():
                    raise ValueError("unexpected member in " + name)
                with archive.extractfile(member) as source, (dest / row["file"]).open("xb") as target:
                    shutil.copyfileobj(source, target, 1 << 20)
