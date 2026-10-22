#!/usr/bin/env python3
"""Diagnose a zip: print the header offset of every member whose CRC/data fails.

Usage: python3 zip_diag.py <archive.zip>   -> prints `<offset> <name>` per bad
member; prints nothing and exits 0 when the archive is fully valid.
"""
import sys
import zipfile

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: zip_diag.py <zip>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    try:
        zf = zipfile.ZipFile(path)
    except Exception as exc:
        print(f"0 <unopenable: {exc}>")
        return 0
    bad = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        try:
            with zf.open(info) as fh:
                while True:
                    if not fh.read(1 << 20):
                        break
        except Exception:
            print(f"{info.header_offset} {info.filename}")
            bad += 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
