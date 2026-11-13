# SPDX-License-Identifier: GPL-3.0-or-later
#
# Builds the release files of UV Shell Gap Overlay into dist/:
#   uv_shell_gap_overlay-<version>.zip   Blender 4.2+ extension (manifest + __init__.py)
#   uv_shell_gap_overlay.py              Blender 3.6+ legacy add-on (the source file as it is)
#   SHA256SUMS.txt                       checksums of both, in `sha256sum` format
#
# Usage (from anywhere, Python 3.10+, nothing to install):
#   python packaging/build_release.py                 build and check
#   python packaging/build_release.py --expect 1.5.0  also check that this is the version built
#   python packaging/build_release.py --blender "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"
#                                                     also let Blender (4.2+) validate the zip
#
# The zip is reproducible: the same two files give the same bytes on any machine (its entries
# carry a fixed date, 1980-01-01, unless SOURCE_DATE_EPOCH says otherwise), so anyone can
# check the published checksums by building again.
#
# Authors: Iurii Kruglov & Claude (Anthropic)

import argparse
import ast
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOURCE = os.path.join(ROOT, "uv_shell_gap_overlay.py")
MANIFEST = os.path.join(HERE, "blender_manifest.toml")
README = os.path.join(ROOT, "README.md")
DIST = os.path.join(ROOT, "dist")


def fail(msg):
    sys.exit("build_release: ERROR: " + msg)


def bl_info_of(source_text):
    """bl_info of the add-on, read without running it."""
    tree = ast.parse(source_text)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "bl_info" for t in node.targets):
            return ast.literal_eval(node.value)
    fail("no bl_info in %s" % SOURCE)


def manifest_field(text, key):
    m = re.search(r'^%s\s*=\s*"([^"]*)"' % re.escape(key), text, re.M)
    return m.group(1) if m else None


def build_date():
    """Entry date for the zip: SOURCE_DATE_EPOCH if set, else 1980-01-01 (the first date a zip
    can hold), so that the bytes depend on the files only."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if not epoch:
        return (1980, 1, 1, 0, 0, 0)
    return time.gmtime(max(int(float(epoch)), 315532800))[:6]


def add(zf, name, data, date):
    zi = zipfile.ZipInfo(name, date_time=date)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.create_system = 3                     # the same bytes on every platform
    zi.external_attr = 0o100644 << 16
    zf.writestr(zi, data, compresslevel=9)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Build the release files into dist/.")
    ap.add_argument("--expect", metavar="X.Y.Z", help="fail unless this is the version built (e.g. the tag without 'v')")
    ap.add_argument("--blender", metavar="PATH", help="a Blender 4.2+ executable to validate the extension zip with")
    args = ap.parse_args()

    for path in (SOURCE, MANIFEST):
        if not os.path.isfile(path):
            fail("missing " + os.path.relpath(path, ROOT))
    with open(SOURCE, "rb") as f:
        source = f.read()
    with open(MANIFEST, "rb") as f:
        manifest = f.read()
    text = source.decode("utf-8")
    mtext = manifest.decode("utf-8")

    # --- checks -------------------------------------------------------------------------------
    compile(text, SOURCE, "exec")                                   # valid Python
    if sys.version_info >= (3, 11):
        import tomllib
        tomllib.loads(mtext)                                        # valid TOML
    info = bl_info_of(text)
    version = ".".join(str(v) for v in info["version"])
    mversion = manifest_field(mtext, "version")
    ext_id = manifest_field(mtext, "id")
    if mversion != version:
        fail("version %s in bl_info, %s in the manifest" % (version, mversion))
    if args.expect and args.expect.lstrip("v") != version:
        fail("expected version %s, the files say %s" % (args.expect, version))
    stem = os.path.splitext(os.path.basename(SOURCE))[0]
    if ext_id != stem:
        fail("extension id %r differs from the add-on file name %r" % (ext_id, stem))
    if not text.startswith("# SPDX-License-Identifier: GPL-3.0-or-later"):
        fail("the add-on file lacks its SPDX license line")
    if '"SPDX:GPL-3.0-or-later"' not in mtext:
        fail("the manifest lacks the GPL-3.0-or-later license")
    if re.search(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}", text + mtext):
        fail("an e-mail address is in the add-on or the manifest")
    if os.path.isfile(README):
        with open(README, encoding="utf-8") as f:
            m = re.search(r"\*\*Version:\*\*\s*([0-9.]+)", f.read())
        if m and m.group(1) != version:
            print("build_release: warning: README.md says version %s" % m.group(1))

    # --- build --------------------------------------------------------------------------------
    os.makedirs(DIST, exist_ok=True)
    zip_name = "%s-%s.zip" % (ext_id, version)
    zip_path = os.path.join(DIST, zip_name)
    py_path = os.path.join(DIST, os.path.basename(SOURCE))
    date = build_date()
    with zipfile.ZipFile(zip_path, "w") as zf:
        add(zf, "blender_manifest.toml", manifest, date)
        add(zf, "__init__.py", source, date)
    shutil.copyfile(SOURCE, py_path)
    with zipfile.ZipFile(zip_path) as zf:                           # read back
        if zf.testzip() is not None or zf.read("__init__.py") != source:
            fail("the zip doesn't read back as written")
    sums = os.path.join(DIST, "SHA256SUMS.txt")
    with open(sums, "w", newline="\n") as f:
        for p in (zip_path, py_path):
            f.write("%s  %s\n" % (sha256(p), os.path.basename(p)))

    # --- optional: Blender's own check of the extension ----------------------------------------
    if args.blender:
        r = subprocess.run([args.blender, "--factory-startup", "--command", "extension", "validate", zip_path],
                           capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        print(out)
        if r.returncode != 0:
            fail("Blender did not accept the extension zip")

    print("UV Shell Gap Overlay %s (zip entries dated %04d-%02d-%02d)" % ((version,) + tuple(date[:3])))
    with open(sums) as f:
        for line in f:
            h, name = line.split()
            print("  dist/%-34s %8d bytes  sha256 %s" % (name, os.path.getsize(os.path.join(DIST, name)), h))


if __name__ == "__main__":
    main()
