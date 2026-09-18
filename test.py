# Compact novel next-basket recommendation

`/app/train.csv` contains real purchase histories derived from UCI Online Retail II. Each row is one training customer: an ordered basket history plus the items in that customer's next basket that had **not appeared anywhere in the visible history**.

Build `/output/predict.py`.

It will be called as:

```bash
python /output/predict.py <in.csv> <out.csv>
```

`<in.csv>` has these columns:

```text
query_id,history,gaps_days
```

`history` is a sequence of baskets. Baskets are separated by `|`; item IDs inside a basket are separated by spaces. The last basket in `history` is the most recent observed basket. `gaps_days` contains one non-negative day gap per history basket, in the same order.

Write `<out.csv>` with exactly one column named `prediction`. Each row must contain exactly **10 distinct item IDs**, separated by single spaces, best recommendation first. Output rows must correspond to input rows in the same order.

The held-out label contains only **novel** next-basket items: items already present in that customer's visible history receive no credit. Held-out customers with no novel item in their next basket are not part of evaluation.

The complete `/output` artifact is limited to **64 KiB (65,536 bytes)**. The verifier meters regular-file contents **and** filesystem-encoded relative path/name bytes (plus one entry byte per file or directory), so filenames/directories cannot carry unmetered state. Extended attributes, symlinks, and other special files are not allowed. Final inference runs in an isolated filesystem view: only the submitted `/output` artifact is carried from the solve phase; files left in `/tmp`, `/var/tmp`, `/app`, home directories, or other solve-phase locations are not available to `predict.py`.

The held-out customers are disjoint from the customers in `/app/train.csv`. Their visible histories are supplied only through `<in.csv>` at evaluation time. The predictor cannot access `/app/train.csv` during final evaluation, so any cross-customer statistics or learned model state needed at inference time must fit inside `/output`.

Training columns are:

```text
query_id,history,gaps_days,target_items
```

`target_items` contains only the next-basket items that are novel relative to that training customer's visible history. IDs are opaque remapped identifiers; do not infer meaning from their numeric values.

The score is based on how highly hidden novel next-basket items appear in each held-out customer's top-10 ranking. A train-derived global novel-item popularity recommender, filtered to remove items already seen by each customer, is the score floor. The useful part of the task is deciding which cross-customer item relationships or compressed representations are worth encoding under the artifact-size limit.

Keep the submission deterministic. Reordering input rows must only reorder the corresponding outputs.


reference
#!/usr/bin/env python3
"""Budget-compliant submission reference for compact-next-basket-oe v0.4b.

This file exists to exercise the same solve.sh -> /output path used by a
submission.  It builds a deliberately modest <=64 KiB top-5 association
model.  It is NOT the reward calibration reference, NOT an oracle, NOT a
ceiling, and NOT a claimed optimum.

The reward calibration reference is a separate full-memory author-side model
with a standalone reproducible producer in author_tools/calibration_reference.py
in the source bundle.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import struct
from collections import Counter, defaultdict

ARTIFACT_LIMIT = 65_536
K = 5
SENTINEL = 65_535


def parse_history(text: str) -> list[list[int]]:
    return [
        [int(x) for x in basket.split()]
        for basket in str(text).split("|")
        if basket.strip()
    ]


def build(train_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    source_count: Counter[int] = Counter()
    target_count: Counter[int] = Counter()
    assoc: dict[int, Counter[int]] = defaultdict(Counter)
    all_items: set[int] = set()

    with train_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = ["query_id", "history", "gaps_days", "target_items"]
        if reader.fieldnames != expected:
            raise SystemExit(f"unexpected train schema: {reader.fieldnames}")
        for row in reader:
            seen = {x for basket in parse_history(row["history"]) for x in basket}
            target = {int(x) for x in row["target_items"].split() if x}
            if not seen or not target:
                continue
            all_items.update(seen)
            all_items.update(target)
            source_count.update(seen)
            target_count.update(target)
            for src in seen:
                counts = assoc[src]
                for dst in target:
                    counts[dst] += 1

    if not all_items:
        raise SystemExit("training data produced no items")
    nslots = max(all_items) + 1
    if nslots >= SENTINEL:
        raise SystemExit("reference solution assumes uint16 remapped item ids")

    # Stable global fallback: target frequency, then source frequency, then id.
    popularity = sorted(
        all_items,
        key=lambda item: (-target_count[item], -source_count[item], item),
    )

    table = [[SENTINEL] * K for _ in range(nslots)]
    for src, counts in assoc.items():
        ranked: list[tuple[float, int]] = []
        src_n = max(1, source_count[src])
        for dst, joint_n in counts.items():
            dst_n = max(1, target_count[dst])
            score = joint_n / math.sqrt(src_n * dst_n)
            ranked.append((score, dst))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        for j, (_, dst) in enumerate(ranked[:K]):
            table[src][j] = dst

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.iterdir():
        if old.is_dir():
            import shutil
            shutil.rmtree(old)
        else:
            old.unlink()

    model_path = output_dir / "model.bin"
    with model_path.open("wb") as f:
        f.write(struct.pack("<HH", nslots, len(popularity)))
        f.write(struct.pack("<" + "H" * len(popularity), *popularity))
        flat = [item for row in table for item in row]
        f.write(struct.pack("<" + "H" * len(flat), *flat))

    predictor = f'''#!/usr/bin/env python3
import csv, pathlib, struct, sys
from collections import Counter
K={K}; SENT={SENTINEL}
HERE=pathlib.Path(__file__).resolve().parent
blob=(HERE/'model.bin').read_bytes()
n,m=struct.unpack_from('<HH',blob,0); off=4
pop=struct.unpack_from('<'+'H'*m,blob,off); off+=2*m
assoc=struct.unpack_from('<'+'H'*(n*K),blob,off)
def parse_hist(s):
    return [[int(x) for x in b.split()] for b in str(s).split('|') if b.strip()]
def main():
    if len(sys.argv)!=3: raise SystemExit('usage: predict.py <in.csv> <out.csv>')
    with open(sys.argv[1],newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['query_id','history','gaps_days']: raise SystemExit('bad input schema')
        rows=list(r)
    answers=[]
    for row in rows:
        seen={{x for basket in parse_hist(row['history']) for x in basket}}
        votes=Counter()
        for src in sorted(seen):
            if 0<=src<n:
                base=src*K
                for rank in range(K):
                    dst=assoc[base+rank]
                    if dst!=SENT and dst not in seen:
                        votes[dst]+=K-rank
        out=[item for item,_ in sorted(votes.items(),key=lambda kv:(-kv[1],kv[0]))[:10]]
        have=set(out)
        if len(out)<10:
            for item in pop:
                if item not in seen and item not in have:
                    out.append(item); have.add(item)
                    if len(out)==10: break
        if len(out)!=10: raise SystemExit('could not form ten recommendations')
        answers.append(' '.join(map(str,out)))
    with open(sys.argv[2],'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['prediction']); w.writerows([[x] for x in answers])
if __name__=='__main__': main()
'''
    pred_path = output_dir / "predict.py"
    pred_path.write_text(predictor, encoding="utf-8")
    pred_path.chmod(0o755)

    artifact_bytes = 0
    for p in output_dir.rglob("*"):
        artifact_bytes += len(os.fsencode(p.relative_to(output_dir).as_posix())) + 1
        if p.is_file():
            artifact_bytes += p.stat().st_size
    if artifact_bytes > ARTIFACT_LIMIT:
        raise SystemExit(
            f"reference artifact exceeds budget: {artifact_bytes} > {ARTIFACT_LIMIT}"
        )
    print(f"reference artifact: {artifact_bytes} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("/output"))
    args = parser.parse_args()

    here = pathlib.Path(__file__).resolve()
    task_root = here.parents[1]
    train = args.train
    if train is None:
        if pathlib.Path("/app/train.csv").exists():
            train = pathlib.Path("/app/train.csv")
        else:
            train = task_root / "environment/data/train.csv"
    if not train.exists():
        raise SystemExit(f"training data not found: {train}")
    build(train, args.output)


if __name__ == "__main__":
    main()


test sh
#!/bin/bash
set +e
mkdir -p /logs/verifier
python - <<'PY'
import pathlib,time
pathlib.Path('/logs/verifier/.start_time').write_text(str(time.time()))
PY
cd /tests
python -m pytest -q -p no:cacheprovider test_security.py test_smoke.py test_schema.py test_main.py
status=$?
python /tests/finalize_artifacts.py
exit $status



test security
from __future__ import annotations

import os
import pathlib
import tempfile

import pytest

from verifier_utils import ARTIFACT_LIMIT, artifact_size_and_validate, run_predict


def test_external_solve_phase_state_is_not_visible():
    """A tiny submitted predictor must not be able to read state left outside /output."""
    secret = b'OE_EXTERNAL_STATE_MUST_NOT_BE_VISIBLE_7f93a1\n'
    candidates = [
        pathlib.Path('/tmp/oe_external_state_regression.bin'),
        pathlib.Path('/var/tmp/oe_external_state_regression.bin'),
    ]
    if os.geteuid() == 0:
        candidates.append(pathlib.Path('/oe_external_state_regression.bin'))

    created = []
    try:
        for p in candidates:
            try:
                p.write_bytes(secret)
                p.chmod(0o644)
                created.append(p)
            except OSError:
                pass
        assert created, 'security regression could not create any external-state fixture'

        with tempfile.TemporaryDirectory(prefix='oe_external_artifact_') as ad, tempfile.TemporaryDirectory(prefix='oe_external_io_') as iod:
            artifact = pathlib.Path(ad)
            io = pathlib.Path(iod)
            in_csv = io / 'in.csv'
            out_csv = io / 'out.csv'
            in_csv.write_text('query_id,history,gaps_days\nq,1,0\n', encoding='utf-8')
            literals = repr([str(p) for p in created])
            predictor = f'''import pathlib,sys\nSECRET={secret!r}\nPATHS={literals}\nvisible=[]\nfor raw in PATHS:\n    try:\n        if pathlib.Path(raw).read_bytes()==SECRET: visible.append(raw)\n    except OSError:\n        pass\nif visible:\n    pathlib.Path(sys.argv[2]).write_text("prediction\\nLEAKED\\n",encoding="utf-8")\n    raise SystemExit(0)\nprint("EXTERNAL_STATE_BLOCKED",file=sys.stderr)\nraise SystemExit(91)\n'''
            (artifact / 'predict.py').write_text(predictor, encoding='utf-8')

            with pytest.raises(AssertionError) as exc:
                run_predict(in_csv, out_csv, timeout=10, artifact_root=artifact)
            msg = str(exc.value)
            assert 'code 91' in msg and 'EXTERNAL_STATE_BLOCKED' in msg, msg
            assert not out_csv.exists(), 'external solve-phase state was visible to predictor'
    finally:
        for p in created:
            try:
                p.unlink()
            except OSError:
                pass


def test_filename_encoded_payload_counts_toward_budget():
    """Empty files with ~100 KB of user-controlled names must exceed 64 KiB."""
    with tempfile.TemporaryDirectory(prefix='oe_filename_payload_') as td:
        root = pathlib.Path(td)
        (root / 'predict.py').write_text('raise SystemExit(0)\n', encoding='utf-8')
        # 500 empty files x ~205 bytes/name: >100 KB of filename payload while
        # regular-file contents remain tiny.
        for i in range(500):
            name = f'{i:04d}_' + ('A' * 200)
            (root / name).touch()

        content_only = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        assert content_only < 1024
        with pytest.raises(AssertionError, match='artifact too large'):
            artifact_size_and_validate(root)

        name_bytes = sum(
            len(os.fsencode(p.relative_to(root).as_posix())) + 1
            for p in root.rglob('*')
        )
        assert name_bytes > ARTIFACT_LIMIT

verifier
from __future__ import annotations

import csv
import math
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile

ARTIFACT_LIMIT = 65_536
OUTPUT_DIR = pathlib.Path('/output')
PREDICT = OUTPUT_DIR / 'predict.py'
TEST_DIR = pathlib.Path(__file__).resolve().parent
CATALOG_PATH = TEST_DIR / 'catalog.csv'


def _encoded_relative_path_bytes(root: pathlib.Path, path: pathlib.Path) -> int:
    """Bytes needed to represent one submitted directory entry name/path.

    File contents are not the whole artifact: filenames and directory names are
    user-controlled state too.  Count the filesystem-encoded relative path plus
    one entry-type byte for every file/directory under /output.
    """
    rel = path.relative_to(root)
    return len(os.fsencode(rel.as_posix())) + 1


def _assert_no_xattrs(path: pathlib.Path) -> None:
    """Reject extended attributes so they cannot become an unmetered state channel."""
    if not hasattr(os, 'listxattr'):
        return
    try:
        attrs = os.listxattr(path, follow_symlinks=False)
    except OSError as exc:
        # Filesystems without xattr support cannot carry an xattr payload.
        if exc.errno in {getattr(os, 'ENOTSUP', 95), 95}:
            return
        raise AssertionError(f'cannot validate extended attributes on {path}: {exc}') from exc
    if attrs:
        raise AssertionError(f'extended attributes not allowed in /output: {path}')


def artifact_size_and_validate(root=OUTPUT_DIR):
    root = pathlib.Path(root)
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise AssertionError('/output directory missing or invalid')
    _assert_no_xattrs(root)

    total = 0
    entries = sorted(root.rglob('*'), key=lambda p: os.fsencode(p.relative_to(root).as_posix()))
    for p in entries:
        if p.is_symlink():
            raise AssertionError(f'symlink not allowed in /output: {p}')
        try:
            st = p.stat(follow_symlinks=False)
        except OSError as exc:
            raise AssertionError(f'cannot stat /output entry {p}: {exc}') from exc
        _assert_no_xattrs(p)

        # Meter filenames/directory entries as part of the 64 KiB artifact.
        total += _encoded_relative_path_bytes(root, p)
        if stat.S_ISDIR(st.st_mode):
            pass
        elif stat.S_ISREG(st.st_mode):
            total += st.st_size
        else:
            raise AssertionError(f'special file not allowed in /output: {p}')

        if total > ARTIFACT_LIMIT:
            raise AssertionError(f'artifact too large: {total} > {ARTIFACT_LIMIT} bytes')
    return total


def load_catalog():
    with CATALOG_PATH.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        if r.fieldnames != ['item_id']:
            raise AssertionError('invalid verifier catalog')
        out = {str(row['item_id']).strip() for row in r}
    if len(out) < 10:
        raise AssertionError('catalog too small')
    return out


def _copy_artifact_canonical(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy only names/content into the inference sandbox.

    Metadata such as mtimes, uid/gid, hard-link topology, ACLs and xattrs must
    not become hidden model state.  Executability is preserved as a single
    boolean so packaged helper executables remain usable.
    """
    dst.mkdir(parents=True, exist_ok=False)

    def rec(a: pathlib.Path, b: pathlib.Path) -> None:
        for child in sorted(a.iterdir(), key=lambda p: os.fsencode(p.name)):
            target = b / child.name
            st = child.stat(follow_symlinks=False)
            if stat.S_ISDIR(st.st_mode):
                target.mkdir(mode=0o755)
                rec(child, target)
                target.chmod(0o555)
                os.utime(target, ns=(0, 0), follow_symlinks=False)
            elif stat.S_ISREG(st.st_mode):
                shutil.copyfile(child, target, follow_symlinks=False)
                target.chmod(0o555 if (st.st_mode & 0o111) else 0o444)
                os.utime(target, ns=(0, 0), follow_symlinks=False)
            else:
                raise AssertionError(f'special file not allowed in /output: {child}')

    rec(src, dst)
    dst.chmod(0o555)
    os.utime(dst, ns=(0, 0), follow_symlinks=False)


# Trusted launcher executed only after entering a fresh user+mount namespace.
# It constructs a chroot that exposes the Python runtime read-only, the copied
# artifact at /output, one input file, one writable result directory, and fresh
# tmp/home/dev state.  Solve-phase paths such as host /tmp, /var/tmp, /app,
# /root, /home, /tests, /logs, or arbitrary top-level directories are absent.
_SANDBOX_LAUNCHER = r'''\
from __future__ import annotations
import ctypes, glob, os, pathlib, sys

MS_RDONLY=1; MS_NOSUID=2; MS_NODEV=4; MS_BIND=4096; MS_REC=16384; MS_REMOUNT=32; MS_PRIVATE=1<<18
PR_CAPBSET_DROP=24; PR_SET_NO_NEW_PRIVS=38; PR_SET_SECUREBITS=28
SECBIT_NOROOT=1; SECBIT_NOROOT_LOCKED=2; SECBIT_NO_SETUID_FIXUP=4; SECBIT_NO_SETUID_FIXUP_LOCKED=8
LINUX_CAPABILITY_VERSION_3=0x20080522

libc=ctypes.CDLL(None, use_errno=True)

class CapHeader(ctypes.Structure):
    _fields_=[('version',ctypes.c_uint32),('pid',ctypes.c_int)]
class CapData(ctypes.Structure):
    _fields_=[('effective',ctypes.c_uint32),('permitted',ctypes.c_uint32),('inheritable',ctypes.c_uint32)]

def check(rc, what):
    if rc != 0:
        e=ctypes.get_errno(); raise OSError(e, os.strerror(e), what)

def mount(src, dst, flags=0, fstype=None, data=None):
    b=lambda x: None if x is None else os.fsencode(str(x))
    check(libc.mount(b(src), b(dst), b(fstype), ctypes.c_ulong(flags), b(data)), 'mount '+str(dst))

def bind_ro(src: pathlib.Path, dst: pathlib.Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        mount(src, dst, MS_BIND|MS_REC)
        mount(None, dst, MS_BIND|MS_REMOUNT|MS_RDONLY|MS_NOSUID)
    else:
        dst.touch()
        mount(src, dst, MS_BIND)
        mount(None, dst, MS_BIND|MS_REMOUNT|MS_RDONLY|MS_NOSUID)

def bind_rw(src: pathlib.Path, dst: pathlib.Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        mount(src, dst, MS_BIND|MS_REC)
    else:
        dst.touch()
        mount(src, dst, MS_BIND)

def expose_runtime_path(root: pathlib.Path, src: pathlib.Path):
    if not src.exists() and not src.is_symlink():
        return
    dst=root / src.as_posix().lstrip('/')
    if src.is_symlink():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink(): dst.unlink()
        os.symlink(os.readlink(src), dst)
    else:
        bind_ro(src, dst)

def drop_namespace_capabilities():
    # The launcher needs mount/chroot capability, the submission does not.
    for cap in range(64):
        libc.prctl(PR_CAPBSET_DROP, cap, 0, 0, 0)
    secure=(SECBIT_NOROOT|SECBIT_NOROOT_LOCKED|SECBIT_NO_SETUID_FIXUP|SECBIT_NO_SETUID_FIXUP_LOCKED)
    check(libc.prctl(PR_SET_SECUREBITS, secure, 0, 0, 0), 'PR_SET_SECUREBITS')
    hdr=CapHeader(LINUX_CAPABILITY_VERSION_3,0); data=(CapData*2)()
    check(libc.capset(ctypes.byref(hdr), ctypes.byref(data)), 'capset')
    check(libc.prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0), 'PR_SET_NO_NEW_PRIVS')

root=pathlib.Path(sys.argv[1]).resolve()
artifact=pathlib.Path(sys.argv[2]).resolve()
in_csv=pathlib.Path(sys.argv[3]).resolve()
work=pathlib.Path(sys.argv[4]).resolve()
runtime_python=pathlib.Path(sys.argv[5])

mount(None, '/', MS_REC|MS_PRIVATE)

# Expose only runtime code required to execute Python and installed packages.
expose_runtime_path(root, pathlib.Path('/usr'))
for p in (pathlib.Path('/bin'), pathlib.Path('/lib'), pathlib.Path('/lib64'), pathlib.Path('/sbin')):
    expose_runtime_path(root, p)

prefixes=[]
for raw in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, str(runtime_python.parent.parent)):
    p=pathlib.Path(raw).resolve()
    if p == pathlib.Path('/') or p == pathlib.Path('/usr') or pathlib.Path('/usr') in p.parents:
        continue
    if any(p == q or q in p.parents for q in prefixes):
        continue
    prefixes=[q for q in prefixes if p not in q.parents]
    prefixes.append(p)
for p in prefixes:
    expose_runtime_path(root, p)

# Fresh, isolated mutable locations.
for rel, mode in [('tmp',0o1777),('var/tmp',0o1777),('home/sandbox',0o700),('dev',0o755),('dev/shm',0o1777),('work',0o755),('output',0o755)]:
    p=root/rel; p.mkdir(parents=True, exist_ok=True); p.chmod(mode)
try:
    mount('tmpfs', root/'dev/shm', 0, 'tmpfs', 'size=64m,nosuid,nodev')
except OSError:
    # Some kernels disallow tmpfs in nested user namespaces. Empty /dev/shm is
    # still isolated; libraries may fall back to serial execution.
    pass
for name in ('null','zero','random','urandom'):
    src=pathlib.Path('/dev')/name
    if src.exists():
        dst=root/'dev'/name; dst.touch(); mount(src,dst,MS_BIND)

bind_ro(artifact, root/'output')
bind_ro(in_csv, root/'input.csv')
bind_rw(work, root/'work')

# Dynamic loaders in official Python images may expect /usr/local/lib. Avoid
# exposing host /etc/ld.so.cache by giving an explicit library search path.
libdirs=[]
for candidate in ['/usr/local/lib'] + glob.glob('/usr/lib/*-linux-gnu') + glob.glob('/lib/*-linux-gnu') + ['/usr/lib','/lib']:
    if os.path.isdir(candidate) and candidate not in libdirs:
        libdirs.append(candidate)

env={
    'PATH':'/usr/local/bin:/usr/bin:/bin',
    'LD_LIBRARY_PATH':':'.join(libdirs),
    'LANG':'C.UTF-8','LC_ALL':'C.UTF-8',
    'HOME':'/home/sandbox','TMPDIR':'/tmp','XDG_CACHE_HOME':'/tmp/.cache',
    'PYTHONHASHSEED':'0','PYTHONDONTWRITEBYTECODE':'1',
    'JOBLIB_TEMP_FOLDER':'/tmp','MPLCONFIGDIR':'/tmp/matplotlib',
}

os.chroot(root)
os.chdir('/output')
drop_namespace_capabilities()
os.execve(str(runtime_python), [str(runtime_python), '/output/predict.py', '/input.csv', '/work/result.csv'], env)
'''


def run_predict(input_csv, output_csv, timeout=60, artifact_root=OUTPUT_DIR):
    artifact_root = pathlib.Path(artifact_root)
    artifact_size_and_validate(artifact_root)
    predict = artifact_root / 'predict.py'
    if not predict.exists() or not predict.is_file() or predict.is_symlink() or predict.stat().st_size == 0:
        raise AssertionError('/output/predict.py missing or empty')

    unshare = shutil.which('unshare')
    if not unshare:
        raise AssertionError('verifier sandbox requires util-linux unshare')

    sandbox = pathlib.Path(tempfile.mkdtemp(prefix='oe_exec_', dir='/var/tmp' if pathlib.Path('/var/tmp').exists() else None))
    artifact = sandbox / 'artifact'
    rootfs = sandbox / 'rootfs'
    work = sandbox / 'work'
    sandbox_in = sandbox / 'input.csv'
    launcher = sandbox / 'launcher.py'
    work.mkdir()
    rootfs.mkdir()
    shutil.copyfile(input_csv, sandbox_in)
    _copy_artifact_canonical(artifact_root, artifact)
    launcher.write_text(_SANDBOX_LAUNCHER, encoding='utf-8')

    cmd = [
        unshare, '--user', '--map-root-user', '--mount', '--pid', '--fork',
        sys.executable, str(launcher), str(rootfs), str(artifact), str(sandbox_in), str(work), sys.executable,
    ]
    env = {
        'PATH': os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin'),
        'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
        'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1',
    }
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(sandbox),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        sandbox_out = work / 'result.csv'
        if cp.returncode != 0:
            detail = (cp.stderr or cp.stdout)[-4000:]
            raise AssertionError(f'predict.py failed with code {cp.returncode}: {detail}')
        if not sandbox_out.exists() or sandbox_out.is_symlink() or not sandbox_out.is_file():
            raise AssertionError('submission did not create regular output CSV')
        shutil.copyfile(sandbox_out, output_csv)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def parse_predictions(output_csv, expected_rows, catalog):
    with output_csv.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        if r.fieldnames != ['prediction']:
            raise AssertionError('output must have exactly one column named prediction')
        rows = list(r)
    if len(rows) != expected_rows:
        raise AssertionError(f'wrong row count: {len(rows)} != {expected_rows}')
    out = []
    for i, row in enumerate(rows):
        toks = str(row['prediction']).strip().split()
        if len(toks) != 10:
            raise AssertionError(f'row {i}: exactly 10 item IDs required')
        if len(set(toks)) != 10:
            raise AssertionError(f'row {i}: item IDs must be distinct')
        bad = [x for x in toks if x not in catalog]
        if bad:
            raise AssertionError(f'row {i}: unknown item IDs {bad[:3]}')
        out.append(tuple(toks))
    return out


def write_input(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['query_id', 'history', 'gaps_days'])
        for r in rows:
            w.writerow([r[0], r[1], r[2]])


def ndcg_at_10(pred, target):
    if not target:
        return 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(pred, 1) if item in target)
    k = min(10, len(target))
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, k + 1))
    return dcg / idcg if idcg else 0.0


