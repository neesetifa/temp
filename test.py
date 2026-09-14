from __future__ import annotations
import contextlib,csv,math,os,pathlib,shutil,subprocess,tempfile
ARTIFACT_LIMIT=65_536
OUTPUT_DIR=pathlib.Path('/output'); PREDICT=OUTPUT_DIR/'predict.py'; TEST_DIR=pathlib.Path(__file__).resolve().parent; CATALOG_PATH=TEST_DIR/'catalog.csv'

def artifact_size_and_validate(root=OUTPUT_DIR):
    if not root.exists() or not root.is_dir(): raise AssertionError('/output directory missing')
    total=0
    for p in root.rglob('*'):
        if p.is_symlink(): raise AssertionError(f'symlink not allowed in /output: {p}')
        if p.is_dir(): continue
        if not p.is_file(): raise AssertionError(f'special file not allowed in /output: {p}')
        total += p.stat().st_size
    if total>ARTIFACT_LIMIT: raise AssertionError(f'artifact too large: {total} > {ARTIFACT_LIMIT} bytes')
    return total

def load_catalog():
    with CATALOG_PATH.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        if r.fieldnames != ['item_id']: raise AssertionError('invalid verifier catalog')
        out={str(row['item_id']).strip() for row in r}
    if len(out)<10: raise AssertionError('catalog too small')
    return out

@contextlib.contextmanager
def _protected_verifier_paths():
    paths=[pathlib.Path('/tests'),pathlib.Path('/app'),pathlib.Path('/logs/verifier')]; modes=[]
    for p in paths:
        try:
            if p.exists(): modes.append((p,p.stat().st_mode & 0o7777)); p.chmod(0o700)
        except OSError: pass
    try: yield
    finally:
        for p,m in reversed(modes):
            try:p.chmod(m)
            except OSError:pass

@contextlib.contextmanager
def _readonly_submission_output():
    """Keep the submitted /output artifact readable by the sandboxed process but immutable.

    run_predict executes a fresh copy of the artifact from a sandbox directory.  Submissions
    are nevertheless allowed to refer to their own packaged files via absolute /output/...
    paths.  The original artifact is therefore exposed read-only during execution rather
    than hidden together with verifier-private paths.
    """
    if not OUTPUT_DIR.exists():
        yield
        return
    entries=[OUTPUT_DIR,*OUTPUT_DIR.rglob('*')]
    saved=[]
    try:
        for p in entries:
            try:
                mode=p.stat().st_mode & 0o7777
                saved.append((p,mode))
                if p.is_dir():
                    p.chmod(0o555)
                elif p.is_file():
                    p.chmod(0o555 if (mode & 0o111) else 0o444)
            except OSError:
                pass
        yield
    finally:
        for p,mode in reversed(saved):
            try:p.chmod(mode)
            except OSError:pass

def _chown_tree(root,uid,gid):
    for p in [root,*root.rglob('*')]:
        try:os.chown(p,uid,gid)
        except OSError:pass

def run_predict(input_csv,output_csv,timeout=60):
    artifact_size_and_validate()
    if not PREDICT.exists() or not PREDICT.is_file() or PREDICT.stat().st_size==0: raise AssertionError('/output/predict.py missing or empty')
    sandbox=pathlib.Path(tempfile.mkdtemp(prefix='oe_exec_',dir='/var/tmp' if pathlib.Path('/var/tmp').exists() else None)); artifact=sandbox/'artifact'
    shutil.copytree(OUTPUT_DIR,artifact,symlinks=True); sandbox_in=sandbox/'input.csv'; sandbox_out=sandbox/'result.csv'; shutil.copy2(input_csv,sandbox_in)
    home=sandbox/'home'; tmp=sandbox/'tmp'; home.mkdir(); tmp.mkdir(); preexec=None
    if os.geteuid()==0:
        try:
            import pwd; nobody=pwd.getpwnam('nobody'); uid,gid=nobody.pw_uid,nobody.pw_gid
        except Exception: uid=gid=65534
        _chown_tree(sandbox,uid,gid)
        def drop(): os.setgroups([]); os.setgid(gid); os.setuid(uid)
        preexec=drop
    env={'PATH':os.environ.get('PATH','/usr/local/bin:/usr/bin:/bin'),'LANG':'C.UTF-8','LC_ALL':'C.UTF-8','HOME':str(home),'TMPDIR':str(tmp),'XDG_CACHE_HOME':str(home/'.cache'),'PYTHONHASHSEED':'0'}
    try:
        with _readonly_submission_output():
            with _protected_verifier_paths():
                cp=subprocess.run(['python',str(artifact/'predict.py'),str(sandbox_in),str(sandbox_out)],cwd=str(artifact),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout,preexec_fn=preexec)
        if cp.returncode!=0: raise AssertionError(f'predict.py failed with code {cp.returncode}: {(cp.stderr or cp.stdout)[-4000:]}')
        if not sandbox_out.exists() or sandbox_out.is_symlink() or not sandbox_out.is_file(): raise AssertionError('submission did not create regular output CSV')
        shutil.copy2(sandbox_out,output_csv)
    finally: shutil.rmtree(sandbox,ignore_errors=True)

def parse_predictions(output_csv,expected_rows,catalog):
    with output_csv.open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        if r.fieldnames != ['prediction']: raise AssertionError('output must have exactly one column named prediction')
        rows=list(r)
    if len(rows)!=expected_rows: raise AssertionError(f'wrong row count: {len(rows)} != {expected_rows}')
    out=[]
    for i,row in enumerate(rows):
        toks=str(row['prediction']).strip().split()
        if len(toks)!=10: raise AssertionError(f'row {i}: exactly 10 item IDs required')
        if len(set(toks))!=10: raise AssertionError(f'row {i}: item IDs must be distinct')
        bad=[x for x in toks if x not in catalog]
        if bad: raise AssertionError(f'row {i}: unknown item IDs {bad[:3]}')
        out.append(tuple(toks))
    return out

def write_input(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['query_id','history','gaps_days'])
        for r in rows:w.writerow([r[0],r[1],r[2]])

def ndcg_at_10(pred,target):
    if not target:return 0.0
    dcg=sum(1/math.log2(rank+1) for rank,item in enumerate(pred,1) if item in target); k=min(10,len(target)); idcg=sum(1/math.log2(rank+1) for rank in range(1,k+1)); return dcg/idcg if idcg else 0.0
