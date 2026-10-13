#!/bin/bash
# Run the grading test, then write /logs/verifier/reward.txt from metrics.json (0 on any failure).
LOG=/logs/verifier
mkdir -p "$LOG"
rm -rf "$LOG/reward.txt" "$LOG/metrics.json" "$LOG/per_metric"
# Grader-only paths: the codec runs as an unprivileged uid (see verifier_utils.run_codec).
chmod 700 /tests
[ -d /solution ] && chmod 700 /solution
cd /tests
python -m pytest -q -p no:cacheprovider /tests/test_smoke.py /tests/test_isolation.py /tests/test_main.py
status=$?
python - "$status" << 'PY'
import json, math, pathlib, sys
log = pathlib.Path('/logs/verifier')
reward = 0.0
if sys.argv[1] == '0':
    try:
        r = float(json.loads((log / 'metrics.json').read_text())['reward'])
        if math.isfinite(r) and 0.0 <= r <= 1.0:
            reward = r
    except Exception:
        pass
(log / 'reward.txt').write_text(f'{reward:.12f}\n')
PY
exit $status



from __future__ import annotations

import json
import math
import pathlib
import resource
import shutil
import sys
import tempfile

import numpy as np

import verifier_utils as vu

sys.path.insert(0, str(vu.KVREF))
import kvcodec_baseline as kb  # noqa: E402
import kvformat as kf  # noqa: E402

LOG_DIR = pathlib.Path('/logs/verifier')


def _baseline_cr(ids):
    a = kb.Assets(vu.KVREF / 'baseline_assets.npz')
    raw = coded = 0
    for i in ids:
        u = np.load(vu.HIDDEN / f'{i}.kv.npy', allow_pickle=False)
        raw += u.size * 2
        coded += len(kb.encode(u, a))
    return raw / coded


def test_reward():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = vu.CONFIG
    limit = float(cfg['time_limit_sec'])
    scale = math.log(float(cfg['reward_scale_ratio']))
    ids = vu.hidden_ids()
    eps = kf.eps_per_channel(np.load(vu.KVREF / 'bounds.npy')).astype(np.float32)
    eps5 = kf.from_tc(eps[None, :])                                   # (23, 2, 1, 2, 64)

    vu.setup_isolation()
    artifact_bytes = vu.artifact_size()
    with tempfile.TemporaryDirectory(prefix='kv_grade_', dir=vu.store_dir()) as td:
        td = pathlib.Path(td)
        inp, e1, e2, din, dout = (td / n for n in ('in', 'e1', 'e2', 'din', 'dout'))
        for d in (inp, e1, e2, din, dout):
            d.mkdir()
        for i in ids:
            for suf in ('.kv.npy', '.pos.npy'):
                shutil.copyfile(vu.HIDDEN / f'{i}{suf}', inp / f'{i}{suf}')

        # 1-2. encode twice into fresh directories; outputs must be byte-identical
        t_enc1 = vu.run_codec('encode', inp, e1, limit)
        t_enc2 = vu.run_codec('encode', inp, e2, limit)
        want = sorted(f'{i}.bin' for i in ids)
        assert sorted(p.name for p in e1.iterdir()) == want, 'encode.py must write exactly one <id>.bin per cache'
        assert sorted(p.name for p in e2.iterdir()) == want, 'encode.py must write exactly one <id>.bin per cache'
        for n in want:
            assert (e1 / n).read_bytes() == (e2 / n).read_bytes(), f'encoding is not deterministic ({n})'

        # 3. decode the first encoding from a directory holding only .bin + positions
        for i in ids:
            shutil.copyfile(e1 / f'{i}.bin', din / f'{i}.bin')
            shutil.copyfile(inp / f'{i}.pos.npy', din / f'{i}.pos.npy')
        shutil.rmtree(inp)
        t_dec = vu.run_codec('decode', din, dout, limit)
        assert sorted(p.name for p in dout.iterdir()) == sorted(f'{i}.npy' for i in ids), \
            'decode.py must write exactly one <id>.npy per input'

        raw = coded = 0
        max_ratio = 0.0
        for i in ids:
            u = np.load(vu.HIDDEN / f'{i}.kv.npy', allow_pickle=False)
            out = vu.load_decoded(dout / f'{i}.npy', u.shape)
            err = np.abs(out - kf.bf16_to_f32(u))
            ok = err <= eps5                                            # NaN -> False
            assert bool(ok.all()), f'{i}: {int((~ok).sum())} decoded elements outside their bound'
            max_ratio = max(max_ratio, float((err / eps5).max()))
            raw += u.size * 2
            coded += (e1 / f'{i}.bin').stat().st_size

    cr_base = _baseline_cr(ids)
    cr = raw / coded
    reward = max(0.0, min(1.0, math.log(cr / cr_base) / scale))
    assert math.isfinite(reward)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    metrics = {
        'artifact_bytes': artifact_bytes,
        'hidden_caches': len(ids),
        'raw_bytes': raw,
        'encoded_bytes': coded,
        'CR': cr,
        'CR_base': cr_base,
        'reward': reward,
        'reward_scale_ratio': float(cfg['reward_scale_ratio']),
        'encode_wall_sec': [t_enc1, t_enc2],
        'decode_wall_sec': t_dec,
        'time_limit_sec': limit,
        'max_error_over_bound': max_ratio,
        'near_scale_saturation': cr / cr_base > 0.95 * float(cfg['reward_scale_ratio']),
        'peak_child_memory_mb': peak,
    }
    (LOG_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n')
    # Numeric per-metric files: scaffolds that carry verifier numbers into notes forward these
    # (diagnostics only; the reward is unchanged).
    per = LOG_DIR / 'per_metric'
    per.mkdir(parents=True, exist_ok=True)
    for name, value in (('CR', cr), ('CR_base', cr_base), ('encoded_bytes', coded),
                        ('artifact_bytes', artifact_bytes),
                        ('encode_wall_sec', max(t_enc1, t_enc2)), ('decode_wall_sec', t_dec),
                        ('time_limit_sec', limit), ('artifact_limit_bytes', vu.ARTIFACT_LIMIT),
                        ('max_error_over_bound', max_ratio)):
        (per / f'{name}.txt').write_text(f'{float(value):.6g}\n')
