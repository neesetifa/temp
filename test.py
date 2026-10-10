"""Privileged reference estimator (oracle) for learned_cardinality_shift.

It is NOT a legitimate solution: it knows the hidden database's dimension tables,
every customer's realised order count, every product's latent popularity and the
true generative conditionals of the order attributes, and returns the exact
expected count E[count | all of that].  The only thing it does not know is the
per-order randomness (which product each order picked and the order attributes).
The hidden data it uses lives in oracle_data.npz next to this file.
"""
import os

import numpy as np

DOM = [36, 6, 16, 4, 10, 60, 12, 40, 4, 36, 5, 16, 48, 5, 36, 40]
OFF = np.concatenate([[0], np.cumsum(DOM)]).astype(int)
KZ, NT = 8, 160


def _seg(m, i):
    return m[OFF[i]:OFF[i + 1]]


class CardinalityEstimator:
    def __init__(self, data_dir=None):
        here = os.path.dirname(os.path.abspath(__file__))
        d = np.load(os.path.join(here, "oracle_data.npz"), allow_pickle=False)
        g = lambda k: d[k].astype(np.int64)
        self.c_cols = [g("c_age"), g("c_region"), g("c_income"), g("c_tier"), g("c_signup"), g("c_pref")]
        self.p_cols = [g("p_cat"), g("p_price"), g("p_brand"), g("p_launch"), g("p_rating")]
        age, region, income, tier, signup, pref = self.c_cols
        z = g("c_z")
        self.n = d["c_n"].astype(np.float64)
        ctype = (z * 4 + tier) * 5 + np.minimum(income // 8, 4)
        self.cg = ctype * 36 + signup
        self.chkey = (pref * 5 + np.minimum(age // 12, 4)) * 12 + region
        cat, price, brand, launch, rating = self.p_cols
        self.pg = ((launch * 16 + cat) * 8 + price // 6) * 5 + brand
        self.nu = d["p_nu"]
        self.A, self.Bm, self.C, self.Zt = d["A"], d["B"], d["C"], d["Zt"]
        self.p_month, self.p_disc, self.p_qty = d["t_p_month"], d["t_p_disc"], d["t_p_qty"]
        self.p_chan, self.p_ship = d["t_p_chan"], d["t_p_ship"]
        s = np.arange(36)
        self.maxsl = np.maximum(s[:, None], s[None, :])

    def _one(self, tables, m):
        cok = np.ones(len(self.n), bool)
        for k, col in enumerate(self.c_cols):
            sm = _seg(m, 5 + k)
            if not sm.all():
                cok &= sm[col]
        pch = self.p_chan * _seg(m, 3)[None, None, :]
        psh = (self.p_ship * _seg(m, 4)[None, None, :]).sum(-1)
        fcs = np.einsum("pac,cr->par", pch, psh).reshape(-1)
        a = self.n * cok * fcs[self.chkey]
        N = np.bincount(self.cg, weights=a, minlength=NT * 36).reshape(KZ, 4, 5, 36)
        pok = np.ones(len(self.nu), bool)
        for k, col in enumerate(self.p_cols):
            sm = _seg(m, 11 + k)
            if not sm.all():
                pok &= sm[col]
        Wb = np.bincount(self.pg, weights=self.nu * pok, minlength=36 * 16 * 8 * 5).reshape(36, 16, 8, 5)
        X = np.einsum("lcpb,tb->tlcp", Wb, self.Bm)
        pq = (self.p_qty * _seg(m, 2)[None, None, :]).sum(-1)
        Y = np.einsum("ip,pz,tlcp->zitlc", self.C, pq, X) * self.A[:, None, None, None, :]
        pdisc = (self.p_disc * _seg(m, 1)[None, None, :]).sum(-1)
        pmd = np.einsum("acm,mt->act", self.p_month * _seg(m, 0)[None, None, :], pdisc)
        Zr = self.Zt.reshape(KZ, 4, 5)
        tot = 0.0
        for tr in range(4):
            R = pmd[:, :, tr][self.maxsl]
            V = Y[:, :, tr] / Zr[:, tr, :, None, None]
            tot += (N[:, tr] * np.einsum("zilc,slc->zis", V, R)).sum()
        return tot

    def estimate(self, tables, masks):
        tables = np.asarray(tables, bool)
        masks = np.asarray(masks, bool)
        return np.array([self._one(tables[q], masks[q]) for q in range(len(masks))], dtype=np.float64)


solve。sh
#!/bin/bash
cp /solution/estimator.py /solution/oracle_data.npz /app/
