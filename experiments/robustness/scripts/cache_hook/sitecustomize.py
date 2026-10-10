"""Experiment A: memoize GraphST / STAGATE training on identical inputs.

Active only when EXPA_TRAIN_CACHE points to a directory. ChatSpatial runs these
backends in spawned worker processes; this module is imported at interpreter
start (it is on PYTHONPATH), so the patch also applies inside the workers.
The cache key is a hash of everything the training step reads (features,
graph, seed, epochs). Grid configurations that differ only in clustering or
refinement parameters (n_domains, clustering method, radius) then reuse one
trained embedding; the e2e test checks that cached and fresh embeddings match.
"""
import hashlib
import os

_CACHE = os.environ.get("EXPA_TRAIN_CACHE")


def _h(*parts):
    import numpy as np
    m = hashlib.sha1()
    for p in parts:
        if hasattr(p, "detach"):
            p = p.detach().cpu()
            p = p.to_dense() if getattr(p, "is_sparse", False) else p
            p = p.numpy()
        if hasattr(p, "toarray"):
            p = p.toarray()
        if isinstance(p, np.ndarray):
            m.update(str(p.shape).encode()); m.update(np.ascontiguousarray(p).tobytes())
        else:
            m.update(repr(p).encode())
    return m.hexdigest()


def _save(path, arr):
    import numpy as np
    tmp = f"{path}.{os.getpid()}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


if _CACHE:
    os.makedirs(_CACHE, exist_ok=True)
    try:
        import numpy as np
        import GraphST.GraphST as _g

        _orig_train = _g.GraphST.train

        def _train(self):
            # Optional re-seed (pilot only): pre-release ChatSpatial trained GraphST in an
            # unseeded worker. ChatSpatial 1.5.5 seeds the worker itself, so this is off.
            if os.environ.get("EXPA_GRAPHST_RESEED") == "1":
                from GraphST.preprocess import fix_seed as _fix_seed
                _fix_seed(self.random_seed)
            if self.deconvolution:
                return _orig_train(self)
            key = _h("graphst", self.features, self.adj, self.label_CSL, self.random_seed,
                     self.epochs, self.datatype, self.dim_output, self.learning_rate)
            path = os.path.join(_CACHE, f"graphst_{key}.npy")
            if os.path.exists(path):
                self.adata.obsm["emb"] = np.load(path)
                return self.adata
            out = _orig_train(self)
            _save(path, out.obsm["emb"])
            return out

        _g.GraphST.train = _train
    except Exception:  # backend missing: nothing to patch
        pass
    try:
        import numpy as np
        import STAGATE_pyG
        import STAGATE_pyG.Train_STAGATE as _t

        _orig_stagate = _t.train_STAGATE

        def _train_stagate(adata, *args, **kwargs):
            if args or set(kwargs) - {"device", "verbose", "random_seed"}:
                return _orig_stagate(adata, *args, **kwargs)
            net = adata.uns.get("Spatial_Net")
            hv = adata.var["highly_variable"].values if "highly_variable" in adata.var else None
            key = _h("stagate", adata.X, hv, net.values.astype(str) if net is not None else None,
                     str(kwargs.get("device")), kwargs.get("random_seed", 0))
            path = os.path.join(_CACHE, f"stagate_{key}.npy")
            if os.path.exists(path):
                adata.obsm["STAGATE"] = np.load(path)
                return adata
            out = _orig_stagate(adata, **kwargs)
            _save(path, out.obsm["STAGATE"])
            return out

        _t.train_STAGATE = _train_stagate
        STAGATE_pyG.train_STAGATE = _train_stagate
    except Exception:
        pass
