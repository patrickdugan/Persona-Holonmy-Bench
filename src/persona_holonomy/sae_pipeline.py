"""Training-only activation corpus and interoperable PercInterp TopK SAE checkpoints.

This module deliberately stops at reconstruction quality and checkpoint
provenance. It does not invent a direct-attribution graph. All model and SAE
weights remain local, and PHB held-out seeds/probes are excluded from fitting.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any

from .interventions import _unpack_hidden, decoder_blocks, prepare_pairs, select_block
from .perc_components import _hash_file, _load_json, _validate_plan
from .sweep import _partition_pairs

DATA_SCHEMA = "phb.sae.activations.v1"
REGISTRY_SCHEMA = "phb.sae.registry.v1"


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def capture_plan(cases: list[dict[str, Any]], *,
                 heldout_probes: tuple[str, ...] = ("parser_validator",),
                 phase: str = "exit", source: str = "ordered",
                 reference: str = "tail_matched",
                 holdout_fraction: float = .4,
                 split_seed: int = 20261022,
                 sae_validation_fraction: float = .25) -> dict[str, Any]:
    """Freeze data membership before loading model weights.

    SAE validation seeds are withheld from SAE optimization, but remain inside
    the PHB causal *training* split. PHB causal evaluation seeds and probe
    families are never captured. Splitting is by seed, not token.
    """
    if not 0 < sae_validation_fraction < 1:
        raise ValueError("SAE validation fraction must be between zero and one")
    if not heldout_probes:
        raise ValueError("Require at least one held-out technical probe")
    pairs = prepare_pairs(cases, source_conditions=(source,),
                          target_condition=reference, phases=(phase,))
    train_seeds, test_seeds, training, _, train_probes, eval_probes = _partition_pairs(
        pairs, holdout_fraction=holdout_fraction, split_seed=split_seed,
        heldout_probe_ids=heldout_probes)
    if len(train_seeds) < 2:
        raise ValueError("At least two PHB training seeds needed for SAE train/validation")
    shuffled = list(train_seeds)
    random.Random(split_seed ^ 0x5AE).shuffle(shuffled)
    count = max(1, min(len(shuffled) - 1, round(len(shuffled) * sae_validation_fraction)))
    validation_seeds = set(shuffled[:count])
    fit_seeds = set(train_seeds) - validation_seeds
    chosen: dict[str, dict] = {}
    for donor, recipient in training:
        for case in (donor, recipient):
            if case["seed"] in test_seeds or case["probe_id"] in eval_probes:
                raise ValueError("PHB evaluation case leaked into SAE capture")
            if case["case_id"] in chosen:
                raise ValueError("Duplicate case_id among SAE fitting cases")
            chosen[case["case_id"]] = case
    entries = []
    for key in sorted(chosen):
        case = chosen[key]
        entries.append({
            "case_id": key, "seed": case["seed"], "probe_id": case["probe_id"],
            "condition": case["condition"],
            "role": "validation" if case["seed"] in validation_seeds else "fit",
            "messages_sha256": _sha_json(case["messages"]),
        })
    if not entries or not ({e["role"] for e in entries} == {"fit", "validation"}):
        raise ValueError("SAE split produced an empty fitting or validation partition")
    return {
        "schema": DATA_SCHEMA, "source": source, "reference": reference,
        "phase": phase, "split_seed": split_seed,
        "holdout_fraction": holdout_fraction,
        "sae_validation_fraction": sae_validation_fraction,
        "phb_train_seeds": list(train_seeds),
        "phb_test_seeds_excluded": list(test_seeds),
        "phb_training_probes": sorted(train_probes),
        "phb_test_probes_excluded": sorted(eval_probes),
        "sae_fit_seeds": sorted(fit_seeds),
        "sae_validation_seeds": sorted(validation_seeds),
        "selected_cases": entries,
        "n_selected_cases": len(entries),
        "input_cases_sha256": _sha_json(cases),
        "warnings": [
            "Only PHB causal-training seeds and technical probes are captured.",
            "SAE validation is a subset of PHB causal-training seeds, and is not causal evaluation.",
            "Scripted PHB conversations are far too narrow to validate a general-purpose SAE.",
            "Activation matrices can leak information about their input; keep them local.",
        ],
    }


def _resolve_layers(names: str, n: int) -> tuple[int, ...]:
    raw = [s.strip() for s in names.split(",")]
    if not raw or any(not s for s in raw):
        raise ValueError("Provide one or more decoder-block indexes")
    chosen = tuple(select_block(s, n) for s in raw)
    if len(set(chosen)) != len(chosen):
        raise ValueError("Layer aliases must name distinct decoder blocks")
    return chosen


def _choose_tokens(count: int, max_count: int, seed: int) -> list[int]:
    if count < 1 or max_count < 1:
        raise ValueError("Token counts must be positive")
    if count <= max_count:
        return list(range(count))
    # Retain the prefix-to-continuation boundary for downstream coverage.
    earlier = random.Random(seed).sample(range(count - 1), max_count - 1)
    return sorted([*earlier, count - 1])


def collect_activations(*, cases: list[dict], model: Any, tokenizer: Any,
                        model_name: str, checkpoint_revision: str,
                        output_dir: str | Path, layers: str = "0,mid,last",
                        heldout_probes: tuple[str, ...] = ("parser_validator",),
                        phase: str = "exit", source: str = "ordered",
                        reference: str = "tail_matched",
                        holdout_fraction: float = .4,
                        split_seed: int = 20261022,
                        sae_validation_fraction: float = .25,
                        max_input_tokens: int = 4096,
                        max_tokens_per_case: int = 64,
                        seed: int = 20261022,
                        device: str = "cpu") -> dict:
    """Collect bounded train-only token states from real decoder blocks.

    No model parameters or raw tokens/text are written. The captured vectors
    are stored as float32 .npz files, one file per decoder block.
    """
    import numpy as np
    import torch
    if not checkpoint_revision.strip():
        raise ValueError("Pin an immutable model revision or weights hash")
    if max_input_tokens < 1 or max_tokens_per_case < 1:
        raise ValueError("Token caps must be positive")
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Model tokenizer must provide a chat template")
    plan = capture_plan(cases, heldout_probes=heldout_probes, phase=phase,
                        source=source, reference=reference,
                        holdout_fraction=holdout_fraction, split_seed=split_seed,
                        sae_validation_fraction=sae_validation_fraction)
    out = Path(output_dir)
    if (out / "capture-manifest.json").exists():
        raise FileExistsError("Capture manifest already exists; choose a fresh output directory")
    block_path, blocks = decoder_blocks(model)
    indices = _resolve_layers(layers, len(blocks))
    case_lookup = {c["case_id"]: c for c in cases}
    samples = {i: {"fit": [], "validation": []} for i in indices}
    token_counts = defaultdict(int)
    model.eval()
    for entry in plan["selected_cases"]:
        case = case_lookup[entry["case_id"]]
        if entry["messages_sha256"] != _sha_json(case["messages"]):
            raise ValueError("Case content changed after SAE preflight")
        encoded = tokenizer.apply_chat_template(
            case["messages"], tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        if isinstance(encoded, torch.Tensor):
            inputs = {"input_ids": encoded.to(device)}
        else:
            inputs = {k: v.to(device) for k, v in encoded.items()
                      if isinstance(v, torch.Tensor)}
        if "input_ids" not in inputs or inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
            raise ValueError("Chat tokenizer must return one unbatched causal sequence")
        total_tokens = int(inputs["input_ids"].shape[1])
        if total_tokens > max_input_tokens:
            raise ValueError(f"Input {entry['case_id']} has {total_tokens} tokens, above cap {max_input_tokens}")
        sample_seed = int.from_bytes(hashlib.sha256(
            f"{seed}|{entry['case_id']}".encode()).digest()[:8], "big")
        positions = _choose_tokens(total_tokens, max_tokens_per_case, sample_seed)
        recorded: dict[int, Any] = {}
        counters = defaultdict(int)
        with ExitStack() as stack, torch.inference_mode():
            for index in indices:
                def make_hook(idx):
                    def hook(_module, _args, output):
                        h = _unpack_hidden(output)
                        if h.shape[1] != total_tokens:
                            raise ValueError("Unexpected decoder block token dimension")
                        counters[idx] += 1
                        recorded[idx] = (h[0, positions, :].detach().float()
                                         .cpu().numpy().copy())
                    return hook
                handle = blocks[index].register_forward_hook(make_hook(index))
                stack.callback(handle.remove)
            model(**inputs, use_cache=False, return_dict=True)
        if any(counters[i] != 1 for i in indices):
            raise RuntimeError("Every captured decoder block must be called exactly once")
        for index in indices:
            samples[index][entry["role"]].append(recorded[index])
        token_counts[entry["role"]] += len(positions)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {**plan, "model": model_name,
                "tokenizer": str(getattr(tokenizer, "name_or_path", "unknown")),
                "checkpoint_revision": checkpoint_revision,
                "block_path": block_path, "requested_layers": layers,
                "max_input_tokens": max_input_tokens,
                "max_tokens_per_case": max_tokens_per_case,
                "capture_seed": seed, "token_counts": dict(token_counts),
                "blocks": {}}
    for index in indices:
        matrix = {role: np.concatenate(samples[index][role], axis=0).astype(np.float32)
                  for role in ("fit", "validation")}
        if matrix["fit"].ndim != 2 or matrix["fit"].shape[1] != matrix["validation"].shape[1]:
            raise ValueError("SAE feature widths differ across fit and validation")
        if not all(np.isfinite(a).all() for a in matrix.values()):
            raise ValueError("Captured SAE activations contain nonfinite values")
        dest = out / f"block-{index:03d}.npz"
        np.savez_compressed(dest, train=matrix["fit"], validation=matrix["validation"])
        manifest["blocks"][str(index)] = {
            "path": dest.name, "sha256": _hash_file(dest),
            "n_train_tokens": int(matrix["fit"].shape[0]),
            "n_validation_tokens": int(matrix["validation"].shape[0]),
            "hidden_width": int(matrix["fit"].shape[1]),
        }
    (out / "capture-manifest.json").write_text(json.dumps(manifest, indent=2,
        ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def collect_hf(*, input_path: str, output_dir: str, model_name: str,
               checkpoint_revision: str, device: str = "auto", dtype: str = "auto",
               allow_download: bool = False, trust_remote_code: bool = False,
               **kwargs) -> dict:
    """Load cached pretrained HF model only after enforcing data split plan."""
    from .interventions import _read_cases
    cases = _read_cases(input_path)
    options = {k: kwargs[k] for k in ("heldout_probes", "phase", "source",
               "reference", "holdout_fraction", "split_seed",
               "sae_validation_fraction") if k in kwargs}
    capture_plan(cases, **options)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the optional .[mechanistic] dependencies") from exc
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    if device not in ("cpu", "cuda", "mps") or dtype not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("Unsupported device or dtype")
    opts = {"local_files_only": not allow_download, "trust_remote_code": trust_remote_code}
    tok = AutoTokenizer.from_pretrained(model_name, **opts)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        dtype="auto" if dtype == "auto" else getattr(torch, dtype), **opts).to(device)
    return collect_activations(cases=cases, model=model, tokenizer=tok,
        model_name=model_name, checkpoint_revision=checkpoint_revision,
        output_dir=output_dir, device=device, **kwargs)


def _check_capture(capture: dict, root: Path) -> dict[str, tuple[Path, dict]]:
    if capture.get("schema") != DATA_SCHEMA or not capture.get("blocks"):
        raise ValueError("Expected PHB SAE activation capture manifest")
    fit = set(capture.get("sae_fit_seeds", []))
    val = set(capture.get("sae_validation_seeds", []))
    test = set(capture.get("phb_test_seeds_excluded", []))
    if not fit or not val or fit & val or (fit | val) & test:
        raise ValueError("SAE fit/validation and PHB test seeds must be disjoint")
    train_probes = set(capture.get("phb_training_probes", []))
    eval_probes = set(capture.get("phb_test_probes_excluded", []))
    if not train_probes or train_probes & eval_probes:
        raise ValueError("SAE fitting must exclude held-out technical questions")
    files = {}
    for block, entry in capture["blocks"].items():
        if not re.fullmatch(r"\d+", block) or not isinstance(entry, dict):
            raise ValueError("Invalid decoder block in capture manifest")
        path = root / entry["path"]
        if path.suffix != ".npz" or not path.is_file() or _hash_file(path) != entry["sha256"]:
            raise ValueError(f"SAE activation archive missing or hash changed: {path}")
        files[block] = (path, entry)
    return files


def _fvu(sae: Any, array, *, torch: Any, device: str, batch: int) -> tuple[float, float]:
    import numpy as np
    total = float(np.square(array.astype(np.float64) - array.mean(0, dtype=np.float64)).sum())
    if total <= 0:
        raise ValueError("SAE validation corpus has zero variance")
    errors = 0.
    fired = set()
    with torch.no_grad():
        for offset in range(0, len(array), batch):
            xs = torch.as_tensor(array[offset:offset + batch], device=device, dtype=torch.float32)
            reconstruct, activations, _ = sae(xs)
            errors += float((xs - reconstruct).square().sum().item())
            fired.update(torch.nonzero((activations > 0).any(0), as_tuple=False).flatten().tolist())
    return float(errors / total), float(len(fired) / sae.n)


def fit_topk(*, capture_manifest: str, output_dir: str,
             checkpoint_revision: str | None = None,
             n_latents: int = 0, k: int = 8,
             steps: int = 100, batch_size: int = 32,
             lr: float = 3e-4, seed: int = 20261022,
             device: str = "cpu") -> dict:
    """Fit PercInterp TopKSAE on train tokens, validate on unseen *seeds*.

    Training uses actual PercInterp's SAE implementation, preserving its
    checkpoint state() contract. No model inference or network access here.
    """
    import numpy as np
    import torch
    try:
        from percinterp.sae import TopKSAE
    except ImportError as exc:
        raise RuntimeError("Install the known PercInterp source for TopK SAE training") from exc
    if steps < 1 or batch_size < 2 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Steps/batch size/learning rate must be positive")
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("Unsupported SAE training device")
    if n_latents < 0 or k < 1:
        raise ValueError("n_latents must be >= 0 and k positive")
    path = Path(capture_manifest)
    capture = _load_json(path)
    files = _check_capture(capture, path.parent)
    if checkpoint_revision is not None and checkpoint_revision != capture["checkpoint_revision"]:
        raise ValueError("Requested model revision differs from activation capture")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "sae-registry.json").exists():
        raise FileExistsError("SAE registry already exists; choose a new training output directory")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    registry = {
        "schema": REGISTRY_SCHEMA, "model": capture["model"],
        "tokenizer": capture["tokenizer"],
        "checkpoint_revision": capture["checkpoint_revision"],
        "capture_manifest_sha256": _hash_file(path),
        "phb_train_seeds": capture["phb_train_seeds"],
        "phb_test_seeds_excluded": capture["phb_test_seeds_excluded"],
        "phb_training_probes": capture["phb_training_probes"],
        "phb_test_probes_excluded": capture["phb_test_probes_excluded"],
        "sae_fit_seeds": capture["sae_fit_seeds"],
        "sae_validation_seeds": capture["sae_validation_seeds"],
        "training": {"steps": steps, "batch_size": batch_size,
                     "learning_rate": lr, "seed": seed, "device": device},
        "blocks": {},
        "limitations": [
            "A reconstruction validation FVU does not certify identifiable causal concepts.",
            "Train/validation texts are from a narrow scripted benchmark, not a broad corpus.",
            "Actual Qwen pretrained-model weights have not been run by this package author.",
            "SAE feature identities require stability and causal fidelity checks before graph attribution.",
        ],
    }
    for block, (file, info) in sorted(files.items(), key=lambda x: int(x[0])):
        with np.load(file, allow_pickle=False) as data:
            X = np.asarray(data["train"], dtype=np.float32)
            V = np.asarray(data["validation"], dtype=np.float32)
        if X.ndim != 2 or V.ndim != 2 or X.shape[1] != V.shape[1] or X.shape[1] < 1:
            raise ValueError("SAE train/validation matrix dimensions are invalid")
        if not all(np.isfinite(a).all() for a in (X, V)):
            raise ValueError("SAE matrices contain nonfinite values")
        if not len(X) >= 4 or not len(V) >= 2:
            raise ValueError("SAE corpus needs at least four train and two validation token states")
        if len(X) != info["n_train_tokens"] or len(V) != info["n_validation_tokens"]:
            raise ValueError("Activation matrix counts do not match frozen manifest")
        width = int(X.shape[1])
        dictionary = int(n_latents or width * 4)
        if not 1 <= k <= dictionary:
            raise ValueError("SAE k must lie between one and dictionary width")
        sae = TopKSAE(d=width, n_latents=dictionary, k=k, k_aux=min(256, dictionary)).to(device)
        sae.train()
        with torch.no_grad():
            scale_sample = torch.as_tensor(X[:min(1024, len(X))], device=device)
            sae.set_scale(scale_sample)
        optim = torch.optim.Adam(sae.parameters(), lr=lr)
        for _ in range(steps):
            indices = rng.integers(0, len(X), size=batch_size)
            batch = torch.as_tensor(X[indices], dtype=torch.float32, device=device)
            optim.zero_grad(set_to_none=True)
            loss, _ = sae.loss(batch)
            if not torch.isfinite(loss):
                raise ValueError("SAE optimization diverged to nonfinite loss")
            loss.backward()
            sae.remove_parallel_grad()
            optim.step()
            sae.normalize_decoder()
        sae.eval()
        train_fvu, train_active = _fvu(sae, X, torch=torch, device=device, batch=batch_size)
        val_fvu, val_active = _fvu(sae, V, torch=torch, device=device, batch=batch_size)
        if not math.isfinite(train_fvu) or not math.isfinite(val_fvu):
            raise ValueError("Nonfinite SAE evaluation FVU")
        ckpt = out / f"sae-block-{int(block):03d}.pt"
        torch.save(sae.state(), ckpt)
        registry["blocks"][block] = {
            "path": ckpt.name, "sha256": _hash_file(ckpt),
            "top_k": k, "n_latents": dictionary, "hidden_width": width,
            "fit_fvu": train_fvu, "validation_fvu": val_fvu,
            "fit_feature_coverage": train_active,
            "validation_feature_coverage": val_active,
            "n_train_tokens": int(len(X)), "n_validation_tokens": int(len(V)),
            "warnings": (["Train corpus smaller than SAE latent dictionary"]
                         if len(X) < dictionary else []),
        }
    (out / "sae-registry.json").write_text(json.dumps(registry, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")
    return registry


def export_feature_map(*, registry_path: str, selection_path: str,
                       output_path: str) -> dict:
    """Bind previously frozen graph nodes to trusted SAE checkpoint files.

    Nodes use explicit `sae:<block>:<latent>` identifiers. The function
    refuses absent checkpoint or stage mappings and never reselects nodes.
    """
    registry = _load_json(registry_path)
    selection = _load_json(selection_path)
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("Expected PHB SAE registry")
    if not selection.get("selected_nodes") or not selection.get("graph_sha256"):
        raise ValueError("Expected frozen PercInterp selection manifest")
    provenance = selection.get("provenance", {})
    if provenance.get("model") != registry.get("model"):
        raise ValueError("Graph and SAE registry model IDs differ")
    if provenance.get("tokenizer") is not None and provenance["tokenizer"] != registry.get("tokenizer"):
        raise ValueError("Graph and SAE registry tokenizer IDs differ")
    if set(provenance.get("train_seeds", [])) != set(registry.get("phb_train_seeds", [])) or \
       set(provenance.get("training_probe_ids", [])) != set(registry.get("phb_training_probes", [])):
        raise ValueError("Graph and SAE training seed/probe provenance differs")
    filename = Path(output_path)
    features: dict[str, dict] = {}
    for chosen in selection["selected_nodes"].values():
        for name in chosen:
            match = re.fullmatch(r"sae:(\d+):(\d+)", name)
            if not match:
                raise ValueError(f"Graph node {name!r} must use 'sae:<block>:<latent>' naming")
            block, latent = map(int, match.groups())
            info = registry["blocks"].get(str(block))
            if info is None or latent >= info["n_latents"]:
                raise ValueError(f"Selected feature {name} absent from SAE registry")
            if selection["node_to_decoder_block"].get(name) != block:
                raise ValueError(f"Selected feature {name} disagrees with graph block mapping")
            features[name] = {"block": block, "latent": latent}
    if not features:
        raise ValueError("No SAE features selected")
    blocks = sorted({p["block"] for p in features.values()})
    checkpoints = {}
    for block in blocks:
        info = registry["blocks"][str(block)]
        absolute = (Path(registry_path).parent / info["path"]).resolve()
        if not absolute.is_file() or _hash_file(absolute) != info["sha256"]:
            raise ValueError("SAE registry references an absent or changed checkpoint")
        checkpoints[str(block)] = {
            "path": os.path.relpath(absolute, start=filename.parent.resolve()),
            "top_k": info["top_k"], "sha256": info["sha256"],
        }
    feature_map = {
        "schema": "phb.percinterp.feature_map.v1",
        "model": registry["model"], "tokenizer": registry["tokenizer"],
        "checkpoint_revision": registry["checkpoint_revision"],
        "selection_graph_sha256": selection["graph_sha256"],
        "checkpoints": checkpoints, "features": dict(sorted(features.items())),
    }
    _validate_plan(selection, feature_map, model_name=registry["model"],
                   tokenizer_name=registry["tokenizer"])
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(feature_map, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    return feature_map
