# v4 HyperTower Planning Document

## Goals
Rebuild the orchestrator using `run_ntower_cv` as the architectural foundation, with four key improvements: JSON config, decoupled data sources, declarative tower lists, and a cross-tower communication protocol.

---

## 1. JSON Config (replace argparse)

The orchestrator receives a single JSON config file. It has no hardcoded knowledge of what args individual towers or data modules need — it just forwards the relevant subtrees.

```json
{
  "run_name": "v4/ensemble_fused",
  "eval_mode": "binary",
  "epochs": 30,
  "fusion_epochs": 10,
  "fold_seed": 100,
  "seed": 1234,
  "data": {
    "module": "v4.papila.v4papila",
    "args": {
      "iop_corr_method": "ratio",
      "iop_drop_raw": true,
      "exclude_cols": ["Axial_Length"]
    }
  },
  "towers": [
    {
      "name": "img",
      "module": "v3.classes.image_towers",
      "class": "ImageEncoder",
      "args": { "backbone": "refugelike", "freeze_ratio": 0.5, "augment": true },
      "warmup_epochs": 0
    },
    {
      "name": "cd",
      "module": "v3.classes.clinical_towers",
      "class": "ClinicalEncoder",
      "args": { "hidden_dim": 128 },
      "warmup_epochs": 40
    }
  ],
  "bridge": {
    "mode": "embedding_mlp",
    "fusion_dim": 256,
    "hidden_dim": 256
  },
  "training": {
    "lr": 1e-4,
    "batch_size": 16,
    "bcd_prob": 0.5,
    "warmup_tower_epochs": 3,
    "warmup_fused_epochs": 3
  }
}
```

The orchestrator loads this with `json.load`, then calls `importlib.import_module(cfg["data"]["module"]).build_data(cfg["data"]["args"])` and similarly instantiates towers. No argparse anywhere in the orchestrator.

---

## 2. Decoupled Data Sources

`v3/classes/papila_builders.py` → clone to `v4/papila/v4papila.py`.

Merge in the relevant logic from `papila_data.py` (preprocessing, feature typing, IOP correction, etc.) so `v4papila.py` is self-contained.

Contract: every data module must expose:
```python
def build_data(args: dict) -> DataBundle:
    ...
```
The orchestrator calls `build_data` and gets back a `DataBundle`. It knows nothing else about the data source. Future modules (e.g. `v4/eyepacs/eyepacs_data.py`) just implement the same function.

---

## 3. Declarative Tower List

Towers are loaded from the `"towers"` list in the JSON and stored as an ordered dict keyed by `name`. The orchestrator never imports a tower class directly.

```python
towers = {}
for t_cfg in cfg["towers"]:
    mod = importlib.import_module(t_cfg["module"])
    cls = getattr(mod, t_cfg["class"])
    # some tower constructors need data (e.g. ClinicalEncoder needs feature_dim)
    # pass data as an optional kwarg; tower ignores it if not needed
    towers[t_cfg["name"]] = cls(data=data, **t_cfg["args"])
```

Tower-specific training metadata (warmup epochs, batch key) lives entirely in the JSON, not in the orchestrator.

---

## 4. Cross-Tower Communication: `early_pass` Protocol

**Problem:** GeometryTower needs to precompute segmentation maps from images, then inject them into other towers' sample dicts before loaders are built. This is currently done imperatively in the orchestrator.

**Proposed solution: `early_pass` connector interface**

Each tower optionally implements:
```python
class TowerBase:
    def early_pass(self, context: EarlyPassContext) -> None:
        """Called once per fold before loaders are built.
        Tower can read from / write to shared context."""
        pass
```

`EarlyPassContext` is a shared mutable object passed to all towers in order:
```python
@dataclass
class EarlyPassContext:
    eye_train: list[dict]
    bilat_train: list[dict]
    bilat_val: list[dict]
    bilat_test: list[dict]
    image_preprocessor: object
    image_cache: object
    device: torch.device
    store: dict = field(default_factory=dict)  # cross-tower key-value store
```

Example: GeometryTower's `early_pass` computes seg maps and injects them into the sample dicts directly (modifying `eye_train` etc. in place), exactly as it does today — but now the orchestrator just calls:
```python
for tower in towers.values():
    tower.early_pass(context)
```

The cross-talk case the user described (img_tower outputs geometry → cd_tower reads it) uses `context.store`:
```python
# ImageTower.early_pass:
context.store["geometry_maps"] = self._compute_geometry(context)

# ClinicalTower.early_pass:
geo = context.store.get("geometry_maps")
if geo is not None:
    self._inject_geometry(context, geo)
```

Tower ordering in the JSON list determines execution order, so dependencies are declared implicitly. If a tower has no `early_pass`, the default no-op in `TowerBase` is used.

**Alternative considered:** explicit dependency graph / DAG execution. Rejected for now — JSON ordering is simpler and sufficient for current needs. Can revisit if cross-tower dependencies become non-linear.

---

## 5. File Layout

```
v4/
  hypertower/
    v4_hypertower.py      # orchestrator (no argparse, no tower imports)
    split_manager.py      # copy/adapt from v3 (or just import)
  papila/
    v4papila.py           # merged papila_builders + papila_data
  configs/
    ensemble_fused.json   # example config
```

Existing `v3/classes/` tower implementations are reused directly — no duplication needed since they're importable by the JSON `"module"` field.

---

## 6. Open Questions / Decisions Needed

- **DataBundle API**: Does `build_data` need to return anything beyond the current `DataBundle`? Or should `DataBundle` grow a `profile` factory method?
- **Per-tower batch_key convention**: Currently `EYE_KEY_MAP = {"img": "image_1", "cd": "matrix_1"}` is hardcoded. Should this be declared in the tower JSON config or inferred from tower type?
- **cd_warmup loader**: Slot stripping (`if k != "image_1"`) is currently img-tower-aware. Under the new design, each tower should declare which slots it needs for warmup vs full training, so the orchestrator can build the right loader without knowing about `image_1`.
- **Geometry injection today vs `early_pass`**: Geometry currently mutates sample dicts; `early_pass` formalizes this. Needs a migration plan for existing GeometryTower.

---

## Implementation Order (once ntower_cv is validated)

1. Write `v4papila.py` (merge papila_builders + papila_data, expose `build_data(args)`)
2. Add `early_pass(context)` no-op to `TowerBase`; implement in `GeometryTower`
3. Write `v4_hypertower.py` orchestrator using JSON config + importlib tower loading
4. Port one config (ensemble_fused) end-to-end and compare outputs against ntower_cv
