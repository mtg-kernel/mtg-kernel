"""W5 runtime receipt under amendment A1 (collab LANES/lead/w5/RULINGS-r0.md, "Amendment A1" and "A1 correction").

The raw parity checks of `runtime_parity.py` run unchanged and are reported as they are. Exactly three named
differences may then be accepted as declared exceptions, each only under its stated conditions; any other failed
raw check fails the receipt. Raw failures and accepted exceptions are kept in separate fields.

Accepted exceptions (A1):
- `spellbench-neutral-choices`: covers the raw check `neutral_choices_equal`, when the board declares exactly
  `:neutral-v2` and the A1 census result (sha256 below) is supplied.
- `graveyard-order-profile`: covers `training_rules_profile_is_default`, when every training episode uses exactly
  `resolution-boundary-v1`, the board applies no profile, and no card in any training or board deck has Delve
  (the only reader of graveyard order; `data/cards_v1.json` mechanics at each revision).
- `buffered-completion-selection`: covers `selection_equal`, when the board wrapper selects buffered groups by
  `deterministic-completion-logprob/v1` over `mtg-kernel-spellbench-completion-proposals/v1`.

Further raw checks added here (never excepted): the H panels and the S0 recording run the training runtime.

Recorded (A1 condition 1): the amendment (file, sha256, commit); the actual runtimes of training, search (the
w5-xact search runs inside the collection session), H panels and S0; the frozen configs and the pool hash; the
engine, wrapper and scorer identities, with the engine and wrapper bound to the board run's `engine_files`; the
board snapshot, with the run bound to its sources; each exception's evidence.

  runtime_receipt.py --training-config CFG.json [...] --board-run RUN_DIR --board-snapshot SNAPSHOT_DIR
      --repo MTG_KERNEL_REPO [--training-revision REV] --amendment RULINGS-r0.md --amendment-commit SHA
      --census w5-neutral-census.json --h-plan PLAN.json [...] --s0-config CFG.json [...]
      --engine-binary EXE --wrapper kernel_engine_v2.py --scorer EXE --bot kernel_flat_bot.py
      --collab-repo COLLAB_CHECKOUT [--out RECEIPT.json]

Bindings (all must hold): the engine binary and the wrapper are the board run's `agent_bridge_v1.exe` and
`kernel_engine_v2.py` by `engine_files` role; the board run is `pauper-ladder-v2` (a v1 receipt is informative
only); the run is a source of the snapshot; the census is A1's; the amendment file is `RULINGS-r0.md` at the given
collab commit. The raw receipt is recomputed every time and checked against the supplied inputs. H runtimes come from
every match of every pinned H request. The scorer is recorded (the board run records no scorer hash).

**Cube segments** (Cube M2 T12). `--cube-pins` selects a receipt with no board, census, H plans or S0 (all Pauper
arguments are refused with it, and without it nothing above changes):

  runtime_receipt.py --training-config SEGMENT.json [...] --cube-pins PINS.json --cube-pool POOL.json
      --cube-constants CONSTANTS.json --cube-base BASE.json
      [--production --cube-pins-sha256 SIGNED_DIGEST] [--out RECEIPT.json]

The raw cube receipt of `runtime_parity.py` is recomputed and must pass. The segment is then bound to the signed M3
Stage 1 declaration (draft 3, signed text SHA-256 below): the dealer, cube list, gaps and fence_v3 hashes, the 15
fenced and 525 active nonbasic cards (M3.3, M3.4), the A1.3 deal constants with an even `pairs_per_update` in
[4, 60] (M3.3), the M3.5 recipe on the base (`portable-math-v1`, GAE with gamma 1.0, lambda 0.9, entropy 0.0 and
`Gae` pregame credit, learning rate 0.0001, value coefficient 0.5, `binary_v1` trajectories in `zstd1` envelopes, the
default sampler, no teacher, search or KL recovery), the 4,096 MiB batch bound (M3.3) and the M3.6 non-natural
fraction `max(1, floor(0.0625 N)) / N` as an f32 with N = 2 `pairs_per_update`. No requirement may fail.

Identities that only Gate 1 can fix (M3.10; T6, T8 and T9 are not merged) come from the pins' `gate1` block, one
entry per item in `GATE1_ITEMS`, each `{fixture, owner, section, value}`; the value must equal what the inputs
show. An entry with `fixture: true` is reported as `fixture` and listed under `gate1_to_fill`. Production requires
all values filled, the ruled engine windows, a non-placeholder import hash, and `--cube-pins-sha256` from the
signed Gate 1 manifest. Production also checks each resume checkpoint against the signed pins'
`checkpoint_sha256_by_config` list in input order. Fresh configs use null; if omitted, all are fresh. Gate 1 must fill: the compiled `KERNEL_CARDDB_HASH`,
the `cube_card_ids/v1` export SHA-256, the active pool SHA-256, T9's `pairs_per_update` (N / 2) and the
`m3_train_constants.json` SHA-256, T6's FlatV9 initial-source play import, and T8's `card_semantics` block (Arm E).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime_parity as rp  # noqa: E402

SCHEMA = "w5-runtime-receipt-a1/v1"
CENSUS_SHA256 = "811d9af07e8da9365089b30f2e9352336c747b000a502335e22fb3d776ac88e1"
NEUTRAL_ID = ":neutral-v2"
PROFILE = "resolution-boundary-v1"
PROPOSALS = "mtg-kernel-spellbench-completion-proposals/v1"
MAPPING = "deterministic-completion-logprob/v1"
AMENDMENT_MARKERS = ("## Amendment A1: runtime parity", "### A1 correction")
AMENDMENT_PATH = "LANES/lead/w5/RULINGS-r0.md"
BOARD_V2 = "pauper-ladder-v2"
STEPWISE = "sampled-wide-v1"
EXCEPTIONS = {
    "spellbench-neutral-choices": "neutral_choices_equal",
    "graveyard-order-profile": "training_rules_profile_is_default",
    "buffered-completion-selection": "selection_equal",
}


class ReceiptError(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


S0_PLAN_SCHEMA = "w5-s0-record-plan/v1"


def runtime_profile(configs: list[dict]) -> dict[str, int]:
    """Episodes (or S0 recording games) per runtime rules profile. An S0 recording plan (tools/w5/s0.py) declares
    one runtime for all of its games."""
    out = Counter()
    for config in configs:
        if config.get("schema") == S0_PLAN_SCHEMA:
            rules = config["runtime"]["runtime_rules"]
            out["default" if rules is None else str(rules)] += int(config["games"])
            continue
        for ep in rp.episodes(config):
            out["default" if ep.get("runtime_rules") is None else str(ep["runtime_rules"])] += 1
    return dict(out)


def plan_runtime(plan: dict) -> str:
    return "default" if plan.get("runtime_rules") is None else str(plan["runtime_rules"])


def h_plan_runtimes(plan_path: Path) -> list[str]:
    """The runtime of every match an H plan actually executes: each request verified by its pin and read."""
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    runtimes = [plan_runtime(plan)]
    for item in plan.get("requests", []):
        ref = item["request"]
        data = Path(ref["path"]).read_bytes()
        if sha256_bytes(data) != ref["sha256"]:
            raise ReceiptError(f"H request {ref['path']} differs from its pin")
        for match in json.loads(data)["matches"]:
            runtimes.append(plan_runtime(match["config"]))
    if len(runtimes) == 1:
        raise ReceiptError(f"H plan {plan_path} executes no requests")
    return runtimes


def delve_cards(cards: list[dict], deck_ids) -> list[str]:
    return sorted({cards[i]["name"] for i in deck_ids if "delve" in cards[i].get("mechanics", [])})


def module_constants(source: str) -> dict[str, str]:
    """Module-level string constants (`NAME = "..."`), read from the syntax tree, so comments and other strings
    cannot stand in for them."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def selection_facts(wrapper_source: str, bot_source: str) -> dict:
    """How the deployed board selects: the wrapper's buffered groups and the bot's completion mapping."""
    wrapper, bot = module_constants(wrapper_source), module_constants(bot_source)
    return {"buffered_completion_proposals": wrapper.get("PROPOSALS") == PROPOSALS,
            "wrapper_proposals_constant": wrapper.get("PROPOSALS"),
            "bot_reads_completion_proposals": bot.get("PROPOSAL_SCHEMA") == PROPOSALS,
            "bot_completion_mapping": bot.get("PROPOSAL_MAPPING"),
            "bot_stepwise_sampler": bot.get("SAMPLED")}


def evaluate(raw: dict, training_profile: dict, h_runtimes: list[str], s0_profile: dict, board_manifest: dict,
             selection: dict, delve: dict, census_ok: bool) -> dict:
    """Raw checks (as computed) plus A1's three exceptions, each with its conditions."""
    checks = dict(raw["checks"])
    training_runtimes = set(training_profile)
    # Equal only for a known training-style deployment: no buffered groups and the stepwise sampler training uses.
    checks["selection_equal"] = (selection.get("wrapper_proposals_constant") is None
                                 and not selection["buffered_completion_proposals"]
                                 and selection.get("bot_stepwise_sampler") == STEPWISE)
    checks["h_panel_runtime_equals_training"] = bool(h_runtimes) and set(h_runtimes) == training_runtimes
    checks["s0_runtime_equals_training"] = bool(s0_profile) and set(s0_profile) == training_runtimes
    conditions = {
        "spellbench-neutral-choices": {
            "board_declares_neutral_v2": board_manifest["engine"].get("rules_snapshot_id", "").endswith(NEUTRAL_ID),
            "census_evidence_matches": census_ok},
        "graveyard-order-profile": {
            "training_profile_exactly_resolution_boundary": training_runtimes == {PROFILE},
            "board_applies_no_profile": raw["checks"]["board_rules_profile_is_default"],
            "no_delve_card_in_any_deck": not delve["training"] and not delve["board"]},
        "buffered-completion-selection": {
            "wrapper_buffers_with_completion_proposals": selection["buffered_completion_proposals"],
            "bot_reads_completion_proposals": selection["bot_reads_completion_proposals"],
            "bot_maps_by_completion_logprob": selection["bot_completion_mapping"] == MAPPING},
    }
    raw_failures = sorted(k for k, v in checks.items() if not v)
    accepted, refused = [], []
    for name, check in EXCEPTIONS.items():
        if check in raw_failures:
            entry = {"exception": name, "covers": check, "conditions": conditions[name]}
            (accepted if all(conditions[name].values()) else refused).append(entry)
    covered = {e["covers"] for e in accepted}
    unexcepted = [k for k in raw_failures if k not in covered]
    return {"raw_checks": checks, "raw_failures": raw_failures, "accepted_exceptions": accepted,
            "refused_exceptions": refused, "unexcepted_failures": unexcepted}


# ---- Cube segments (T12) -------------------------------------------------------------------------------------

CUBE_SCHEMA = "w5-runtime-receipt-cube/v1"
M3_DECLARATION = {
    "path": "LANES/lead/reviews/cube-briefs-20261122/decl-20261124/m3/M3-STAGE1-DECLARATION-draft3.md",
    "signed_text_sha256": "31fde858a7c02d6aef6f7829e1138e1d71ffefd4b12c810ef68f94fa6447abbf",
}
M3_DEALER_SHA256 = "0e60d49f40eb9509210f4e2f71e8d2cd7dac65e393696427043978556e5af497"  # M3.3 item 1.1
M3_POOL_INPUTS = {  # M3.4 item 1: the frozen cube list and gaps (A1.2) and the active fence_v3
    "cube_list": "e647de154da68a25caf5c97f022a9466311b2bf9d604143f35ceef9ac6aaf02d",
    "gaps": "efad86376670270180b2ada893958b0065a57e8a0905565af1aae4e72a188cdd",
    "fence": "8d02e845f1be12d636d4e6b3f1347b4d42f002e6c7ce67ce339114bcf6775e5c",
}
M3_POOL_COUNTS = {"fenced": 15, "active_nonbasic": 525, "basics": 5}  # M3.4 item 1
M3_CONSTANTS = {"deck_size": 40, "colours_per_seat": 2, "nonland_spells": 23, "nonbasic_lands": 3,
                "basic_split": "even_first_colour_extra"}  # A1.3, unchanged by M3.3 item 1.2
M3_PAIRS_PER_UPDATE = (4, 60)  # M3.3 item 2: an even pairs_per_update in [4, 60]
M3_RECIPE = {"training_math": "portable-math-v1", "trajectory_format": "binary_v1", "record_envelope": "zstd1",
             "max_trajectory_batch_mebibytes": 4096}  # M3.5 item 1 and M3.3 item 2
M3_RECIPE_F32 = {"learning_rate": 0.0001, "value_coefficient": 0.5}  # M3.5 item 1, compared as binary32
M3_LOSS = {"kind": "gae_advantage_value_v1", "gamma": 1.0, "lambda": 0.9, "entropy_coefficient": 0.0}
M3_ABSENT_RUN_KEYS = ("update_backend", "line_b", "e_supported", "soft_q", "search_targets", "ex1")
GATE1_ITEMS = ("kernel_carddb_hash", "id_export_sha256", "pool_sha256", "pairs_per_update", "constants_sha256",
               "flatv9_play_import_sha256", "card_semantics")


def f32_bits(value) -> int | None:
    """The IEEE-754 binary32 bits a JSON number stores as the trainer's f32; None for a non-number."""
    if type(value) not in (int, float):
        return None
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except (OverflowError, struct.error):
        return None


def requirement(rid: str, section: str, required, observed, holds: bool) -> dict:
    return {"id": rid, "section": section, "required": required, "observed": observed,
            "status": "holds" if holds else "fails"}


def non_natural_fraction(pairs_per_update) -> float | None:
    """M3.6: max(1, floor(0.0625 N)) / N with N = 2 pairs_per_update."""
    if type(pairs_per_update) is not int or pairs_per_update < 1:
        return None
    n = 2 * pairs_per_update
    return max(1, math.floor(0.0625 * n)) / n


def gate1_observed(raw: dict, pool: dict, constants: dict, base: dict) -> dict:
    source = base.get("initial_source")
    play_import = source.get("play_import") if isinstance(source, dict) else None
    return {"kernel_carddb_hash": pool.get("kernel_carddb_hash"),
            "id_export_sha256": (pool.get("input_sha256") or {}).get("id_export"),
            "pool_sha256": raw["inputs"]["pool_sha256"],
            "pairs_per_update": constants.get("pairs_per_update"),
            "constants_sha256": raw["inputs"]["constants_sha256"],
            "flatv9_play_import_sha256": play_import.get("sha256") if isinstance(play_import, dict) else None,
            "card_semantics": base.get("card_semantics")}


def cube_requirements(raw: dict, pins: dict, pool: dict, constants: dict, base: dict) -> list[dict]:
    """The signed M3 Stage 1 identities (holds or fails) and the Gate 1 items (holds, fixture or fails)."""
    inputs = pool.get("input_sha256") or {}
    counts = {"fenced": len(pool.get("fenced", [])),
              "active_nonbasic": len(pool.get("spells", [])) + len(pool.get("nonbasic_lands", [])),
              "basics": len(pool.get("basics", []))}
    pairs = constants.get("pairs_per_update")
    loss = base.get("loss_selection")
    loss_holds = isinstance(loss, dict) and set(loss) - {"pregame_advantage"} == set(M3_LOSS) \
        and loss.get("pregame_advantage", "gae") == "gae" and loss["kind"] == M3_LOSS["kind"] \
        and all(f32_bits(loss[k]) == f32_bits(M3_LOSS[k]) for k in ("gamma", "lambda", "entropy_coefficient"))
    fraction = non_natural_fraction(pairs)
    out = [
        requirement("raw_cube_parity", "M3.3 item 1 (A1.4 dealt pairs), M3.4 item 1", True, raw["pass"], raw["pass"]),
        requirement("dealer_sha256", "M3.3 item 1.1", M3_DEALER_SHA256, raw["inputs"]["dealer_sha256"],
                    raw["inputs"]["dealer_sha256"] == M3_DEALER_SHA256),
        *(requirement(f"pool_input.{k}", "M3.4 item 1", v, inputs.get(k), inputs.get(k) == v)
          for k, v in M3_POOL_INPUTS.items()),
        requirement("pool_counts", "M3.4 item 1", M3_POOL_COUNTS, counts, counts == M3_POOL_COUNTS),
        requirement("deal_constants", "M3.3 item 1.2 (A1.3)", M3_CONSTANTS,
                    {k: constants.get(k) for k in M3_CONSTANTS},
                    all(rp.same_json(constants.get(k), v) for k, v in M3_CONSTANTS.items())),
        requirement("pairs_per_update_even_in_range", "M3.3 item 2", list(M3_PAIRS_PER_UPDATE), pairs,
                    type(pairs) is int and pairs % 2 == 0 and M3_PAIRS_PER_UPDATE[0] <= pairs <= M3_PAIRS_PER_UPDATE[1]),
        *(requirement(f"recipe.{k}", "M3.5 item 1" if k != "max_trajectory_batch_mebibytes" else "M3.3 item 2",
                      v, base.get(k), rp.same_json(base.get(k), v)) for k, v in M3_RECIPE.items()),
        *(requirement(f"recipe.{k}", "M3.5 item 1", v, base.get(k),
                      k in base and f32_bits(base[k]) == f32_bits(v)) for k, v in M3_RECIPE_F32.items()),
        requirement("recipe.loss_selection", "M3.5 items 1 and 2", M3_LOSS, loss, loss_holds),
        requirement("recipe.collection_sampler", "M3.5 item 1", "legacy (absent)", base.get("collection_sampler"),
                    base.get("collection_sampler", "legacy") == "legacy"),
        requirement("recipe.absent_paths", "M3.5 item 1", [], sorted(k for k in M3_ABSENT_RUN_KEYS if k in base),
                    not any(k in base for k in M3_ABSENT_RUN_KEYS)),
        requirement("max_non_natural_episode_fraction", "M3.6", fraction, base.get("max_non_natural_episode_fraction"),
                    fraction is not None and f32_bits(base.get("max_non_natural_episode_fraction")) == f32_bits(fraction)),
    ]
    block = pins.get("gate1")
    if not isinstance(block, dict) or set(block) != set(GATE1_ITEMS):
        raise ReceiptError(f"the cube pins' gate1 block must hold exactly {list(GATE1_ITEMS)}")
    observed = gate1_observed(raw, pool, constants, base)
    for item in GATE1_ITEMS:
        entry = block[item]
        if not isinstance(entry, dict) or type(entry.get("fixture")) is not bool or "value" not in entry:
            raise ReceiptError(f"gate1.{item} must hold a boolean fixture and a value")
        equal = rp.same_json(entry["value"], observed[item])
        out.append({"id": f"gate1.{item}", "section": entry.get("section"), "owner": entry.get("owner"),
                    "required": entry["value"], "observed": observed[item],
                    "status": "fails" if not equal else "fixture" if entry["fixture"] else "holds"})
    return out


PAUPER_ARGUMENTS = ("board_run", "board_snapshot", "repo", "amendment", "amendment_commit", "census", "h_plan",
                    "s0_config", "engine_binary", "wrapper", "scorer", "bot", "collab_repo")


def cube_main(args, parser) -> int:
    given = [f"--{name.replace('_', '-')}" for name in PAUPER_ARGUMENTS if getattr(args, name) is not None]
    if args.training_revision != "HEAD":
        given.append("--training-revision")
    if given:
        parser.error(f"{', '.join(given)} not allowed with --cube-pins")
    missing = [flag for flag, value in (("--cube-pool", args.cube_pool), ("--cube-constants", args.cube_constants),
                                        ("--cube-base", args.cube_base)) if value is None]
    if missing:
        parser.error(f"--cube-pins requires {', '.join(missing)}")
    try:
        raw_receipt_path = (args.out or Path("runtime-parity-raw.json")).with_suffix(".raw.json")
        raw_receipt_path.unlink(missing_ok=True)  # never reuse an earlier raw receipt
        inputs = {"pins": args.cube_pins, "pool": args.cube_pool, "constants": args.cube_constants,
                  "base": args.cube_base}
        rp_args = [x for p in args.training_config for x in ("--training-config", str(p))]
        rp_args += [x for name, path in inputs.items() for x in (f"--cube-{name}", str(path))]
        with contextlib.redirect_stdout(io.StringIO()):  # the raw receipt goes to its own file
            code = rp.main([*rp_args, "--out", str(raw_receipt_path)])
        if not raw_receipt_path.exists():
            raise ReceiptError(f"the raw cube parity receipt could not be computed (exit {code})")
        raw = json.loads(raw_receipt_path.read_text(encoding="utf-8"))
        data = {name: path.read_bytes() for name, path in inputs.items()}
        supplied = [sha256_bytes(p.read_bytes()) for p in args.training_config]
        if raw.get("schema") != rp.CUBE_SCHEMA \
                or [c["sha256"] for c in raw["inputs"]["training_configs"]] != supplied \
                or any(raw["inputs"][f"{name}_sha256"] != sha256_bytes(blob) for name, blob in data.items()):
            raise ReceiptError("the raw receipt was computed from other inputs")
        docs = {name: json.loads(blob) for name, blob in data.items()}
        requirements = cube_requirements(raw, docs["pins"], docs["pool"], docs["constants"], docs["base"])
    except (ReceiptError, rp.ParityError, KeyError, ValueError, TypeError, AttributeError) as exc:
        print(f"error: {exc!r}", file=sys.stderr)
        return 1
    fixtures = [r["id"] for r in requirements if r["status"] == "fixture"]
    if args.production or args.cube_pins_sha256 is not None:
        trusted = args.cube_pins_sha256
        import_sha = gate1_observed(raw, docs["pool"], docs["constants"], docs["base"])["flatv9_play_import_sha256"]
        windows = docs["pins"]["episode"].get("priority_windows")
        checkpoints = []
        for fields in raw["per_segment_fields"]:
            source = fields.get("initial_source")
            checkpoint = source.get("checkpoint") if isinstance(source, dict) else None
            checkpoints.append(checkpoint.get("sha256") if isinstance(checkpoint, dict) else None)
        expected_checkpoints = docs["pins"].get("checkpoint_sha256_by_config", [None] * len(args.training_config))
        requirements.extend([
            requirement("gate1.external_pins_sha256", "M3.10", trusted, raw["inputs"]["pins_sha256"],
                        rp.is_sha256(trusted) and trusted == raw["inputs"]["pins_sha256"]),
            requirement("gate1.non_placeholder_play_import", "M3.4 and M3.10", "nonzero SHA-256", import_sha,
                        rp.is_sha256(import_sha) and import_sha != "0" * 64),
            requirement("gate1.segment_checkpoints", "M3.5 and M3.10", expected_checkpoints, checkpoints,
                        rp.same_json(checkpoints, expected_checkpoints)),
            requirement("recipe.engine_windows", "M3.4 and DECISIONS-2", ["engine-windows-v1"] * 2, windows,
                        rp.same_json(windows, ["engine-windows-v1"] * 2)),
        ])
    failing = [r["id"] for r in requirements if r["status"] == "fails"]
    production = args.cube_pins_sha256 is not None and not fixtures and not failing
    receipt = {
        "schema": CUBE_SCHEMA,
        "pass": raw["pass"] and not failing and (not args.production or production),
        "production": production, "production_required": args.production,
        "raw_failures": sorted(k for k, v in raw["checks"].items() if not v),
        "failing_requirements": failing, "gate1_to_fill": fixtures,
        "declaration": M3_DECLARATION, "segment": raw["schedule"]["segment"], "requirements": requirements,
        "raw_receipt": {"path": str(raw_receipt_path), "sha256": sha256_bytes(raw_receipt_path.read_bytes())},
        "inputs": raw["inputs"],
    }
    text = json.dumps(receipt, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
    return 0 if receipt["pass"] else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    cube = any(arg == "--cube-pins" or arg.startswith("--cube-pins=") for arg in argv)
    pauper = not cube
    description = __doc__ if cube else __doc__.split("\n\n**Cube segments**", 1)[0]
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training-config", type=Path, action="append", required=True)
    parser.add_argument("--board-run", type=Path, required=pauper)
    parser.add_argument("--board-snapshot", type=Path, required=pauper)
    parser.add_argument("--repo", type=Path, required=pauper)
    parser.add_argument("--training-revision", default="HEAD")
    parser.add_argument("--amendment", type=Path, required=pauper)
    parser.add_argument("--amendment-commit", required=pauper)
    parser.add_argument("--census", type=Path, required=pauper)
    parser.add_argument("--h-plan", type=Path, action="append", required=pauper)
    parser.add_argument("--s0-config", type=Path, action="append", required=pauper)
    parser.add_argument("--engine-binary", type=Path, required=pauper)
    parser.add_argument("--wrapper", type=Path, required=pauper)
    parser.add_argument("--scorer", type=Path, required=pauper)
    parser.add_argument("--bot", type=Path, required=pauper, help="the deployed bot (kernel_flat_bot.py)")
    parser.add_argument("--collab-repo", type=Path, required=pauper, help="a collab checkout holding the amendment")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cube-pins", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cube-pool", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cube-constants", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cube-base", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--production", action="store_true", help=argparse.SUPPRESS)  # see "Cube segments"
    parser.add_argument("--cube-pins-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cube_pins is not None:
        return cube_main(args, parser)
    if args.production or any(v is not None for v in (args.cube_pool, args.cube_constants, args.cube_base,
                                                       args.cube_pins_sha256)):
        parser.error("Cube arguments require --cube-pins")
    try:
        raw_receipt_path = (args.out or Path("runtime-parity-raw.json")).with_suffix(".raw.json")
        raw_receipt_path.unlink(missing_ok=True)  # never reuse an earlier raw receipt
        config_args = [x for p in args.training_config for x in ("--training-config", str(p))]
        with contextlib.redirect_stdout(io.StringIO()):  # the raw receipt goes to its own file
            code = rp.main([*config_args, "--board-run", str(args.board_run), "--repo", str(args.repo),
                            "--training-revision", args.training_revision, "--out", str(raw_receipt_path)])
        if not raw_receipt_path.exists():
            raise ReceiptError(f"the raw parity receipt could not be computed (exit {code})")
        raw = json.loads(raw_receipt_path.read_text(encoding="utf-8"))
        supplied = [sha256_bytes(p.read_bytes()) for p in args.training_config]
        if ([c["sha256"] for c in raw["inputs"]["training_configs"]] != supplied
                or raw["inputs"]["board_manifest_sha256"] != sha256_bytes((args.board_run / "manifest.json").read_bytes())):
            raise ReceiptError("the raw receipt was computed from other inputs")
        amendment = args.amendment.read_bytes()
        amendment_text = amendment.decode("utf-8")
        missing = [m for m in AMENDMENT_MARKERS if m not in amendment_text]
        if missing:
            raise ReceiptError(f"the amendment file lacks {missing}")
        # The committed bytes against the supplied bytes, line endings normalized explicitly (a CRLF working copy
        # of the committed LF file matches); no checkout filter takes part.
        committed = subprocess.run(["git", "-C", str(args.collab_repo), "show",
                                    f"{args.amendment_commit}:{AMENDMENT_PATH}"], capture_output=True)
        if committed.returncode != 0 or (committed.stdout.replace(b"\r\n", b"\n")
                                         != amendment.replace(b"\r\n", b"\n")):
            raise ReceiptError(f"the amendment file is not {AMENDMENT_PATH} at commit {args.amendment_commit}")
        census_sha = sha256_bytes(args.census.read_bytes())
        configs = [json.loads(p.read_bytes()) for p in args.training_config]
        s0_configs = [json.loads(p.read_bytes()) for p in args.s0_config]
        h_plans = [json.loads(p.read_bytes()) for p in args.h_plan]
        h_runtimes = [r for p in args.h_plan for r in h_plan_runtimes(p)]
        bot_raw = args.bot.read_bytes()
        manifest_raw = (args.board_run / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
        snapshot_raw = (args.board_snapshot / "manifest.json").read_bytes()
        snapshot = json.loads(snapshot_raw)
        wrapper_raw = args.wrapper.read_bytes()
        engine_sha, wrapper_sha, scorer_sha = (sha256_bytes(p.read_bytes()) for p in
                                               (args.engine_binary, args.wrapper, args.scorer))
        engine_files = {f.get("file_name"): f["sha256"] for f in manifest.get("engine_files", [])}
        training_rev = raw["inputs"]["training_revision"]
        board_rev = raw["inputs"]["board_revision"]
        cards_t = json.loads(rp.git_show(args.repo, training_rev, "data/cards_v1.json"))["cards"]
        cards_b = json.loads(rp.git_show(args.repo, board_rev, "data/cards_v1.json"))["cards"]
        decks_b = json.loads(rp.git_show(args.repo, board_rev, "data/runtime_decks_v1.json"))
        pool = [d["catalog_id"] for d in json.loads((args.board_run / "config.json").read_bytes())["deck_pool"]]
        training_ids = [i for c in configs for ep in rp.episodes(c) for d in ep["selected"] for i in d["mainboard"]]
        board_ids = [r["card_id"] for d in decks_b["decks"] if d["id"] in pool for r in d["materialized_mainboard"]]
        delve = {"training": delve_cards(cards_t, training_ids), "board": delve_cards(cards_b, board_ids)}
        training_profile = runtime_profile(configs)
        selection = selection_facts(wrapper_raw.decode("utf-8"), bot_raw.decode("utf-8"))
        result = evaluate(raw, training_profile, h_runtimes, runtime_profile(s0_configs), manifest, selection, delve,
                          census_sha == CENSUS_SHA256)
    except (ReceiptError, rp.ParityError, KeyError, ValueError, SyntaxError) as exc:
        print(f"error: {exc!r}", file=sys.stderr)
        return 1
    bindings = {
        "engine_binary_is_the_board_bridge": engine_files.get("agent_bridge_v1.exe", engine_files.get("agent_bridge_v1"))
        == engine_sha,
        "wrapper_is_the_board_wrapper": engine_files.get("kernel_engine_v2.py") == wrapper_sha,
        "board_is_v2": manifest.get("run", {}).get("benchmark_id") == BOARD_V2,
        "run_is_a_snapshot_source": sha256_bytes(manifest_raw) in {s.get("manifest_sha256")
                                                                   for s in snapshot.get("sources", [])},
        "census_hash_is_a1s": census_sha == CENSUS_SHA256,
    }
    pool_hash = sha256_bytes(json.dumps(sorted({d["label"]: d["mainboard"] for c in configs for ep in rp.episodes(c)
                                                for d in ep["selected"]}.items()),
                                        separators=(",", ":")).encode())
    search = [{"config": str(p), "ex1": c.get("ex1")} for p, c in zip(args.training_config, configs)]
    receipt = {
        "schema": SCHEMA,
        "pass": not result["unexcepted_failures"] and not result["refused_exceptions"] and all(bindings.values()),
        **result, "bindings": bindings,
        "amendment": {"path": str(args.amendment), "sha256": sha256_bytes(amendment), "commit": args.amendment_commit,
                      "sections": list(AMENDMENT_MARKERS)},
        "runtimes": {
            "training": training_profile,
            "search": {"runs_in": "the collection session (w5-xact-v1 hook), so it shares the training runtime",
                       "configs": search},
            "h_panels": [{"plan": str(p), "sha256": sha256_bytes(p.read_bytes()), "runtime_rules": plan_runtime(h),
                          "seed_namespace": h.get("seed_namespace"),
                          "match_runtimes": dict(Counter(h_plan_runtimes(p)[1:]))}
                         for p, h in zip(args.h_plan, h_plans)],
            "s0": runtime_profile(s0_configs)},
        "configs": {"training": [{"path": str(p), "sha256": sha256_bytes(p.read_bytes())} for p in args.training_config],
                    "s0": [{"path": str(p), "sha256": sha256_bytes(p.read_bytes())} for p in args.s0_config],
                    "training_pool_sha256": pool_hash,
                    "board_runtime_decks_sha256": raw["inputs"]["board_runtime_decks_sha256"]},
        "identities": {"engine_binary": {"path": str(args.engine_binary), "sha256": engine_sha},
                       "wrapper": {"path": str(args.wrapper), "sha256": wrapper_sha},
                       "scorer": {"path": str(args.scorer), "sha256": scorer_sha,
                                  "binding": "recorded only: the board run records no scorer hash"},
                       "bot": {"path": str(args.bot), "sha256": sha256_bytes(bot_raw),
                               "binding": "recorded only: the board run records no bot hash"},
                       "board_engine": manifest.get("engine"), "board_engine_files": manifest.get("engine_files")},
        "board_snapshot": {"path": str(args.board_snapshot), "manifest_sha256": sha256_bytes(snapshot_raw),
                           "run_manifest_sha256": sha256_bytes(manifest_raw)},
        "exception_evidence": {"census": {"path": str(args.census), "sha256": census_sha},
                               "delve_cards": delve, "selection": selection},
        "raw_receipt": {"path": str(raw_receipt_path), "sha256": sha256_bytes(raw_receipt_path.read_bytes())},
        "limitation": ("A1 condition 3: whether the mismatch affects the arms differently is unmeasured; the Elves and "
                       "Spy partition decisions are plausibly outcome-equivalent, unverified."),
    }
    text = json.dumps(receipt, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
