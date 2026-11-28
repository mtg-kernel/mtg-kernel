"""Cube M2 T12: the runtime parity and receipt tools accept a correctly pinned Cube segment dealt by the T11 dealer,
reject incompatible deck, schedule and receipt identities, and leave the Pauper mode as it was.

Fixtures (tools/w5/tests/fixtures/cube): the census-r2 fence_v3 pool, A1.3 constants with T9's planning
`pairs_per_update` 8, the smallest Cube base config with the M3.5 recipe, and pins whose `gate1` block marks every
identity that only Gate 1 can fix as a fixture."""
import contextlib
import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import runtime_parity as rp  # noqa: E402
import runtime_receipt as rr  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cube"
NAMES = {"pins": "pins.json", "pool": "pool-v3.json", "constants": "constants.json", "base": "base.json"}
DEALER = rp.load_dealer()


def load(name):
    return json.loads((FIXTURES / NAMES[name]).read_text(encoding="utf-8"))


def dump(doc) -> bytes:
    return (json.dumps(doc, indent=2) + "\n").encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def segment(pins=None, **overrides):
    """The segment the pins declare (optionally for another segment block), as the chain would write it."""
    pins = copy.deepcopy(pins or load("pins"))
    pins["segment"].update(overrides)
    return rp.cube_segment_config(DEALER, load("pool"), load("constants"), load("base"), pins)


class CubeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.docs = {name: load(name) for name in NAMES}

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, configs, docs=None, repin=("pool", "constants", "base")):
        """Write the configs and inputs; the pins name the written hashes of the inputs in `repin` (and the gate1
        values follow them), so only the change under test differs."""
        docs = copy.deepcopy(docs or self.docs)
        blobs = {name: dump(docs[name]) for name in ("pool", "constants", "base")}
        pins = docs["pins"]
        for name in repin:
            pins[name]["sha256"] = sha(blobs[name])
        pins["gate1"]["pool_sha256"]["value"] = pins["pool"]["sha256"]
        pins["gate1"]["constants_sha256"]["value"] = pins["constants"]["sha256"]
        blobs["pins"] = dump(docs["pins"])
        paths = {}
        for name, blob in blobs.items():
            paths[name] = self.dir / NAMES[name]
            paths[name].write_bytes(blob)
        args = []
        for i, config in enumerate(configs):
            path = self.dir / f"segment-{i}.json"
            path.write_bytes(dump(config))
            args += ["--training-config", str(path)]
        for name in NAMES:
            args += [f"--cube-{name}", str(paths[name])]
        return args

    def parity(self, args):
        out = self.dir / "parity.json"
        out.unlink(missing_ok=True)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = rp.main([*args, "--out", str(out)])
        return code, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)

    def receipt(self, args, *extra):
        out = self.dir / "receipt.json"
        out.unlink(missing_ok=True)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = rr.main([*args, *extra, "--out", str(out)])
        return code, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)

    def failed(self, receipt):
        return sorted(k for k, v in receipt["checks"].items() if not v)


class AcceptTests(CubeCase):
    def test_committed_pins_name_the_committed_fixtures_and_the_signed_dealer(self):
        pins = self.docs["pins"]
        for name in ("pool", "constants", "base"):
            self.assertEqual(pins[name]["sha256"], sha((FIXTURES / NAMES[name]).read_bytes()), name)
        self.assertEqual(pins["dealer"]["sha256"], rr.M3_DEALER_SHA256)
        self.assertEqual(sha(rp.CUBE_DEALER.read_bytes()), rr.M3_DEALER_SHA256)

    def test_cli_accepts_the_pinned_segment_and_lists_what_gate_1_fills(self):
        path = self.dir / "segment.json"
        path.write_bytes(dump(segment()))
        args = ["--training-config", str(path)] + [x for name in NAMES for x in (f"--cube-{name}",
                                                                                 str(FIXTURES / NAMES[name]))]
        for tool, extra, code in (("runtime_parity.py", [], 0), ("runtime_receipt.py", [], 0),
                                  ("runtime_receipt.py", ["--production"], 1)):
            out = self.dir / f"{tool}.{len(extra)}.json"
            done = subprocess.run([sys.executable, str(HERE / tool), *args, *extra, "--out", str(out)],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, code, (tool, extra, done.stderr))
            receipt = json.loads(out.read_text(encoding="utf-8"))
            if tool == "runtime_parity.py":
                self.assertEqual(receipt["schema"], rp.CUBE_SCHEMA)
                self.assertTrue(all(receipt["checks"].values()), receipt["checks"])
                self.assertEqual(receipt["decks"]["episodes"], 32)
                self.assertIsNone(receipt["schedule"]["first_difference"])
            else:
                self.assertEqual(receipt["schema"], rr.CUBE_SCHEMA)
                expected = ["gate1.external_pins_sha256", "gate1.non_placeholder_play_import",
                            "gate1.model_transfer", "recipe.episode_profile", "recipe.prepared_update"] if extra else []
                self.assertEqual(receipt["failing_requirements"], expected)
                self.assertFalse(receipt["production"])
                self.assertEqual(receipt["gate1_to_fill"], [f"gate1.{item}" for item in rr.GATE1_ITEMS])
                self.assertEqual(receipt["pass"], not extra)

    def test_every_dealt_deck_is_a_forty_card_disjoint_cube_deck_with_both_opt_ins(self):
        config = segment()
        episodes = list(rp.episodes(config))
        self.assertEqual(len(episodes), 2 * 2 * 8)
        for ep in episodes:
            self.assertEqual((ep["deck_format"], ep["cube_flat_action_contract"]), ("limited40-v1", True))
            self.assertEqual([len(d["mainboard"]) for d in ep["registered"]], [40, 40])
        self.assertEqual(rp.cube_decks([config], self.docs["pool"], 40)["violating_episodes"], 0)

    def test_a_segment_split_across_configs_and_per_segment_fields_are_accepted(self):
        whole = segment()
        first, second = copy.deepcopy(whole), copy.deepcopy(whole)
        first["iterations"], second["iterations"] = whole["iterations"][:1], whole["iterations"][1:]
        second["output_directory"] = "segment-1"
        second["initial_source"]["play_import"]["path"] = "staged-import.json"
        second["initial_source"]["checkpoint"] = {"path": "prior-update.json", "sha256": "1" * 64}
        code, receipt = self.parity(self.write([first, second]))
        self.assertEqual(code, 0, receipt and self.failed(receipt))
        self.assertEqual(receipt["per_segment_fields"][1]["output_directory"], "segment-1")

    def production_inputs(self):
        docs = copy.deepcopy(self.docs)
        source = docs["base"]["initial_source"]
        source["play_import"]["sha256"] = "3" * 64
        source["feature_transfer"] = "flat-v9-test-profile"
        docs["base"]["collection_workers"] = 2
        docs["base"]["preparation_workers"] = 2
        docs["base"]["card_semantics"] = {"table_path": "t.json", "table_sha256": "2" * 64}
        docs["base"]["iterations"] = [{"episodes": [{"episode": {
            "opening": "london-v1", "mulligan_modes": ["learned", "learned"],
            "max_physical_decisions": 16384, "max_policy_steps": 32768,
        }}]}]
        for entry in docs["pins"]["gate1"].values():
            entry["fixture"] = False
        for item, value in (
            ("card_semantics", docs["base"]["card_semantics"]),
            ("flatv9_play_import_sha256", source["play_import"]["sha256"]),
            ("flatv9_feature_transfer", source["feature_transfer"]),
        ):
            docs["pins"]["gate1"][item]["value"] = value
        docs["pins"]["episode"].update(priority_windows=["engine-windows-v1"] * 2, max_stack_objects=128)
        return docs

    def test_production_requires_native_admission_of_the_pinned_segment(self):
        docs = self.production_inputs()
        config = rp.cube_segment_config(DEALER, docs["pool"], docs["constants"], docs["base"], docs["pins"])
        args = self.write([config], docs)
        trusted_digest = sha((self.dir / NAMES["pins"]).read_bytes())
        code, receipt = self.receipt(args, "--production", "--cube-pins-sha256", trusted_digest)
        self.assertEqual((code, receipt["failing_requirements"]), (1, ["gate1.native_config_admission"]))
        config["initial_source"]["checkpoint"] = {"path": "other.json", "sha256": "4" * 64}
        code, receipt = self.receipt(self.write([config], docs), "--production", "--cube-pins-sha256", trusted_digest)
        self.assertEqual(code, 1)
        self.assertIn("raw_cube_parity", receipt["failing_requirements"])
        self.assertIn("gate1.segment_checkpoints", receipt["failing_requirements"])

    def test_native_validation_must_bind_exact_config_bytes_and_carddb(self):
        trainer = self.dir / "pinned-trainer"
        trainer.write_bytes(b"trainer fixture")
        docs = self.production_inputs()
        docs["pins"]["trainer_sha256"] = sha(trainer.read_bytes())
        config = rp.cube_segment_config(DEALER, docs["pool"], docs["constants"], docs["base"], docs["pins"])
        args = self.write([config], docs)
        config_path = self.dir / "segment-0.json"
        validation = {
            "schema": "phase1-native-expanded-config-validation/v1",
            "config_sha256": sha(config_path.read_bytes()),
            "validation_scope": "config-shape-and-registered-cards-only",
            "native_execution": False, "model_loaded": False,
            "runtime": {"card_db_hash": docs["pool"]["kernel_carddb_hash"]},
        }
        production_args = ("--production", "--cube-pins-sha256", sha((self.dir / NAMES["pins"]).read_bytes()),
                           "--cube-trainer", str(trainer))
        command = [str(trainer.resolve()), "--validate-config", str(config_path.resolve())]
        with patch.object(rr.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess(command, 0, json.dumps(validation), "")
            code, receipt = self.receipt(args, *production_args)
            self.assertEqual((code, receipt["native_config_validation"]), (0, [validation]))
            run.assert_called_once_with(command, capture_output=True, text=True, timeout=60)

            validation["config_sha256"] = "5" * 64
            run.return_value.stdout = json.dumps(validation)
            code, receipt = self.receipt(args, *production_args)
            self.assertEqual((code, receipt["failing_requirements"]), (1, ["gate1.native_config_admission"]))

    def test_empty_colliding_and_nonresuming_splits_refuse(self):
        whole = segment()
        first, second = copy.deepcopy(whole), copy.deepcopy(whole)
        first["iterations"], second["iterations"] = whole["iterations"][:1], whole["iterations"][1:]
        second["initial_source"]["checkpoint"] = {"path": "prior.json", "sha256": "1" * 64}
        for configs in ([dict(first, iterations=[]), whole], [first, second],
                        [first, dict(second, output_directory="other", initial_source=first["initial_source"])]):
            code, receipt = self.parity(self.write(configs))
            self.assertEqual((code, self.failed(receipt)), (1, ["run_fields_equal_the_base"]))

    def test_production_checks_actual_episode_profile_and_preparation_recipe(self):
        cases = (
            ("recipe.episode_profile", lambda docs: docs["pins"]["episode"].pop("priority_windows")),
            ("recipe.episode_profile", lambda docs: docs["base"]["iterations"][0]["episodes"][0]["episode"].update(
                runtime_rules="resolution-boundary-v1")),
            ("recipe.prepared_update", lambda docs: docs["base"].update(update_backward_execution="fixed_partition_4")),
            ("recipe.prepared_update", lambda docs: docs["base"].update(preparation_workers=1)),
        )
        for check, mutate in cases:
            docs = self.production_inputs()
            mutate(docs)
            config = rp.cube_segment_config(DEALER, docs["pool"], docs["constants"], docs["base"], docs["pins"])
            args = self.write([config], docs)
            code, receipt = self.receipt(args, "--production", "--cube-pins-sha256",
                                         sha((self.dir / NAMES["pins"]).read_bytes()))
            self.assertEqual((code, receipt["failing_requirements"]), (1, [check]))

    def test_relabelled_fixtures_cannot_make_a_production_receipt(self):
        docs = copy.deepcopy(self.docs)
        for entry in docs["pins"]["gate1"].values():
            entry["fixture"] = False
        args = self.write([segment()], docs)
        code, receipt = self.receipt(args, "--production")
        self.assertEqual(code, 1)
        self.assertFalse(receipt["production"])
        self.assertIn("gate1.external_pins_sha256", receipt["failing_requirements"])
        self.assertIn("gate1.non_placeholder_play_import", receipt["failing_requirements"])

    def test_different_initial_source_refuses_before_receipting_the_segment(self):
        mutations = (
            {"unrelated": True},
            {"play_import": {"path": "other.json", "sha256": "1" * 64}, "checkpoint": None},
        )
        for source in mutations:
            config = segment()
            config["initial_source"] = source
            code, receipt = self.parity(self.write([config]))
            self.assertEqual((code, self.failed(receipt)), (1, ["run_fields_equal_the_base"]))


class DeckIdentityTests(CubeCase):
    def edited(self, edit):
        config = segment()
        ep = config["iterations"][0]["episodes"][2]["episode"]
        edit(ep)
        code, receipt = self.parity(self.write([config]))
        self.assertEqual(code, 1)
        self.assertIn("decks_are_dealt_cube_decks", self.failed(receipt))
        self.assertIn("schedule_equals_the_dealer", self.failed(receipt))
        return receipt["decks"]["violations"][0]["problems"]

    def test_a_card_outside_the_pinned_pool_is_refused(self):
        pool = load("pool")
        outside = 1 + max(c["card_id"] for key in ("spells", "nonbasic_lands", "basics") for c in pool[key])

        def swap(ep):  # a card the fence_v3 pool does not hold (fenced, gapped or from another format)
            for deck in ep["registered"] + ep["selected"]:
                deck["mainboard"][0] = outside
        self.assertIn("outside the pinned pool", " ".join(self.edited(swap)))

    def test_a_nonbasic_card_in_both_decks_is_refused(self):
        def share(ep):
            card = next(i for i in ep["registered"][0]["mainboard"] if i not in {b["card_id"] for b in
                                                                                  load("pool")["basics"]})
            for side in ("registered", "selected"):
                ep[side][1]["mainboard"][-1] = card
        self.assertIn("nonbasic cards in both decks", " ".join(self.edited(share)))

    def test_a_sixty_card_deck_or_a_sideboard_is_refused(self):
        def grow(ep):
            for side in ("registered", "selected"):
                ep[side][0]["mainboard"] += ep[side][0]["mainboard"][:20]
                ep[side][1]["sideboard"] = [ep[side][1]["mainboard"][0]]
        problems = " ".join(self.edited(grow))
        self.assertIn("holds 60 cards, not 40", problems)
        self.assertIn("has a sideboard", problems)


class ScheduleIdentityTests(CubeCase):
    def test_another_segment_of_valid_decks_is_refused_with_its_first_difference(self):
        for overrides in ({"run": "seg1"}, {"update_start": 1, "update_end": 3}, {"namespace": "m2-t12-heldout"}):
            code, receipt = self.parity(self.write([segment(**overrides)]))
            self.assertEqual(code, 1, overrides)
            self.assertEqual(self.failed(receipt), ["schedule_equals_the_dealer"], overrides)
            self.assertEqual(receipt["schedule"]["first_difference"]["slot"], 0)

    def test_a_changed_seed_seat_or_opponent_is_located(self):
        cases = {"episode.seed": lambda e: e["episode"].update(seed=e["episode"]["seed"] + 1),
                 "episode.learner_seat": lambda e: e["episode"].update(learner_seat=1 - e["episode"]["learner_seat"]),
                 "opponent": lambda e: e.update(opponent={"kind": "initial"})}
        for field, edit in cases.items():
            config = segment()
            edit(config["iterations"][1]["episodes"][5])
            code, receipt = self.parity(self.write([config]))
            self.assertEqual(code, 1, field)
            self.assertEqual(receipt["schedule"]["first_difference"],
                             {"iteration": 1, "slot": 5, "fields": [field],
                              "expected_id": segment()["iterations"][1]["episodes"][5]["episode"]["id"]})

    def test_missing_opt_ins_and_a_changed_run_field_are_refused(self):
        config = segment()
        for item in config["iterations"]:
            for entry in item["episodes"]:
                del entry["episode"]["deck_format"]
        code, receipt = self.parity(self.write([config]))
        self.assertEqual(self.failed(receipt), ["episodes_carry_the_pinned_opt_ins", "schedule_equals_the_dealer"])
        config = segment()
        config["learning_rate"] = 0.001
        code, receipt = self.parity(self.write([config]))
        self.assertEqual((code, self.failed(receipt)), (1, ["run_fields_equal_the_base"]))

    def test_the_pins_must_declare_both_opt_ins_and_stamp_only_known_fields(self):
        for episode in ({"deck_format": "limited40-v1"}, {"deck_format": "limited40-v1",
                                                          "cube_flat_action_contract": 1},
                        dict(rp.CUBE_OPT_INS, london_keep_offset=0.0)):
            docs = copy.deepcopy(self.docs)
            docs["pins"]["episode"] = episode
            code, receipt = self.parity(self.write([segment()], docs))
            self.assertEqual((code, receipt), (1, None), episode)


class ReceiptIdentityTests(CubeCase):
    def test_unpinned_inputs_fail_their_identity_checks(self):
        docs = copy.deepcopy(self.docs)
        docs["pins"]["dealer"]["sha256"] = "4" * 64
        docs["pins"]["pool"]["kernel_carddb_hash"] = "0123456789abcdef"
        code, receipt = self.parity(self.write([segment()], docs))
        self.assertEqual((code, self.failed(receipt)), (1, ["dealer_is_pinned", "pool_is_pinned"]))
        checks = {"pool": "pool_is_pinned", "constants": "constants_are_pinned", "base": "base_is_pinned"}
        for name, check in checks.items():  # an input whose bytes are not the pinned ones
            docs = copy.deepcopy(self.docs)
            docs["pins"][name]["sha256"] = "6" * 64
            code, receipt = self.parity(self.write([segment()], docs,
                                                   repin=[n for n in NAMES if n not in ("pins", name)]))
            self.assertEqual((code, self.failed(receipt)), (1, [check]), name)

    def test_signed_identities_fail_the_receipt_even_when_consistently_repinned(self):
        cases = {
            "recipe.training_math": lambda d: d["base"].pop("training_math"),
            "recipe.loss_selection": lambda d: d["base"]["loss_selection"].update(pregame_advantage="terminal_v1"),
            "recipe.learning_rate": lambda d: d["base"].update(learning_rate=0.0002),
            "recipe.absent_paths": lambda d: d["base"].update(update_backend="gpu"),
            "max_non_natural_episode_fraction": lambda d: d["base"].update(max_non_natural_episode_fraction=0.01),
            "pool_input.fence": lambda d: d["pool"]["input_sha256"].update(fence="5" * 64)
            or d["pins"]["pool"]["input_sha256"].update(fence="5" * 64),
            "pool_counts": lambda d: d["pool"]["fenced"].pop(),
        }
        for rid, edit in cases.items():
            docs = copy.deepcopy(self.docs)
            edit(docs)
            code, receipt = self.receipt(self.write([rp.cube_segment_config(DEALER, docs["pool"], docs["constants"],
                                                                            docs["base"], docs["pins"])], docs))
            self.assertEqual(code, 1, rid)
            self.assertEqual(receipt["failing_requirements"], [rid], rid)
            self.assertEqual(receipt["raw_failures"], [], rid)

    def test_t9_batch_layout_binds_the_non_natural_fraction_and_the_gate1_value(self):
        self.assertEqual(rr.non_natural_fraction(8), 0.0625)
        self.assertEqual(rr.non_natural_fraction(12), 1 / 24)
        docs = copy.deepcopy(self.docs)
        docs["constants"]["pairs_per_update"] = 12
        docs["base"]["max_non_natural_episode_fraction"] = 1 / 24
        config = rp.cube_segment_config(DEALER, docs["pool"], docs["constants"], docs["base"], docs["pins"])
        code, receipt = self.receipt(self.write([config], docs))
        self.assertEqual((code, receipt["failing_requirements"]), (1, ["gate1.pairs_per_update"]))
        docs["pins"]["gate1"]["pairs_per_update"]["value"] = 12
        code, receipt = self.receipt(self.write([config], docs))
        self.assertEqual((code, receipt["failing_requirements"]), (0, []))
        docs["constants"]["pairs_per_update"] = 7  # odd: the dealer refuses it, so no receipt
        self.assertEqual(self.receipt(self.write([config], docs)), (1, None))

    def test_a_gate1_block_must_name_every_item(self):
        docs = copy.deepcopy(self.docs)
        del docs["pins"]["gate1"]["card_semantics"]
        self.assertEqual(self.receipt(self.write([segment()], docs)), (1, None))


class PauperUnchangedTests(unittest.TestCase):
    def run_main(self, main, argv):
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err), \
                self.assertRaises(SystemExit) as stop:
            main(argv)
        return stop.exception.code, err.getvalue()

    def test_without_cube_pins_the_board_arguments_stay_required(self):
        code, err = self.run_main(rp.main, ["--training-config", "t.json"])
        self.assertEqual(code, 2)
        self.assertIn("the following arguments are required: --board-run, --repo", err)
        code, err = self.run_main(rr.main, ["--training-config", "t.json"])
        self.assertEqual(code, 2)
        self.assertIn("required: --board-run, --board-snapshot, --repo, --amendment", err)

    def test_modes_do_not_mix(self):
        cube = ["--cube-pins", "p", "--cube-pool", "q", "--cube-constants", "c", "--cube-base", "b"]
        code, err = self.run_main(rp.main, ["--training-config", "t", *cube, "--repo", "r"])
        self.assertEqual(code, 2)
        self.assertIn("--repo not allowed with --cube-pins", err)
        code, err = self.run_main(rp.main, ["--training-config", "t", "--board-run", "b", "--repo", "r",
                                            "--cube-pool", "q"])
        self.assertIn("need --cube-pins", err)
        code, err = self.run_main(rp.main, ["--training-config", "t", "--cube-pins", "p"])
        self.assertIn("--cube-pins requires --cube-pool, --cube-constants, --cube-base", err)
        code, err = self.run_main(rr.main, ["--training-config", "t", *cube, "--census", "c"])
        self.assertIn("--census not allowed with --cube-pins", err)


if __name__ == "__main__":
    unittest.main()
