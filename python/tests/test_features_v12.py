"""FlatV9 (features_v12.py): the Cube feature contract on FlatV8.

Every FlatV9 tensor is the FlatV8 tensor of the relaxed observation, with the
FlatV9 object and state columns appended; each column reads its own public
field; the relaxation clears exactly the fields the columns encode; and the
descriptor and the Rust identity match the actual source.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

import torch

from mtg_kernel_rl import features_v11 as v11
from mtg_kernel_rl import features_v12 as v12
from fixtures import legal_actions
from test_features_v10 import TENSORS, in_game


ROOT = Path(__file__).resolve().parents[2]
O = v12.PREDECESSOR_OBJECT_FEATURE_DIM
S = v12.PREDECESSOR_STATE_FEATURE_DIM


def public_cards(obs: dict) -> list[dict]:
    p = obs["projection"]
    return [*p["battlefield"][0], *p["battlefield"][1], *p["graveyards"][0], *p["graveyards"][1], *p["exile"]]


PLAIN_HISTORY = {"turn_sequence": 0, "turns_begun": [0, 0], "consecutive_turn": False,
                 "attacked": [False, False], "creature_died": False, "lost_life": [False, False],
                 "tutor": [{"status": "none", "depth": 0, "draws_since": 0}] * 2}


def with_extension(obs: dict, facts: dict[int, dict] | None = None, attacked: int = 0,
                   history: dict | None = None) -> dict:
    """The observation with an extension of default facts, overridden by
    arena id from ``facts``."""
    obs = copy.deepcopy(obs)
    cards = []
    for card in public_cards(obs):
        zone = card["stable"]["zone"]
        entry = {"stable": copy.deepcopy(card["stable"]), "keywords": 0, "planeswalker": False, "loyalty": 0,
                 "entered_turns_ago": 3 if zone == "Battlefield" else None,
                 "plotted_turns_ago": 0 if zone == "Exile" and card.get("plotted_turn") is not None else None}
        entry.update((facts or {}).get(card["stable"]["arena_id"], {}))
        cards.append(entry)
    obs["extensions"]["flat_v9"] = {"cards": cards, "creatures_attacked_this_turn": attacked,
                                    "history": copy.deepcopy(history or PLAIN_HISTORY)}
    return obs


def encode(obs: dict) -> v12.EncodedDecision:
    return v12.encode_decision(copy.deepcopy(obs), legal_actions())


def node_of(obs: dict, arena_id: int) -> int:
    base = v11._base_observation(v12.sanitize_observation(v12.split_extension(obs)[0]))
    registry = v12.frozen_v4._objects(base)[0]
    card = next(card for card in public_cards(obs) if card["stable"]["arena_id"] == arena_id)
    return registry.resolve(card["stable"])


class FeaturesV12Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.obs = in_game()
        self.assertTrue(public_cards(self.obs), "the fixture needs public cards")
        self.card = public_cards(self.obs)[0]
        self.arena = self.card["stable"]["arena_id"]

    def assert_v8_plus_columns(self, obs: dict, new: v12.EncodedDecision) -> None:
        old = v11.encode_decision(v12.sanitize_observation(v12.split_extension(obs)[0]), legal_actions())
        for name in TENSORS:
            if name in ("state", "object_features"):
                continue
            self.assertTrue(torch.equal(getattr(old, name), getattr(new, name)), name)
        self.assertTrue(torch.equal(new.object_features[:, :O], old.object_features))
        self.assertTrue(torch.equal(new.state[:S], old.state))
        self.assertTrue(torch.equal(new.pairing_node_indices, old.pairing_node_indices))

    def test_plain_observation_is_v8_with_zero_columns(self):
        obs = with_extension(self.obs)
        new = encode(obs)
        self.assert_v8_plus_columns(obs, new)
        self.assertEqual(tuple(new.object_features.shape[1:]), (v12.OBJECT_FEATURE_DIM,))
        self.assertEqual(new.state.shape[0], v12.STATE_FEATURE_DIM)
        # The fixture's one exile permission, held by the actor.
        permission = obs["projection"]["exile_play_permissions"][0]
        self.assertEqual(permission["holder"], obs["acting_player"])
        expected = torch.zeros((new.object_features.shape[0], 25))
        expected[node_of(obs, permission["object"]["arena_id"]), 10] = 1.0
        self.assertTrue(torch.equal(new.object_features[:, O:O + 25], expected))
        # Only the tutor markers' "none" columns are set in a plain history.
        nonzero = [i for i, x in enumerate(new.state[S:].tolist()) if x != 0.0]
        self.assertEqual(nonzero, [37, 44])
        self.assertEqual(new.schema.object_feature_dim, 131)
        self.assertEqual(new.schema.state_dim, 270)
        self.assertEqual(new.schema.contract_digest, v12.feature_contract_fingerprint())
        self.assertEqual(new.schema.encoding_digest, v12.encoding_contract_fingerprint())

    def test_the_extension_is_required_and_exact(self):
        with self.assertRaises(v12.FeatureSchemaError):
            encode(self.obs)
        obs = with_extension(self.obs)
        obs["extensions"]["flat_v9"]["cards"].pop()
        with self.assertRaises(v12.FeatureSchemaError):
            encode(obs)
        obs = with_extension(self.obs)
        obs["extensions"]["flat_v9"]["cards"][0]["stable"]["zone_change_count"] += 1
        with self.assertRaises(v12.FeatureSchemaError):
            encode(obs)
        obs = with_extension(self.obs)
        obs["extensions"]["flat_v9"]["extra"] = 1
        with self.assertRaises(v12.FeatureSchemaError):
            encode(obs)

    def test_planeswalker_loyalty_and_keywords_read_the_extension(self):
        obs = with_extension(self.obs, {self.arena: {"planeswalker": True, "loyalty": 4, "keywords": 0b1010001}})
        new = encode(obs)
        self.assert_v8_plus_columns(obs, new)
        row = new.object_features[node_of(obs, self.arena), O:].tolist()
        self.assertEqual(row[0], 1.0)
        self.assertAlmostEqual(row[1], 0.4, places=6)
        self.assertEqual(row[3:10], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        # A planeswalker's loyalty changes the tensor.
        other = encode(with_extension(self.obs, {self.arena: {"planeswalker": True, "loyalty": 3}}))
        self.assertFalse(torch.equal(other.object_features, new.object_features))

    def test_charge_counters_are_relaxed_and_encoded(self):
        obs = with_extension(self.obs)
        self.card_in(obs)["counters"]["charge"] = 3
        with self.assertRaises(v12.FeatureSchemaError):
            v11.encode_decision(v12.split_extension(obs)[0], legal_actions())
        new = encode(obs)
        self.assert_v8_plus_columns(obs, new)
        self.assertAlmostEqual(new.object_features[node_of(obs, self.arena), O + 2].item(), 0.3, places=6)
        # The relaxed observation is the plain one.
        plain = encode(with_extension(self.obs))
        self.assertTrue(torch.equal(new.object_features[:, :O], plain.object_features[:, :O]))

    def test_effective_identity_and_creature_upgrade(self):
        obs = with_extension(self.obs)
        card = self.card_in(obs)
        card["characteristics"]["effective_identity"] = {"names": ["A", "B"], "mana_value": 5,
                                                         "supertypes": ["Legendary"]}
        card["creature_upgrade"] = {"finality": 1, "temporary_creature": [2, 2, 7], "suppressed_by": [],
                                    "graveyard_adventure": None, "haste_blockers_only": True,
                                    "creature_types": None, "base_stats": None, "color": None,
                                    "keyword_grants": [[1, 3]], "keyword_losses": [], "combat_impulse": 9,
                                    "once_activated": [], "wurmlet_resolved_turn": None}
        new = encode(obs)
        self.assert_v8_plus_columns(obs, new)
        row = new.object_features[node_of(obs, self.arena), O:].tolist()
        self.assertEqual(row[14], 1.0)
        self.assertAlmostEqual(row[15], 0.5, places=6)
        self.assertEqual((row[16], row[17]), (1.0, 0.5))
        self.assertEqual(row[18], 1.0)
        self.assertAlmostEqual(row[19], 0.1, places=6)
        self.assertEqual((row[20], row[21], row[22], row[23], row[24]), (1.0, 0.125, 0.0, 1.0, 1.0))

    def test_exile_permissions(self):
        obs = copy.deepcopy(self.obs)
        p = obs["projection"]
        exiled = copy.deepcopy(p["battlefield"][0][0])
        exiled["stable"]["arena_id"] = 999
        exiled["stable"]["zone"] = "Exile"
        p["exile"].append(exiled)
        p["exile_play_permissions"].append({
            "without_mana_cost": True, "object": copy.deepcopy(exiled["stable"]), "holder": "p1",
            "play_or_cast": "cast", "zone_change_generation": 0,
            "expiry": {"expiry_kind": "until_holders_next_end_step"}})
        obs = with_extension(obs)
        relaxed = v12.sanitize_observation(v12.split_extension(obs)[0])
        permission = relaxed["projection"]["exile_play_permissions"][-1]
        self.assertNotIn("without_mana_cost", permission)
        self.assertEqual(permission["expiry"], {"expiry_kind": "end_of_turn"})
        new = encode(obs)
        self.assert_v8_plus_columns(obs, new)
        row = new.object_features[node_of(obs, 999), O:].tolist()
        self.assertEqual(row[10:14], [0.0, 1.0, 1.0, 1.0])

    def test_state_columns(self):
        obs = copy.deepcopy(self.obs)
        p = obs["projection"]
        p["restricted_mana"] = [[{"color": "C", "restriction": "ArtifactSpellV1", "source_card_def": 1,
                                  "source": None}],
                                [{"color": "G", "restriction": {"ChosenCreatureTypeV60": None},
                                  "source_card_def": 2, "source": None},
                                 {"color": "G", "restriction": "CreatureSpell", "source_card_def": 2,
                                  "source": None}]]
        p["poison_counters"] = [2, 5]
        p["public_shroud"] = {"permanents": [], "players": [False, True]}
        p["turn_schedule"] = {"occurrence": 4, "legacy_round": 2, "own_turns_started": [3, 1],
                              "previous_player": "p0", "next_regular_player": "p1",
                              "current_extra": {"creation_ordinal": 1, "captured_controller": "p0",
                                                "physical_owner": "p0", "spell_copy": False},
                              "queued_next_first": [{"creation_ordinal": 2, "captured_controller": "p1",
                                                     "physical_owner": "p1", "spell_copy": False}]}
        obs = with_extension(obs, attacked=3)
        state = encode(obs).state[S:].tolist()
        actor_is_p0 = obs["acting_player"] == "p0"
        mine, theirs = (0, 1) if actor_is_p0 else (1, 0)
        colorless = [0.0] * 5 + [0.1]
        green = [0.0, 0.0, 0.0, 0.0, 0.2, 0.0]
        self.assertEqual([round(x, 6) for x in state[0:12]], (colorless + green) if actor_is_p0 else (green + colorless))
        artifact, creature = [0.0, 0.1, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]
        self.assertEqual([round(x, 6) for x in state[12:20]],
                         (artifact + creature) if actor_is_p0 else (creature + artifact))
        self.assertAlmostEqual(state[20], 0.3, places=6)
        self.assertEqual([round(state[21], 6), round(state[22], 6)], [[0.2, 0.5], [0.5, 0.2]][mine])
        self.assertEqual(state[23:25], [[0.0, 1.0], [1.0, 0.0]][mine])
        self.assertEqual(state[25], 1.0)
        self.assertEqual(state[26:28], [0.0, 0.25] if actor_is_p0 else [0.25, 0.0])
        del theirs

    def test_history_columns(self):
        me = 0 if self.obs["acting_player"] == "p0" else 1
        history = copy.deepcopy(PLAIN_HISTORY)
        history.update({"turn_sequence": 40, "turns_begun": [20, 19], "consecutive_turn": True,
                        "attacked": [False, False], "creature_died": True, "lost_life": [False, False]})
        history["attacked"][1 - me] = True
        history["lost_life"][me] = True
        history["tutor"] = [None, None]
        history["tutor"][me] = {"status": "tracked", "depth": 2, "draws_since": 9}
        history["tutor"][1 - me] = {"status": "drawn_or_removed", "depth": 0, "draws_since": 1}
        obs = with_extension(self.obs, {self.arena: {"entered_turns_ago": 1}}, history=history)
        new = encode(obs)
        self.assert_v8_plus_columns(obs, new)
        state = new.state[S:].tolist()
        self.assertEqual(state[28], 1.0)
        self.assertEqual(state[29:31], [1.0, 1.0])
        self.assertEqual(state[31:37], [1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        self.assertEqual(state[37:44], [0.0, 0.0, 1.0, 0.0, 0.0, 0.25, 1.0])
        self.assertEqual(state[44:51], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.125])
        row = new.object_features[node_of(obs, self.arena), O:].tolist()
        self.assertEqual(row[25:29], [1.0, 0.0, 1.0, 0.125])
        self.assertEqual(row[29:33], [0.0, 0.0, 0.0, 0.0])
        # Ages saturate at the bound; a battlefield card must carry one.
        far = encode(with_extension(self.obs, {self.arena: {"entered_turns_ago": 30}}))
        self.assertEqual(far.object_features[node_of(obs, self.arena), O + 28].item(), 1.0)
        with self.assertRaises(v12.FeatureSchemaError):
            encode(with_extension(self.obs, {self.arena: {"entered_turns_ago": None}}))

    def card_in(self, obs: dict) -> dict:
        return next(card for card in public_cards(obs) if card["stable"]["arena_id"] == self.arena)

    def test_descriptor_and_rust_identity_match_the_source(self):
        directory = ROOT / "data" / "flat_policy_v9"
        descriptor_bytes = (directory / "feature_contract_v9.json").read_bytes()
        descriptor = json.loads(descriptor_bytes)
        source = (ROOT / "python" / "mtg_kernel_rl" / "features_v12.py").read_bytes()
        self.assertEqual(descriptor["features_source_sha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(descriptor["feature_contract_digest"], v12.feature_contract_fingerprint())
        self.assertEqual(descriptor["feature_encoding_digest"], v12.encoding_contract_fingerprint())
        identity = (directory / "feature_identity.rs").read_text()
        self.assertIn(hashlib.sha256(descriptor_bytes).hexdigest(), identity)
        self.assertIn(v12.feature_contract_fingerprint(), identity)
        self.assertNotEqual(v12.feature_contract_fingerprint(), v11.feature_contract_fingerprint())


class RustParityTests(unittest.TestCase):
    """Rust FlatV9 cases (``flat_v9_parity_cases_for_python`` in
    ``flat_policy_v6_tests.rs``) read from ``MTG_KERNEL_FLAT_V9_PARITY_OUT``;
    the checked-in golden fixture follows the contract freeze."""

    def test_rust_cases_match_bit_for_bit(self):
        import os
        path = os.environ.get("MTG_KERNEL_FLAT_V9_PARITY_OUT")
        if not path:
            self.skipTest("MTG_KERNEL_FLAT_V9_PARITY_OUT is not set")
        fixture = json.loads(Path(path).read_text())
        self.assertEqual(fixture["feature_contract_digest"], v12.feature_contract_fingerprint())
        self.assertTrue(fixture["cases"])
        for index, case in enumerate(fixture["cases"]):
            encoded = v12.encode_decision(case["observation"], case["legal_actions"])
            for name, expected in case["tensor"].items():
                actual = getattr(encoded, name)
                want = torch.tensor(expected, dtype=actual.dtype).reshape(actual.shape)
                if actual.dtype == torch.float32:
                    self.assertTrue(torch.equal(actual.view(torch.int32), want.view(torch.int32)),
                                    f"case {index} {name}")
                else:
                    self.assertTrue(torch.equal(actual, want), f"case {index} {name}")


if __name__ == "__main__":
    unittest.main()
