"""FlatV9: the Cube feature contract (M2 task T6) on FlatV8.

The frozen features_v11.py (FlatV8) encoder supplies every tensor of a
*relaxed* observation, and FlatV9 appends columns:

- **Relaxed observation.** :func:`sanitize_observation` clears exactly the
  public fields the FlatV8 encoder refuses or ignores and FlatV9 encodes
  itself: ``projection.creatures_attacked_this_turn``,
  ``projection.poison_counters``, ``projection.restricted_mana``,
  ``projection.public_shroud`` and ``projection.turn_schedule``; each public card's charge counters,
  ``creature_upgrade`` and ``characteristics.effective_identity``; each exile
  play permission's ``without_mana_cost``, and an
  ``until_holders_next_end_step`` expiry reads ``end_of_turn``; and
  ``projection.engine_context.planeswalkers``. Under an extra-turn schedule
  ``projection.turn`` becomes its occurrence and goads expiring at 0 expire
  at it, since turn-keyed fields hold the occurrence. The Rust
  encoder clears the same fields of its typed observation
  (``flat_policy_v6.rs``, ``sanitize_observation_v1``).
- **Object columns** (``OBJECT_EXTRA_DIM``, appended to every object row): a
  public card's (battlefield, graveyard, exile) planeswalker type, loyalty,
  charge counters, seven keyword bits FlatV8 lacks, exile play permissions,
  effective identity and creature upgrade; zeros on every other row.
- **State columns** (``STATE_EXTRA_DIM``, appended to the state row):
  restricted mana by seat, color and restriction class, creatures attacked
  this turn, poison counters, player shroud and the extra-turn schedule.

- **History columns** (object and state, ``flat_v9_history_v1.rs``): the
  exact turn of each battlefield entry and plot (turns ago, from a turn
  sequence that advances on every actual turn, issue 740), turns begun by
  each seat, consecutive turns, this turn's attacks, creature deaths and life
  loss, and each library's identity-free tutor marker with draws since.

Facts the observation does not carry come from ``extensions.flat_v9``
(``FlatV9ExtensionV1``): per public card, its extra keyword bits, its
planeswalker type, loyalty and turn ages; the turn's attack count; and the
history facts. Every value is public. Every other tensor equals the FlatV8 tensor of the relaxed
observation bit for bit.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import torch

from . import features_v7 as frozen_v4
from . import features_v11 as predecessor

FeatureSchemaError = predecessor.FeatureSchemaError
FeatureSchema = predecessor.FeatureSchema
FEATURE_SCHEMA_VERSION = "actor-relative-v12-python-1"
FEATURE_REGISTRY_VERSION = "rust-observation-v12-action-v8-registry-1"
ENCODING_CONTRACT_VERSION = "actor-relative-node-graph-21"
MODEL_CONTRACT_VERSION = predecessor.MODEL_CONTRACT_VERSION
PREDECESSOR_STATE_FEATURE_DIM = predecessor.STATE_FEATURE_DIM
PREDECESSOR_OBJECT_FEATURE_DIM = predecessor.OBJECT_FEATURE_DIM
OBJECT_EXTRA_DIM = 33
STATE_EXTRA_DIM = 51
AGE_BOUND = 8
TURN_SEQUENCE_BOUND = 32
TURNS_BEGUN_BOUND = 16
TUTOR_STATUSES = ("none", "on_top", "buried", "drawn_or_removed", "invalidated")
STATE_FEATURE_DIM = PREDECESSOR_STATE_FEATURE_DIM + STATE_EXTRA_DIM
OBJECT_FEATURE_DIM = PREDECESSOR_OBJECT_FEATURE_DIM + OBJECT_EXTRA_DIM
EDGE_FEATURE_DIM = predecessor.EDGE_FEATURE_DIM
ACTION_FEATURE_DIM = predecessor.ACTION_FEATURE_DIM
ACTION_REF_FEATURE_DIM = predecessor.ACTION_REF_FEATURE_DIM

EXTRA_KEYWORDS = ("islandwalk", "flash", "cant_be_blocked", "toxic", "shroud",
                  "protection_from_everything", "infect")
OBJECT_COLUMNS = (
    "planeswalker", "loyalty/10", "charge_counters/10",
    *(f"keyword_{name}" for name in EXTRA_KEYWORDS),
    "exile_permission_self", "exile_permission_opponent", "exile_permission_without_mana_cost",
    "exile_permission_until_next_end_step",
    "effective_identity", "effective_identity_mana_value/10", "effective_identity_legendary",
    "effective_identity_names/4",
    "creature_upgrade", "creature_upgrade_finality/10", "creature_upgrade_temporary_creature",
    "creature_upgrade_keyword_grants/8", "creature_upgrade_keyword_losses/8",
    "creature_upgrade_combat_impulse", "creature_upgrade_haste_blockers_only",
    "entered", "entered_this_turn", "entered_previous_turn", f"entered_turns_ago/{AGE_BOUND}",
    "plotted", "plotted_this_turn", "plotted_previous_turn", f"plotted_turns_ago/{AGE_BOUND}",
)
MANA_COLORS = ("W", "U", "B", "R", "G", "C")
RESTRICTION_CLASSES = {
    "CreatureSpell": 0, "CreatureSpellOrCreatureAbility": 0, "CreatureSpellOfChosenTypeV1": 0,
    "ChosenCreatureTypeV60": 0,
    "ArtifactSpellV1": 1, "PowerstoneV1": 1,
    "NoncreatureSpellV1": 2,
    "LegendarySpell": 3, "Unrestricted": 3, "ColorlessEldraziSpellV1": 3, "SpellWithSubtype": 3,
}
STATE_COLUMNS = (
    *(f"self_restricted_{color}/10" for color in MANA_COLORS),
    *(f"opponent_restricted_{color}/10" for color in MANA_COLORS),
    *(f"self_restricted_class_{c}/10" for c in ("creature", "artifact", "noncreature", "other")),
    *(f"opponent_restricted_class_{c}/10" for c in ("creature", "artifact", "noncreature", "other")),
    "creatures_attacked_this_turn/10", "self_poison/10", "opponent_poison/10",
    "self_shroud", "opponent_shroud",
    "extra_turn_in_progress", "self_queued_extra_turns/4", "opponent_queued_extra_turns/4",
    f"turn_sequence/{TURN_SEQUENCE_BOUND}", f"self_turns_begun/{TURNS_BEGUN_BOUND}",
    f"opponent_turns_begun/{TURNS_BEGUN_BOUND}", "consecutive_turn",
    "self_attacked_this_turn", "opponent_attacked_this_turn", "creature_died_this_turn",
    "self_lost_life_this_turn", "opponent_lost_life_this_turn",
    *(f"{seat}_tutor_{column}" for seat in ("self", "opponent")
      for column in (*TUTOR_STATUSES, f"depth/{AGE_BOUND}", f"draws_since/{AGE_BOUND}")),
)
assert len(OBJECT_COLUMNS) == OBJECT_EXTRA_DIM and len(STATE_COLUMNS) == STATE_EXTRA_DIM

EncodedDecision = predecessor.EncodedDecision


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def feature_contract_fingerprint() -> str:
    return _digest({
        "registry_version": FEATURE_REGISTRY_VERSION,
        "predecessor_contract": predecessor.feature_contract_fingerprint(),
        "observation": "FlatV8 of the relaxed observation plus extensions.flat_v9",
        "relaxed": ["projection.creatures_attacked_this_turn", "projection.poison_counters",
                    "projection.restricted_mana", "projection.public_shroud", "projection.turn_schedule",
                    "card.counters.charge", "card.creature_upgrade",
                    "card.characteristics.effective_identity",
                    "exile_play_permission.without_mana_cost",
                    "exile_play_permission.expiry until_holders_next_end_step",
                    "engine_context.planeswalkers",
                    "turn and goad expiry read as the schedule occurrence"],
        "extension": {"cards": ["stable", "keywords", "planeswalker", "loyalty", "entered_turns_ago",
                                "plotted_turns_ago"],
                      "creatures_attacked_this_turn": "count",
                      "history": ["turn_sequence", "turns_begun", "consecutive_turn", "attacked",
                                  "creature_died", "lost_life", "tutor"]},
        "history": {
            "turn_sequence": "actual turns begun, the current one included; extra and consecutive turns count",
            "stamps": "exact turn of each battlefield entry and plot; a zone change is a new incarnation",
            "tutor": "identity-free marker of a tutor's public top placement, moved only by public library notes",
            "bounds": {"age": AGE_BOUND, "turn_sequence": TURN_SEQUENCE_BOUND, "turns_begun": TURNS_BEGUN_BOUND},
        },
        "object_columns": list(OBJECT_COLUMNS),
        "state_columns": list(STATE_COLUMNS),
    })


def encoding_contract_fingerprint() -> str:
    return _digest({
        "encoding_version": ENCODING_CONTRACT_VERSION,
        "predecessor_encoding": predecessor.encoding_contract_fingerprint(),
        "dimensions": {"state": STATE_FEATURE_DIM, "object": OBJECT_FEATURE_DIM, "edge": EDGE_FEATURE_DIM,
                       "action": ACTION_FEATURE_DIM, "action_ref": ACTION_REF_FEATURE_DIM},
        "object_columns": {"first": PREDECESSOR_OBJECT_FEATURE_DIM, "names": list(OBJECT_COLUMNS),
                           "rows": "public battlefield, graveyard and exile cards; zeros elsewhere"},
        "state_columns": {"first": PREDECESSOR_STATE_FEATURE_DIM, "names": list(STATE_COLUMNS)},
        "scaling": "float64 value / scale, rounded once to float32",
        "other_tensors": "FlatV8 tensors of the relaxed observation bit for bit",
    })


def model_contract_fingerprint(schema: FeatureSchema) -> str:
    return _digest({"model_contract_version": MODEL_CONTRACT_VERSION,
                    "feature_schema": schema.__dict__})


def classification_registry() -> dict[str, str]:
    return predecessor.classification_registry()


def _seat_index(seat: Any) -> int:
    if seat == "p0":
        return 0
    if seat == "p1":
        return 1
    raise FeatureSchemaError("seat must be p0 or p1")


def _public_lists(observation: dict[str, Any]) -> list[list[dict[str, Any]]]:
    projection = observation["projection"]
    return [projection["battlefield"][0], projection["battlefield"][1],
            projection["graveyards"][0], projection["graveyards"][1], projection["exile"]]


def split_extension(observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The observation without ``extensions.flat_v9``, and that extension."""
    if not isinstance(observation, dict) or not isinstance(observation.get("extensions"), dict):
        raise FeatureSchemaError("observation requires an extensions object")
    extensions = observation["extensions"]
    extension = extensions.get("flat_v9")
    if not isinstance(extension, dict) or set(extension) != {"cards", "creatures_attacked_this_turn", "history"}:
        raise FeatureSchemaError("FlatV9 requires extensions.flat_v9 with cards, creatures_attacked_this_turn "
                                 "and history")
    _validate_history(extension["history"])
    count = extension["creatures_attacked_this_turn"]
    if type(count) is not int or count < 0:
        raise FeatureSchemaError("flat_v9.creatures_attacked_this_turn must be a nonnegative integer")
    cards = extension["cards"]
    if not isinstance(cards, list):
        raise FeatureSchemaError("flat_v9.cards must be a list")
    for card in cards:
        if (not isinstance(card, dict)
                or set(card) != {"stable", "keywords", "planeswalker", "loyalty", "entered_turns_ago",
                                 "plotted_turns_ago"}
                or type(card["keywords"]) is not int or not 0 <= card["keywords"] < 1 << len(EXTRA_KEYWORDS)
                or type(card["planeswalker"]) is not bool
                or type(card["loyalty"]) is not int or card["loyalty"] < 0
                or not all(card[key] is None or (type(card[key]) is int and card[key] >= 0)
                           for key in ("entered_turns_ago", "plotted_turns_ago"))):
            raise FeatureSchemaError("flat_v9 card facts require stable, keywords, planeswalker, loyalty and "
                                     "turn ages")
    base = {**observation, "extensions": {k: v for k, v in extensions.items() if k != "flat_v9"}}
    return base, extension


def _validate_history(history: Any) -> None:
    keys = {"turn_sequence", "turns_begun", "consecutive_turn", "attacked", "creature_died", "lost_life", "tutor"}
    if not isinstance(history, dict) or set(history) != keys:
        raise FeatureSchemaError(f"flat_v9.history requires exactly {sorted(keys)}")

    def count(value: Any) -> bool:
        return type(value) is int and value >= 0

    def pair(value: Any, check) -> bool:
        return isinstance(value, list) and len(value) == 2 and all(check(item) for item in value)

    def flag(value: Any) -> bool:
        return type(value) is bool

    def tutor(value: Any) -> bool:
        return (isinstance(value, dict) and set(value) == {"status", "depth", "draws_since"}
                and value["status"] in ("none", "tracked", "drawn_or_removed", "invalidated")
                and count(value["depth"]) and count(value["draws_since"]))

    if not (count(history["turn_sequence"]) and pair(history["turns_begun"], count)
            and flag(history["consecutive_turn"]) and pair(history["attacked"], flag)
            and flag(history["creature_died"]) and pair(history["lost_life"], flag)
            and pair(history["tutor"], tutor)):
        raise FeatureSchemaError("flat_v9.history has an invalid value")


def sanitize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """A deep copy with the FlatV9 fields cleared, as the Rust encoder clears them."""
    observation = copy.deepcopy(observation)
    projection = observation["projection"]
    schedule = projection.get("turn_schedule")
    if schedule is not None:
        # Turn-keyed fields hold the occurrence under the schedule, and goads
        # expire at 0; without it they read against the turn.
        occurrence = schedule["occurrence"]
        projection["turn"] = occurrence
        for cards in _public_lists(observation):
            for card in cards:
                for goad in card["goaded_by"]:
                    if goad["expires_at_turn"] == 0:
                        goad["expires_at_turn"] = occurrence
    projection["engine_context"].pop("planeswalkers", None)
    for key in ("creatures_attacked_this_turn", "poison_counters", "restricted_mana", "public_shroud",
                "turn_schedule"):
        projection.pop(key, None)
    for cards in _public_lists(observation):
        for card in cards:
            card["counters"].pop("charge", None)
            card.pop("creature_upgrade", None)
            card["characteristics"].pop("effective_identity", None)
    for permission in projection["exile_play_permissions"]:
        permission.pop("without_mana_cost", None)
        if permission["expiry"] == {"expiry_kind": "until_holders_next_end_step"}:
            permission["expiry"] = {"expiry_kind": "end_of_turn"}
    return observation


def _scaled(value: int, scale: float) -> float:
    return float(value) / scale


def _same_card(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (left["arena_id"], left["zone_change_count"]) == (right["arena_id"], right["zone_change_count"])


def _card_row(card: dict[str, Any], facts: dict[str, Any], observation: dict[str, Any]) -> list[float]:
    if facts["stable"] != card["stable"]:
        raise FeatureSchemaError("flat_v9 card facts differ from the observation's card")
    actor = observation["acting_player"]
    row = [0.0] * OBJECT_EXTRA_DIM
    row[0] = float(facts["planeswalker"])
    row[1] = _scaled(facts["loyalty"], 10.0)
    row[2] = _scaled(card["counters"].get("charge", 0), 10.0)
    for bit in range(len(EXTRA_KEYWORDS)):
        row[3 + bit] = float(facts["keywords"] >> bit & 1)
    for permission in observation["projection"]["exile_play_permissions"]:
        if not _same_card(permission["object"], card["stable"]):
            continue
        row[10 if permission["holder"] == actor else 11] = 1.0
        if permission.get("without_mana_cost", False):
            row[12] = 1.0
        if permission["expiry"].get("expiry_kind") == "until_holders_next_end_step":
            row[13] = 1.0
    identity = card["characteristics"].get("effective_identity")
    if identity is not None:
        row[14] = 1.0
        row[15] = _scaled(identity["mana_value"], 10.0)
        row[16] = float("Legendary" in identity["supertypes"])
        row[17] = _scaled(len(identity["names"]), 4.0)
    upgrade = card.get("creature_upgrade")
    if upgrade is not None:
        required = {"finality", "temporary_creature", "keyword_grants", "keyword_losses", "combat_impulse",
                    "haste_blockers_only"}
        if not isinstance(upgrade, dict) or not required <= set(upgrade):
            raise FeatureSchemaError(f"creature_upgrade requires {sorted(required)}")
        row[18] = 1.0
        row[19] = _scaled(upgrade["finality"], 10.0)
        row[20] = float(upgrade["temporary_creature"] is not None)
        row[21] = _scaled(len(upgrade["keyword_grants"]), 8.0)
        row[22] = _scaled(len(upgrade["keyword_losses"]), 8.0)
        row[23] = float(upgrade["combat_impulse"] is not None)
        row[24] = float(upgrade["haste_blockers_only"])
    zone = card["stable"]["zone"]
    expect_entry = zone == "Battlefield"
    expect_plot = zone == "Exile" and card.get("plotted_turn") is not None
    if (facts["entered_turns_ago"] is not None) != expect_entry or (facts["plotted_turns_ago"] is not None) != expect_plot:
        raise FeatureSchemaError("flat_v9 turn ages belong to battlefield entries and plotted exiled cards")
    for first, age in ((25, facts["entered_turns_ago"]), (29, facts["plotted_turns_ago"])):
        if age is not None:
            row[first] = 1.0
            row[first + 1] = float(age == 0)
            row[first + 2] = float(age == 1)
            row[first + 3] = _scaled(min(age, AGE_BOUND), float(AGE_BOUND))
    return row


def _restriction_name(restriction: Any) -> str:
    if isinstance(restriction, str):
        name = restriction
    elif isinstance(restriction, dict) and len(restriction) == 1:
        name = next(iter(restriction))
    else:
        raise FeatureSchemaError("restricted mana requires a restriction variant")
    if name not in RESTRICTION_CLASSES:
        raise FeatureSchemaError(f"unknown mana restriction {name}")
    return name


def columns(observation: dict[str, Any], extension: dict[str, Any]) -> tuple[list[tuple[dict[str, Any], list[float]]], list[float]]:
    """Each public card with its object columns, and the state columns, of
    the unsanitized observation (without ``extensions.flat_v9``)."""
    projection = observation["projection"]
    me = _seat_index(observation["acting_player"])
    relative = (me, 1 - me)
    cards = [card for cards in _public_lists(observation) for card in cards]
    if len(cards) != len(extension["cards"]):
        raise FeatureSchemaError("flat_v9 requires one card entry per public card")
    rows = [(card, _card_row(card, facts, observation)) for card, facts in zip(cards, extension["cards"])]

    state = [0.0] * STATE_EXTRA_DIM
    pools = projection.get("restricted_mana")
    if pools is not None:
        for order, seat in enumerate(relative):
            colors = [0] * 6
            classes = [0] * 4
            for unit in pools[seat]:
                colors[MANA_COLORS.index(unit["color"])] += 1
                classes[RESTRICTION_CLASSES[_restriction_name(unit["restriction"])]] += 1
            for color in range(6):
                state[order * 6 + color] = _scaled(colors[color], 10.0)
            for cls in range(4):
                state[12 + order * 4 + cls] = _scaled(classes[cls], 10.0)
    state[20] = _scaled(extension["creatures_attacked_this_turn"], 10.0)
    poison = projection.get("poison_counters")
    if poison is not None:
        state[21] = _scaled(poison[relative[0]], 10.0)
        state[22] = _scaled(poison[relative[1]], 10.0)
    shroud = projection.get("public_shroud")
    if shroud is not None:
        state[23] = float(shroud["players"][relative[0]])
        state[24] = float(shroud["players"][relative[1]])
    schedule = projection.get("turn_schedule")
    if schedule is not None:
        state[25] = float(schedule.get("current_extra") is not None)
        queued = [0, 0]
        for turn in schedule["queued_next_first"]:
            queued[int(_seat_index(turn["captured_controller"]) != me)] += 1
        state[26] = _scaled(queued[0], 4.0)
        state[27] = _scaled(queued[1], 4.0)
    history = extension["history"]
    state[28] = _scaled(min(history["turn_sequence"], TURN_SEQUENCE_BOUND), float(TURN_SEQUENCE_BOUND))
    for order, seat in enumerate(relative):
        state[29 + order] = _scaled(min(history["turns_begun"][seat], TURNS_BEGUN_BOUND), float(TURNS_BEGUN_BOUND))
        state[32 + order] = float(history["attacked"][seat])
        state[35 + order] = float(history["lost_life"][seat])
        tutor = history["tutor"][seat]
        first = 37 + order * 7
        status = {"none": 0, "drawn_or_removed": 3, "invalidated": 4}.get(tutor["status"])
        if status is None:
            status = 1 if tutor["depth"] == 0 else 2
        state[first + status] = 1.0
        state[first + 5] = _scaled(min(tutor["depth"], AGE_BOUND), float(AGE_BOUND))
        state[first + 6] = _scaled(min(tutor["draws_since"], AGE_BOUND), float(AGE_BOUND))
    state[31] = float(history["consecutive_turn"])
    state[34] = float(history["creature_died"])
    return rows, state


def _schema_v12(schema: FeatureSchema) -> FeatureSchema:
    return replace(schema, version=FEATURE_SCHEMA_VERSION, registry_version=FEATURE_REGISTRY_VERSION,
                   contract_digest=feature_contract_fingerprint(),
                   encoding_digest=encoding_contract_fingerprint(),
                   state_dim=STATE_FEATURE_DIM, object_feature_dim=OBJECT_FEATURE_DIM)


def encode_decision(observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> EncodedDecision:
    raw, extension = split_extension(observation)
    relaxed = sanitize_observation(raw)
    encoded = predecessor.encode_decision(relaxed, legal_actions)
    rows, state = columns(raw, extension)
    objects = encoded.object_features
    extra = torch.zeros((objects.shape[0], OBJECT_EXTRA_DIM), dtype=torch.float32)
    base = predecessor._base_observation(relaxed)
    if rows and base is None:
        raise FeatureSchemaError("a pregame decision has no public cards")
    if base is not None and rows:
        registry = frozen_v4._objects(base)[0]
        seen: set[int] = set()
        for card, row in rows:
            node = registry.resolve(card["stable"])
            if node in seen or not 0 <= node < objects.shape[0]:
                raise FeatureSchemaError("each public card has its own object node")
            seen.add(node)
            extra[node] = torch.tensor(row, dtype=torch.float32)
    fields = {name: getattr(encoded, name) for name in encoded.__dataclass_fields__ if name != "schema"}
    fields["object_features"] = torch.cat([objects, extra], dim=1)
    fields["state"] = torch.cat([encoded.state, torch.tensor(state, dtype=torch.float32)])
    return EncodedDecision(schema=_schema_v12(encoded.schema), **fields)
