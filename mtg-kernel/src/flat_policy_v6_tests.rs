//! FlatV9 (`flat_policy_v6.rs`, `native_flat_tensorizer_v9.rs`): the relaxed
//! observation keeps every FlatV8 tensor, each FlatV9 column reads its own
//! public field, and the Python reference agrees on real sessions.

use super::*;
use crate::flat_policy_v2::{FlatScoringDecisionViewV2, FlatScoringOwnedBuffersV2};
use crate::flat_policy_v4::{FlatDecisionEncoderV4, FlatScoringDecisionViewV4};
use crate::ids::{ObjectId, PlayerId};
use crate::native_flat_tensorizer_v2::NativeFlatDecisionTensorV2;
use crate::native_flat_tensorizer_v8::NativeFlatTensorizerV8;
use crate::native_flat_tensorizer_v9::NativeFlatTensorizerV9;
use crate::policy_observation_v6::tests::{put, ready_state_with_turn_authority};
use crate::rl_session::{FastActorResponseV1, FastActorSessionV1};
use crate::state::Zone;

const O: usize = crate::native_flat_tensorizer_v2::NATIVE_FLAT_OBJECT_FEATURE_DIM_V2;
const S: usize = crate::native_flat_tensorizer_v2::NATIVE_FLAT_STATE_FEATURE_DIM_V2;
const V9_O: usize = crate::native_flat_tensorizer_v9::V9_OBJECT_FEATURE_DIM_V1;

#[derive(Default)]
struct OwnedV1 {
    objects: Vec<crate::flat_policy_v2::FlatObjectCoreV2>,
    relations: Vec<crate::flat_policy_v2::FlatRelationV2>,
    object_subtypes: Vec<crate::flat_policy_v2::FlatObjectSubtypeV2>,
    ability_uses: Vec<crate::flat_policy_v2::FlatObjectAbilityUseV2>,
    goads: Vec<crate::flat_policy_v2::FlatObjectGoadV2>,
    completed_dungeons: Vec<crate::flat_policy_v2::FlatCompletedDungeonV2>,
    effect_subtype_changes: Vec<crate::flat_policy_v2::FlatEffectSubtypeChangeV2>,
    context_path_elements: Vec<crate::flat_policy_v2::FlatContextPathElementV2>,
    actions: Vec<crate::flat_policy_v2::FlatScorerActionCoreV2>,
    action_refs: Vec<crate::flat_policy_v2::FlatScorerActionRefV2>,
}

impl OwnedV1 {
    fn buffers(&mut self) -> FlatScoringOwnedBuffersV2<'_> {
        FlatScoringOwnedBuffersV2 {
            objects: &mut self.objects,
            relations: &mut self.relations,
            object_subtypes: &mut self.object_subtypes,
            ability_uses: &mut self.ability_uses,
            goads: &mut self.goads,
            completed_dungeons: &mut self.completed_dungeons,
            effect_subtype_changes: &mut self.effect_subtype_changes,
            context_path_elements: &mut self.context_path_elements,
            actions: &mut self.actions,
            action_refs: &mut self.action_refs,
        }
    }

    fn view<'a>(
        &'a self,
        decision: &'a crate::flat_policy_v4::FlatDecisionV4,
    ) -> FlatScoringDecisionViewV4<'a> {
        FlatScoringDecisionViewV4::new(
            FlatScoringDecisionViewV2::new(
                &decision.globals,
                &self.objects,
                &self.relations,
                &self.object_subtypes,
                &self.ability_uses,
                &self.goads,
                &self.completed_dungeons,
                &self.effect_subtype_changes,
                &self.context_path_elements,
                &self.actions,
                &self.action_refs,
            ),
            &decision.extensions,
        )
    }
}

fn decision_v1(session: &FastActorSessionV1) -> crate::rl_session::FastActorDecisionV1 {
    let FastActorResponseV1::Decision(decision) = session.current_response() else {
        panic!(
            "fixture is not a live decision: {:?}",
            session.current_response()
        );
    };
    decision
}

/// The FlatV8 tensors and pairing of the session's decision.
fn v8_tensors_v1(
    session: &FastActorSessionV1,
) -> Result<(NativeFlatDecisionTensorV2, Vec<i64>), FlatDecisionErrorV2> {
    let mut owned = OwnedV1::default();
    let decision = session.encode_current_flat_scoring_decision_owned_v4(
        decision_v1(session),
        &mut FlatDecisionEncoderV4::default(),
        &mut owned.buffers(),
    )?;
    let (mut tensor, mut pairs) = (NativeFlatDecisionTensorV2::default(), Vec::new());
    NativeFlatTensorizerV8::default()
        .fill(owned.view(&decision), &mut tensor, &mut pairs)
        .unwrap();
    Ok((tensor, pairs))
}

/// The FlatV9 decision, tensors and pairing of the session's decision.
fn v9_tensors_v1(
    session: &FastActorSessionV1,
) -> Result<(FlatDecisionV9, NativeFlatDecisionTensorV2, Vec<i64>), FlatDecisionErrorV2> {
    let mut owned = OwnedV1::default();
    let decision = session.encode_current_flat_scoring_decision_owned_v9(
        decision_v1(session),
        &mut FlatDecisionEncoderV4::default(),
        &mut owned.buffers(),
    )?;
    let (mut tensor, mut pairs) = (NativeFlatDecisionTensorV2::default(), Vec::new());
    NativeFlatTensorizerV9::default()
        .fill(
            FlatScoringDecisionViewV9 {
                v4: owned.view(&decision.v4),
                columns: &decision.columns,
            },
            &mut tensor,
            &mut pairs,
        )
        .unwrap();
    Ok((decision, tensor, pairs))
}

/// The FlatV9 tensors with the FlatV9 columns removed.
fn strip_v9_v1(tensor: &NativeFlatDecisionTensorV2) -> NativeFlatDecisionTensorV2 {
    let mut stripped = tensor.clone();
    stripped.object_features = tensor
        .object_features
        .chunks_exact(V9_O)
        .flat_map(|row| row[..O].iter().copied())
        .collect();
    stripped.state.truncate(S);
    stripped
}

/// A main-phase decision of `actor` with a Mountain each and an opposing
/// Lotus Petal carrying `charge` charge counters.
fn charge_fixture_v1(actor: PlayerId, charge: i32) -> (FastActorSessionV1, ObjectId) {
    let opponent = actor.opponent();
    let mut state = ready_state_with_turn_authority(actor);
    put(&mut state, actor, "Mountain", Zone::Battlefield);
    put(&mut state, actor, "Lightning Bolt", Zone::Hand);
    put(&mut state, opponent, "Mountain", Zone::Battlefield);
    let petal = put(&mut state, opponent, "Lotus Petal", Zone::Battlefield);
    for owner in [actor, opponent] {
        put(&mut state, owner, "Forest", Zone::Library);
    }
    state.objects.get_mut(petal).counters.charge = charge;
    crate::flat_v9_history_v1::enable_v1(&mut state);
    (FastActorSessionV1::from_v3_fixture_state(state), petal)
}

/// The FlatV9 object columns of the node whose public card is `arena_id`.
fn object_columns_v1(
    decision: &FlatDecisionV9,
    tensor: &NativeFlatDecisionTensorV2,
    group: FlatObjectGroupV2,
    ordinal: usize,
) -> Vec<f32> {
    let rows = &decision.columns.public_rows[match group {
        FlatObjectGroupV2::SelfBattlefield => 0,
        FlatObjectGroupV2::OpponentBattlefield => 1,
        FlatObjectGroupV2::SelfGraveyard => 2,
        FlatObjectGroupV2::OpponentGraveyard => 3,
        _ => 4,
    }];
    let wanted = rows[ordinal];
    let matches: Vec<_> = tensor
        .object_features
        .chunks_exact(V9_O)
        .filter(|row| row[O..] == wanted)
        .collect();
    assert!(!matches.is_empty(), "no node carries the card's columns");
    matches[0][O..].to_vec()
}

#[test]
fn flat_v9_relaxes_charge_counters_and_keeps_the_relaxed_v8_tensors() {
    for actor in [PlayerId::P0, PlayerId::P1] {
        let (plain, _) = charge_fixture_v1(actor, 0);
        let (charged, petal) = charge_fixture_v1(actor, 3);
        // FlatV8 refuses the charged state and encodes the plain one.
        assert_eq!(
            v8_tensors_v1(&charged).unwrap_err(),
            FlatDecisionErrorV2::ObservationContract
        );
        let (v8, v8_pairs) = v8_tensors_v1(&plain).unwrap();
        // FlatV9 of either state is FlatV8 of the plain state plus columns.
        let (plain_decision, plain_v9, plain_pairs) = v9_tensors_v1(&plain).unwrap();
        let (decision, v9, pairs) = v9_tensors_v1(&charged).unwrap();
        assert_eq!(strip_v9_v1(&plain_v9), v8, "{actor:?}");
        assert_eq!(strip_v9_v1(&v9), v8, "{actor:?}");
        assert_eq!((&plain_pairs, &pairs), (&v8_pairs, &v8_pairs));
        assert_eq!(
            v9.state.len(),
            crate::native_flat_tensorizer_v9::V9_STATE_FEATURE_DIM_V1
        );
        // The charge column is the only difference, on the Petal's row.
        let petal_ordinal = charged.game_state().players[actor.opponent().index()]
            .battlefield
            .iter()
            .position(|&id| id == petal)
            .unwrap();
        let row = object_columns_v1(
            &decision,
            &v9,
            FlatObjectGroupV2::OpponentBattlefield,
            petal_ordinal,
        );
        assert_eq!(row[2], 0.3_f32);
        let differing: Vec<_> = v9
            .object_features
            .iter()
            .zip(&plain_v9.object_features)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(index, _)| index % V9_O)
            .collect();
        assert_eq!(differing, vec![O + 2], "{actor:?}");
        assert_eq!(v9.state, plain_v9.state);
        assert_eq!(plain_decision.columns.state, decision.columns.state);
        // Outside the FlatV9 scope nothing is relaxed.
        assert!(!relaxed_v1());
        assert!(v8_tensors_v1(&charged).is_err());
    }
}

#[test]
fn relaxed_scope_nests_and_restores() {
    assert!(!relaxed_v1());
    {
        let _outer = relaxed_scope_v1();
        assert!(relaxed_v1());
        {
            let _inner = relaxed_scope_v1();
            assert!(relaxed_v1());
        }
        assert!(relaxed_v1());
    }
    assert!(!relaxed_v1());
}

/// The unsanitized observation of a fixture decision, its extension and
/// the actor.
fn observation_fixture_v1(actor: PlayerId) -> (FastActorSessionV1, ObservationV6) {
    let (session, _) = charge_fixture_v1(actor, 0);
    let observation = session
        .flat_policy_observation_raw_v4(decision_v1(&session))
        .unwrap();
    (session, observation)
}

#[test]
fn sanitize_clears_exactly_the_flat_v9_fields() {
    let (_, mut observation) = observation_fixture_v1(PlayerId::P0);
    let before = serde_json::to_value(&observation).unwrap();
    // A plain observation is unchanged.
    sanitize_observation_v1(&mut observation);
    assert_eq!(serde_json::to_value(&observation).unwrap(), before);
    let mut value = before.clone();
    let projection = &mut value["projection"];
    projection["creatures_attacked_this_turn"] = serde_json::json!(2);
    projection["poison_counters"] = serde_json::json!([1, 0]);
    projection["restricted_mana"] = serde_json::json!([[], []]);
    projection["public_shroud"] = serde_json::json!({"permanents": [], "players": [true, false]});
    let card = &mut projection["battlefield"][1][0];
    card["counters"]["charge"] = serde_json::json!(2);
    card["characteristics"]["effective_identity"] =
        serde_json::json!({"names": ["X"], "mana_value": 1, "supertypes": []});
    card["creature_upgrade"] = serde_json::json!({"finality": 1});
    let mut dirty: ObservationV6 = serde_json::from_value(value).unwrap();
    sanitize_observation_v1(&mut dirty);
    assert_eq!(serde_json::to_value(&dirty).unwrap(), before);
}

#[test]
fn flat_v9_columns_read_each_public_field() {
    for actor in [PlayerId::P0, PlayerId::P1] {
        let (session, observation) = observation_fixture_v1(actor);
        let extension = extension_v1(session.game_state(), &observation).unwrap();
        let plain = columns_v1(&observation, &extension).unwrap();
        assert!(plain.state[..28].iter().all(|&x| x == 0.0));
        let me = seat_index_v1(PlayerSeatV1::from(actor));
        let mut value = serde_json::to_value(&observation).unwrap();
        let projection = &mut value["projection"];
        let mut pools = [serde_json::json!([]), serde_json::json!([])];
        pools[me] = serde_json::json!([
            {"color": "C", "restriction": "ArtifactSpellV1", "source_card_def": 1, "source": null}
        ]);
        pools[1 - me] = serde_json::json!([
            {"color": "G", "restriction": {"ChosenCreatureTypeV60": null}, "source_card_def": 2, "source": null},
            {"color": "G", "restriction": "CreatureSpell", "source_card_def": 2, "source": null}
        ]);
        projection["restricted_mana"] = serde_json::json!(pools);
        let mut poison = [0, 0];
        poison[me] = 2;
        poison[1 - me] = 5;
        projection["poison_counters"] = serde_json::json!(poison);
        let mut shroud = [false, false];
        shroud[1 - me] = true;
        projection["public_shroud"] = serde_json::json!({"permanents": [], "players": shroud});
        let opponent_seat = if me == 0 { "p1" } else { "p0" };
        projection["turn_schedule"] = serde_json::json!({
            "occurrence": 4, "legacy_round": 2, "own_turns_started": [3, 1],
            "previous_player": "p0", "next_regular_player": "p1",
            "current_extra": {"creation_ordinal": 1, "captured_controller": "p0",
                              "physical_owner": "p0", "spell_copy": false},
            "queued_next_first": [{"creation_ordinal": 2, "captured_controller": opponent_seat,
                                   "physical_owner": opponent_seat, "spell_copy": false}]
        });
        let opponent_index = 1 - me;
        let card = &mut projection["battlefield"][opponent_index][0];
        card["counters"]["charge"] = serde_json::json!(3);
        card["characteristics"]["effective_identity"] =
            serde_json::json!({"names": ["A", "B"], "mana_value": 5, "supertypes": ["Legendary"]});
        card["creature_upgrade"] = serde_json::json!({
            "finality": 1, "temporary_creature": [2, 2, 7], "haste_blockers_only": true,
            "keyword_grants": [[1, 3]], "combat_impulse": 9
        });
        let observation: ObservationV6 = serde_json::from_value(value).unwrap();
        let mut extension = extension.clone();
        let facts_index = if opponent_index == 0 {
            0
        } else {
            observation.projection.surface.battlefield[0].len()
        };
        extension.cards[facts_index].planeswalker = true;
        extension.cards[facts_index].loyalty = 4;
        extension.cards[facts_index].keywords = 0b101_0001;
        extension.creatures_attacked_this_turn = 3;
        let columns = columns_v1(&observation, &extension).unwrap();
        let row = columns.public_rows[1][0];
        assert_eq!(row[0..3], [1.0, 0.4, 0.3]);
        assert_eq!(row[3..10], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
        assert_eq!(row[14..18], [1.0, 0.5, 1.0, 0.5]);
        assert_eq!(row[18..25], [1.0, 0.1, 1.0, 0.125, 0.0, 1.0, 1.0]);
        let state = &columns.state;
        assert_eq!(
            state[0..12],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0]
        );
        assert_eq!(state[12..20], [0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0]);
        assert_eq!(state[20..25], [0.3, 0.2, 0.5, 0.0, 1.0]);
        assert_eq!(state[25], 1.0);
        assert_eq!(state[26..28], [0.0, 0.25]);
        // A planeswalker's loyalty changes its row.
        extension.cards[facts_index].loyalty = 3;
        let lower = columns_v1(&observation, &extension).unwrap();
        assert_ne!(lower.public_rows[1][0], row);
        // Facts must describe the observation's cards, one each, in order.
        let mut wrong = extension.clone();
        wrong.cards[facts_index].stable.zone_change_count += 1;
        assert!(columns_v1(&observation, &wrong).is_err());
        let mut short = extension.clone();
        short.cards.pop();
        assert!(columns_v1(&observation, &short).is_err());
    }
}

#[test]
fn flat_v9_exile_permission_columns() {
    let (_, observation) = observation_fixture_v1(PlayerId::P0);
    let mut value = serde_json::to_value(&observation).unwrap();
    let mut exiled = value["projection"]["battlefield"][1][0].clone();
    exiled["stable"]["arena_id"] = serde_json::json!(999);
    exiled["stable"]["zone"] = serde_json::json!("Exile");
    let stable = exiled["stable"].clone();
    value["projection"]["exile"]
        .as_array_mut()
        .unwrap()
        .push(exiled);
    value["projection"]["exile_play_permissions"] = serde_json::json!([
        {"without_mana_cost": true, "object": stable, "holder": "p1", "play_or_cast": "cast",
         "zone_change_generation": 0, "expiry": {"expiry_kind": "until_holders_next_end_step"}},
        {"object": stable, "holder": "p0", "play_or_cast": "play",
         "zone_change_generation": 0, "expiry": {"expiry_kind": "end_of_turn"}}
    ]);
    let observation: ObservationV6 = serde_json::from_value(value).unwrap();
    let mut extension = FlatV9ExtensionV1::default();
    for card in public_cards_v1(&observation) {
        extension.cards.push(FlatV9CardFactsV1 {
            stable: card.stable.clone(),
            keywords: 0,
            planeswalker: false,
            loyalty: 0,
            entered_turns_ago: None,
            plotted_turns_ago: None,
        });
    }
    let columns = columns_v1(&observation, &extension).unwrap();
    let row = columns.public_rows[4].last().unwrap();
    assert_eq!(row[10..14], [1.0, 1.0, 1.0, 1.0]);
    let mut relaxed = observation.clone();
    sanitize_observation_v1(&mut relaxed);
    let permissions = &relaxed.projection.surface.exile_play_permissions;
    assert!(permissions.iter().all(|p| !p.without_mana_cost.0));
    assert!(permissions
        .iter()
        .all(|p| p.expiry == PlayPermissionExpiryV2::EndOfTurn));
}

/// The extension reads keywords, the planeswalker type and loyalty from the
/// game state.
#[cfg(feature = "eternal-fixtures")]
#[test]
fn flat_v9_extension_reads_planeswalkers_and_keywords() {
    let (session, observation) = observation_fixture_v1(PlayerId::P0);
    let mut state = session.game_state().clone();
    let jace = put(
        &mut state,
        PlayerId::P1,
        "Jace, the Mind Sculptor",
        Zone::Battlefield,
    );
    crate::planeswalker_v1::after_zone_change(&mut state, jace, None);
    let loyalty = crate::planeswalker_v1::loyalty(&state, jace).unwrap();
    let mut value = serde_json::to_value(&observation).unwrap();
    value["projection"]["battlefield"][1][0]["stable"]["arena_id"] = serde_json::json!(jace.0);
    value["projection"]["battlefield"][1][0]["stable"]["zone_change_count"] =
        serde_json::json!(state.objects.get(jace).zone_change_count);
    let observation: ObservationV6 = serde_json::from_value(value).unwrap();
    let extension = extension_v1(&state, &observation).unwrap();
    let index = observation.projection.surface.battlefield[0].len();
    assert!(extension.cards[index].planeswalker);
    assert_eq!(extension.cards[index].loyalty, loyalty);
    crate::planeswalker_v1::change_loyalty(&mut state, jace, 1);
    let higher = extension_v1(&state, &observation).unwrap();
    assert_eq!(higher.cards[index].loyalty, loyalty + 1);
    assert_ne!(
        columns_v1(&observation, &higher).unwrap().public_rows[1][0],
        columns_v1(&observation, &extension).unwrap().public_rows[1][0]
    );
    // Other cards are not planeswalkers.
    assert!(!extension.cards[0].planeswalker);

    // A face-down exiled planeswalker shows none of its characteristics.
    let hidden = put(
        &mut state,
        PlayerId::P1,
        "Jace, the Mind Sculptor",
        Zone::Exile,
    );
    state.objects.get_mut(hidden).v4.face_down_v1 = Some(crate::state::FaceDownV1 {
        disguised: false,
        lookers: 0,
        hidden_by: None,
        hidden_by_source: None,
    });
    let mut value = serde_json::to_value(&observation).unwrap();
    let mut exiled = value["projection"]["battlefield"][1][0].clone();
    exiled["stable"]["arena_id"] = serde_json::json!(hidden.0);
    exiled["stable"]["zone"] = serde_json::json!("Exile");
    exiled["stable"]["zone_change_count"] =
        serde_json::json!(state.objects.get(hidden).zone_change_count);
    value["projection"]["exile"]
        .as_array_mut()
        .unwrap()
        .push(exiled);
    let observation: ObservationV6 = serde_json::from_value(value).unwrap();
    let facts = extension_v1(&state, &observation).unwrap();
    let last = facts.cards.last().unwrap();
    assert_eq!(
        (last.planeswalker, last.loyalty, last.keywords),
        (false, 0, 0)
    );
}

/// Writes Rust FlatV9 cases for the Python reference
/// (`python/tests/test_features_v12.py`, `RustParityTests`) to the path in
/// `MTG_KERNEL_FLAT_V9_PARITY_OUT`. The checked-in golden fixture is written
/// only once the contract is frozen.
#[test]
fn flat_v9_parity_cases_for_python() {
    let mut cases = Vec::new();
    for actor in [PlayerId::P0, PlayerId::P1] {
        for charge in [0, 3] {
            let (session, _) = charge_fixture_v1(actor, charge);
            let d = decision_v1(&session);
            let (_, t, pairs) = v9_tensors_v1(&session).unwrap();
            let observation = session.flat_v9_observation_json_v1(d).unwrap();
            let (_, semantics) = session.diagnostic_current_decision_input_v4(d).unwrap();
            let actions: Vec<_> = semantics
                .iter()
                .enumerate()
                .map(|(index, semantic)| {
                    serde_json::json!({
                        "schema_version": 5,
                        "selected_index": index,
                        "stable_id": format!("legal-action-v5:v9-fixture-{index}"),
                        "display_text": null,
                        "semantic": semantic,
                    })
                })
                .collect();
            cases.push(serde_json::json!({
                "observation": observation,
                "legal_actions": actions,
                "tensor": {
                    "state": t.state,
                    "object_features": t.object_features,
                    "object_card_ids": t.object_card_ids,
                    "object_groups": t.object_groups,
                    "object_node_ids": t.object_node_ids,
                    "edge_features": t.edge_features,
                    "edge_source_indices": t.edge_source_indices,
                    "edge_target_indices": t.edge_target_indices,
                    "action_features": t.action_features,
                    "action_ref_features": t.action_ref_features,
                    "action_ref_card_ids": t.action_ref_card_ids,
                    "action_ref_action_indices": t.action_ref_action_indices,
                    "action_ref_node_indices": t.action_ref_node_indices,
                    "pairing_node_indices": pairs,
                },
            }));
        }
    }
    if let Ok(path) = std::env::var("MTG_KERNEL_FLAT_V9_PARITY_OUT") {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema": "flat-policy-v9-parity-cases-1",
                "feature_contract_digest": crate::native_flat_tensorizer_v9::FEATURE_CONTRACT_DIGEST_V9,
                "cases": cases,
            }))
            .unwrap(),
        )
        .unwrap();
    }
}
