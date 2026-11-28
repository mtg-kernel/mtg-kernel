//! FlatV9 observation layer (Cube M2 task T6; Python
//! `python/mtg_kernel_rl/features_v12.py`, `data/flat_policy_v9/`).
//!
//! FlatV9 is an opt-in successor of FlatV8. A FlatV9 decision is encoded in
//! two parts:
//!
//! 1. **The inherited FlatV8 tensors of a relaxed observation.** The shared
//!    flat encoder (`flat_policy_v2.rs`) refuses the public fields Cube games
//!    show and FlatV8 has no column for. While [`relaxed_scope_v1`] is held,
//!    the fast session's flat observation (`flat_policy_observation_v4`) is
//!    passed through [`sanitize_observation_v1`], which clears exactly those
//!    fields. The FlatV8 tensors are then computed unchanged from that
//!    observation, so every FlatV3 to FlatV8 encoder, digest and golden is
//!    untouched: outside the scope nothing is cleared, and every older
//!    contract keeps refusing these states.
//! 2. **FlatV9 columns**, appended to each object row and to the state row.
//!    They are computed from the unsanitized observation and a small
//!    extension of public facts the observation does not carry
//!    ([`FlatV9ExtensionV1`], serialized for Python under
//!    `extensions.flat_v9`).
//!
//! Every value here is public: printed or effective characteristics of
//! public cards, public counters, public mana pools, public permissions and
//! the public turn schedule. No hidden zone is read.


use crate::card_def::{CardType, Keywords, ManaSpendRestrictionDef};
use crate::flat_policy_v2::{FlatDecisionErrorV2, FlatObjectGroupV2, FlatObjectSourceKindV2};
use crate::policy_observation_v6::ObservationV6;
use crate::rl::{CardPublicV2, CardStableRefV1, PlayPermissionExpiryV2, PlayerSeatV1};
use crate::state::GameState;
use serde::{Deserialize, Serialize};
use std::cell::Cell;

/// Width of the FlatV9 columns appended to each object row.
pub(crate) const FLAT_V9_OBJECT_EXTRA_DIM_V1: usize = 33;
/// Width of the FlatV9 columns appended to the state row.
pub(crate) const FLAT_V9_STATE_EXTRA_DIM_V1: usize = 51;

/// The seven engine keyword bits FlatV8's fourteen keyword columns lack, in
/// mask order (`FlatV9CardFactsV1::keywords`).
pub(crate) const FLAT_V9_EXTRA_KEYWORDS_V1: [Keywords; 7] = [
    Keywords::ISLANDWALK,
    Keywords::FLASH,
    Keywords::CANT_BE_BLOCKED,
    Keywords::TOXIC_1,
    Keywords::SHROUD,
    Keywords::PROTECTION_FROM_EVERYTHING,
    Keywords::INFECT,
];

/// Bound of turn ages, tutor depth and draws since placement: values at or
/// above it read as the bound.
pub(crate) const AGE_BOUND_V1: u32 = 8;
/// Bound of the turn sequence column.
pub(crate) const TURN_SEQUENCE_BOUND_V1: u32 = 32;
/// Bound of each seat's turns-begun column.
pub(crate) const TURNS_BEGUN_BOUND_V1: u32 = 16;

thread_local! {
    static RELAXED_V1: Cell<bool> = const { Cell::new(false) };
}

/// Holds the FlatV9 relaxation for the current thread; see the module
/// documentation. Restores the previous value when dropped.
pub(crate) struct RelaxedScopeV1 {
    previous: bool,
}

impl Drop for RelaxedScopeV1 {
    fn drop(&mut self) {
        RELAXED_V1.with(|cell| cell.set(self.previous));
    }
}

/// Enters the FlatV9 relaxation for the current thread.
pub(crate) fn relaxed_scope_v1() -> RelaxedScopeV1 {
    RelaxedScopeV1 {
        previous: RELAXED_V1.with(|cell| cell.replace(true)),
    }
}

/// Whether the current thread is inside [`relaxed_scope_v1`].
pub(crate) fn relaxed_v1() -> bool {
    RELAXED_V1.with(Cell::get)
}

/// Clears the public fields FlatV9 encodes in its own columns, so that the
/// shared flat encoder admits the observation. Python mirrors this exactly
/// (`features_v12.sanitize_observation`):
///
/// - `projection.creatures_attacked_this_turn`, `projection.poison_counters`,
///   `projection.restricted_mana`, `projection.public_shroud` and
///   `projection.turn_schedule` become absent;
/// - every public card's charge counters become zero, and its
///   `creature_upgrade` and `characteristics.effective_identity` absent;
/// - every exile play permission loses `without_mana_cost`, and an
///   `UntilHoldersNextEndStep` expiry reads `EndOfTurn` (FlatV9's columns
///   carry the true values);
/// - `projection.engine_context.planeswalkers` becomes absent;
/// - under an extra-turn schedule, `projection.turn` becomes the schedule's
///   occurrence and every goad expiring at 0 expires at it.
pub(crate) fn sanitize_observation_v1(observation: &mut ObservationV6) {
    let projection = &mut observation.projection;
    // Under the extra-turn schedule every turn-keyed field holds the
    // schedule's occurrence and every goad expires at 0 ("its player's next
    // turn"); without the schedule they read against `turn`, so `turn`
    // becomes the occurrence and each such goad expires at it.
    if let Some(occurrence) = projection.turn_schedule.as_ref().map(|s| s.occurrence) {
        let surface = &mut projection.surface;
        surface.turn = occurrence;
        for card in surface
            .battlefield
            .iter_mut()
            .flatten()
            .chain(surface.graveyards.iter_mut().flatten())
            .chain(surface.exile.iter_mut())
        {
            for goad in &mut card.goaded_by {
                if goad.expires_at_turn == 0 {
                    goad.expires_at_turn = occurrence;
                }
            }
        }
    }
    // FlatV9 reads loyalty from its extension; the shared encoder ignores
    // this list and the Python FlatV7 reference refuses it.
    projection.surface.engine_context.planeswalkers = None;
    projection.creatures_attacked_this_turn = None;
    projection.poison_counters = None;
    projection.restricted_mana = None;
    // Read only by FlatV9's columns; the shared flat encoder ignores both
    // and the Python FlatV7 reference refuses them.
    projection.public_shroud = None;
    projection.turn_schedule = None;
    let surface = &mut projection.surface;
    for card in surface
        .battlefield
        .iter_mut()
        .flatten()
        .chain(surface.graveyards.iter_mut().flatten())
        .chain(surface.exile.iter_mut())
    {
        card.counters.charge = 0;
        card.creature_upgrade = None;
        card.characteristics.effective_identity = None;
    }
    for permission in &mut surface.exile_play_permissions {
        permission.without_mana_cost.0 = false;
        if permission.expiry == PlayPermissionExpiryV2::UntilHoldersNextEndStep {
            permission.expiry = PlayPermissionExpiryV2::EndOfTurn;
        }
    }
}

/// Public facts of one public card that the observation does not carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlatV9CardFactsV1 {
    /// The card it describes, equal to the observation's reference.
    pub(crate) stable: CardStableRefV1,
    /// Bit `i` is `FLAT_V9_EXTRA_KEYWORDS_V1[i]`, effective now.
    pub(crate) keywords: u8,
    /// The object has the planeswalker card type now.
    pub(crate) planeswalker: bool,
    /// Loyalty counters of a battlefield permanent, else 0.
    pub(crate) loyalty: u32,
    /// Battlefield cards: actual turns since this incarnation entered (0 this
    /// turn). None elsewhere.
    pub(crate) entered_turns_ago: Option<u32>,
    /// Plotted exiled cards: actual turns since it was plotted (0 this turn).
    /// None elsewhere.
    pub(crate) plotted_turns_ago: Option<u32>,
}

/// Public facts FlatV9 reads beside the observation, serialized for Python
/// as `extensions.flat_v9`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlatV9ExtensionV1 {
    /// One entry per public card, in observation order: seat 0's
    /// battlefield, seat 1's, seat 0's graveyard, seat 1's, then exile.
    pub(crate) cards: Vec<FlatV9CardFactsV1>,
    /// Creatures declared as attackers this turn (the observation omits the
    /// count unless a card reads it).
    pub(crate) creatures_attacked_this_turn: u32,
    /// Turn sequence, turn facts and tutor markers (`flat_v9_history_v1`).
    pub(crate) history: crate::flat_v9_history_v1::FlatV9HistoryFactsV1,
}

fn public_cards_v1(observation: &ObservationV6) -> impl Iterator<Item = &CardPublicV2> {
    let surface = &observation.projection.surface;
    surface
        .battlefield
        .iter()
        .flatten()
        .chain(surface.graveyards.iter().flatten())
        .chain(surface.exile.iter())
}

/// The extension of `observation` (unsanitized) in `state`.
pub(crate) fn extension_v1(
    state: &GameState,
    observation: &ObservationV6,
) -> Result<FlatV9ExtensionV1, FlatDecisionErrorV2> {
    let history = state
        .flat_v9_history_v1
        .as_deref()
        .ok_or(FlatDecisionErrorV2::ObservationContract)?;
    let mask = Keywords(FLAT_V9_EXTRA_KEYWORDS_V1.iter().fold(0, |all, k| all | k.0));
    let cards = public_cards_v1(observation)
        .map(|card| {
            let id = crate::ids::ObjectId(card.stable.arena_id);
            let object = state
                .objects
                .try_get(id)
                .ok_or(FlatDecisionErrorV2::InvalidReference)?;
            if object.zone_change_count != card.stable.zone_change_count {
                return Err(FlatDecisionErrorV2::InconsistentReference);
            }
            // A face-down card off the battlefield is a hidden card behind a
            // neutral sentinel row, so none of its facts are read. On the
            // battlefield the engine's effective characteristics are the
            // face-down permanent's own, public ones.
            let face_down = object.v4.face_down_v1.is_some()
                && card.stable.zone != crate::state::Zone::Battlefield;
            let present = if face_down {
                Keywords::NONE
            } else {
                crate::engine::effective_keywords(state, id, mask)
            };
            let keywords = FLAT_V9_EXTRA_KEYWORDS_V1
                .iter()
                .enumerate()
                .fold(0_u8, |bits, (index, keyword)| {
                    bits | (u8::from(present.has(*keyword)) << index)
                });
            let (arena, generation) = (card.stable.arena_id, card.stable.zone_change_count);
            Ok(FlatV9CardFactsV1 {
                stable: card.stable.clone(),
                keywords,
                planeswalker: !face_down
                    && crate::engine::object_has_type(state, id, CardType::Planeswalker),
                loyalty: if face_down {
                    0
                } else {
                    crate::planeswalker_v1::loyalty(state, id).unwrap_or(0)
                },
                entered_turns_ago: (card.stable.zone == crate::state::Zone::Battlefield)
                    .then(|| history.entered_turns_ago_v1(arena, generation)),
                plotted_turns_ago: (card.stable.zone == crate::state::Zone::Exile
                    && card.plotted_turn.is_some())
                .then(|| history.plotted_turns_ago_v1(arena, generation)),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(FlatV9ExtensionV1 {
        cards,
        creatures_attacked_this_turn: state.creatures_attacked_this_turn_v1(),
        history: history.facts_v1(state),
    })
}

/// Restriction classes of restricted mana (state columns 12 to 19).
fn restriction_class_v1(restriction: ManaSpendRestrictionDef) -> usize {
    use ManaSpendRestrictionDef as R;
    match restriction {
        R::CreatureSpell
        | R::CreatureSpellOrCreatureAbility
        | R::CreatureSpellOfChosenTypeV1
        | R::ChosenCreatureTypeV60(_) => 0,
        R::ArtifactSpellV1 | R::PowerstoneV1 => 1,
        R::NoncreatureSpellV1 => 2,
        R::LegendarySpell
        | R::Unrestricted
        | R::ColorlessEldraziSpellV1
        | R::SpellWithSubtype(_) => 3,
    }
}

fn scaled_v1(value: i64, scale: f64) -> f32 {
    (value as f64 / scale) as f32
}

fn seat_index_v1(seat: PlayerSeatV1) -> usize {
    match seat {
        PlayerSeatV1::P0 => 0,
        PlayerSeatV1::P1 => 1,
    }
}

/// The FlatV9 columns of one decision: one row per public card, by its
/// actor-relative group and position, and the state row.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct FlatV9ColumnsV1 {
    /// Self battlefield, opponent battlefield, self graveyard, opponent
    /// graveyard, exile, each in observation order.
    pub(crate) public_rows: [Vec<[f32; FLAT_V9_OBJECT_EXTRA_DIM_V1]>; 5],
    pub(crate) state: Vec<f32>,
}

impl FlatV9ColumnsV1 {
    /// The FlatV9 object columns of a flat object row: a public card's own
    /// row, zeros for every other row (hand, stack, context, known cards).
    pub(crate) fn object_row_v1(
        &self,
        object: &crate::flat_policy_v2::FlatObjectCoreV2,
    ) -> Result<[f32; FLAT_V9_OBJECT_EXTRA_DIM_V1], FlatDecisionErrorV2> {
        let list = match object.group {
            FlatObjectGroupV2::SelfBattlefield => 0,
            FlatObjectGroupV2::OpponentBattlefield => 1,
            FlatObjectGroupV2::SelfGraveyard => 2,
            FlatObjectGroupV2::OpponentGraveyard => 3,
            FlatObjectGroupV2::Exile => 4,
            _ => return Ok([0.0; FLAT_V9_OBJECT_EXTRA_DIM_V1]),
        };
        if object.source_kind != FlatObjectSourceKindV2::Card || !object.card_details_present {
            return Ok([0.0; FLAT_V9_OBJECT_EXTRA_DIM_V1]);
        }
        let ordinal = usize::try_from(object.visible_ordinal)
            .map_err(|_| FlatDecisionErrorV2::CheckedIntegerRange)?;
        self.public_rows[list]
            .get(ordinal)
            .copied()
            .ok_or(FlatDecisionErrorV2::InvalidReference)
    }
}

fn same_card_v1(left: &CardStableRefV1, right: &CardStableRefV1) -> bool {
    left.arena_id == right.arena_id && left.zone_change_count == right.zone_change_count
}

fn card_row_v1(
    card: &CardPublicV2,
    facts: &FlatV9CardFactsV1,
    observation: &ObservationV6,
) -> Result<[f32; FLAT_V9_OBJECT_EXTRA_DIM_V1], FlatDecisionErrorV2> {
    if facts.stable != card.stable {
        return Err(FlatDecisionErrorV2::InconsistentReference);
    }
    let actor = observation.acting_player;
    let mut row = [0.0_f32; FLAT_V9_OBJECT_EXTRA_DIM_V1];
    row[0] = f32::from(facts.planeswalker);
    row[1] = scaled_v1(i64::from(facts.loyalty), 10.0);
    row[2] = scaled_v1(i64::from(card.counters.charge), 10.0);
    for bit in 0..FLAT_V9_EXTRA_KEYWORDS_V1.len() {
        row[3 + bit] = f32::from(facts.keywords & (1 << bit) != 0);
    }
    for permission in &observation.projection.surface.exile_play_permissions {
        if !same_card_v1(&permission.object, &card.stable) {
            continue;
        }
        if permission.holder == actor {
            row[10] = 1.0;
        } else {
            row[11] = 1.0;
        }
        if permission.without_mana_cost.0 {
            row[12] = 1.0;
        }
        if permission.expiry == PlayPermissionExpiryV2::UntilHoldersNextEndStep {
            row[13] = 1.0;
        }
    }
    if let Some(identity) = &card.characteristics.effective_identity {
        row[14] = 1.0;
        row[15] = scaled_v1(i64::from(identity.mana_value), 10.0);
        row[16] = f32::from(identity.supertypes.iter().any(|s| s == "Legendary"));
        row[17] = scaled_v1(
            i64::try_from(identity.names.len())
                .map_err(|_| FlatDecisionErrorV2::CheckedIntegerRange)?,
            4.0,
        );
    }
    if let Some(upgrade) = &card.creature_upgrade {
        row[18] = 1.0;
        row[19] = scaled_v1(i64::from(upgrade.finality), 10.0);
        row[20] = f32::from(upgrade.temporary_creature.is_some());
        row[21] = scaled_v1(
            i64::try_from(upgrade.keyword_grants.len())
                .map_err(|_| FlatDecisionErrorV2::CheckedIntegerRange)?,
            8.0,
        );
        row[22] = scaled_v1(
            i64::try_from(upgrade.keyword_losses.len())
                .map_err(|_| FlatDecisionErrorV2::CheckedIntegerRange)?,
            8.0,
        );
        row[23] = f32::from(upgrade.combat_impulse.is_some());
        row[24] = f32::from(upgrade.haste_blockers_only);
    }
    for (first, age) in [(25, facts.entered_turns_ago), (29, facts.plotted_turns_ago)] {
        if let Some(age) = age {
            row[first] = 1.0;
            row[first + 1] = f32::from(age == 0);
            row[first + 2] = f32::from(age == 1);
            row[first + 3] = scaled_v1(i64::from(age.min(AGE_BOUND_V1)), f64::from(AGE_BOUND_V1));
        }
    }
    Ok(row)
}

/// The FlatV9 columns of `observation` (unsanitized) and its extension.
pub(crate) fn columns_v1(
    observation: &ObservationV6,
    extension: &FlatV9ExtensionV1,
) -> Result<FlatV9ColumnsV1, FlatDecisionErrorV2> {
    let actor = observation.acting_player;
    let me = seat_index_v1(actor);
    let relative = [me, 1 - me];
    let surface = &observation.projection.surface;
    // Extension entries in absolute observation order.
    let mut facts = extension.cards.iter();
    let mut absolute: [Vec<[f32; FLAT_V9_OBJECT_EXTRA_DIM_V1]>; 5] = Default::default();
    let lists: [&[CardPublicV2]; 5] = [
        &surface.battlefield[0],
        &surface.battlefield[1],
        &surface.graveyards[0],
        &surface.graveyards[1],
        &surface.exile,
    ];
    for (index, list) in lists.iter().enumerate() {
        for card in list.iter() {
            let fact = facts
                .next()
                .ok_or(FlatDecisionErrorV2::InconsistentReference)?;
            absolute[index].push(card_row_v1(card, fact, observation)?);
        }
    }
    if facts.next().is_some() {
        return Err(FlatDecisionErrorV2::InconsistentReference);
    }
    let [b0, b1, g0, g1, exile] = absolute;
    let (battlefields, graveyards) = ([b0, b1], [g0, g1]);
    let public_rows = [
        battlefields[relative[0]].clone(),
        battlefields[relative[1]].clone(),
        graveyards[relative[0]].clone(),
        graveyards[relative[1]].clone(),
        exile,
    ];

    let mut state = vec![0.0_f32; FLAT_V9_STATE_EXTRA_DIM_V1];
    if let Some(pools) = &observation.projection.restricted_mana {
        let mut colors = [[0_i64; 6]; 2];
        let mut classes = [[0_i64; 4]; 2];
        for (order, &seat) in relative.iter().enumerate() {
            for unit in &pools[seat] {
                colors[order][unit.color.pool_index()] += 1;
                classes[order][restriction_class_v1(unit.restriction)] += 1;
            }
        }
        for order in 0..2 {
            for color in 0..6 {
                state[order * 6 + color] = scaled_v1(colors[order][color], 10.0);
            }
            for class in 0..4 {
                state[12 + order * 4 + class] = scaled_v1(classes[order][class], 10.0);
            }
        }
    }
    state[20] = scaled_v1(i64::from(extension.creatures_attacked_this_turn), 10.0);
    if let Some(poison) = observation.projection.poison_counters {
        state[21] = scaled_v1(i64::from(poison[relative[0]]), 10.0);
        state[22] = scaled_v1(i64::from(poison[relative[1]]), 10.0);
    }
    if let Some(shroud) = &observation.projection.public_shroud {
        state[23] = f32::from(shroud.players[relative[0]]);
        state[24] = f32::from(shroud.players[relative[1]]);
    }
    if let Some(schedule) = &observation.projection.turn_schedule {
        state[25] = f32::from(schedule.current_extra.is_some());
        let mut queued = [0_i64; 2];
        for turn in &schedule.queued_next_first {
            queued[usize::from(seat_index_v1(turn.captured_controller) != me)] += 1;
        }
        state[26] = scaled_v1(queued[0], 4.0);
        state[27] = scaled_v1(queued[1], 4.0);
    }
    let history = &extension.history;
    state[28] = scaled_v1(
        i64::from(history.turn_sequence.min(TURN_SEQUENCE_BOUND_V1)),
        f64::from(TURN_SEQUENCE_BOUND_V1),
    );
    for (order, &seat) in relative.iter().enumerate() {
        state[29 + order] = scaled_v1(
            i64::from(history.turns_begun[seat].min(TURNS_BEGUN_BOUND_V1)),
            f64::from(TURNS_BEGUN_BOUND_V1),
        );
        state[32 + order] = f32::from(history.attacked[seat]);
        state[35 + order] = f32::from(history.lost_life[seat]);
        let tutor = history.tutor[seat];
        let first = 37 + order * 7;
        use crate::flat_v9_history_v1::TutorStatusV1 as T;
        let status = match tutor.status {
            T::None => 0,
            T::Tracked if tutor.depth == 0 => 1,
            T::Tracked => 2,
            T::DrawnOrRemoved => 3,
            T::Invalidated => 4,
        };
        state[first + status] = 1.0;
        state[first + 5] = scaled_v1(
            i64::from(tutor.depth.min(AGE_BOUND_V1)),
            f64::from(AGE_BOUND_V1),
        );
        state[first + 6] = scaled_v1(
            i64::from(tutor.draws_since.min(AGE_BOUND_V1)),
            f64::from(AGE_BOUND_V1),
        );
    }
    state[31] = f32::from(history.consecutive_turn);
    state[34] = f32::from(history.creature_died);
    Ok(FlatV9ColumnsV1 { public_rows, state })
}

/// A FlatV9 decision: the FlatV8 (V4-generation) decision of the relaxed
/// observation, the FlatV9 extension and columns.
#[derive(Debug, Clone)]
pub(crate) struct FlatDecisionV9 {
    pub(crate) v4: crate::flat_policy_v4::FlatDecisionV4,
    pub(crate) extension: FlatV9ExtensionV1,
    pub(crate) columns: FlatV9ColumnsV1,
}

/// Borrowed FlatV9 scorer view: the FlatV4-generation view and the FlatV9
/// columns of the same decision.
#[derive(Clone, Copy)]
pub(crate) struct FlatScoringDecisionViewV9<'a> {
    pub(crate) v4: crate::flat_policy_v4::FlatScoringDecisionViewV4<'a>,
    pub(crate) columns: &'a FlatV9ColumnsV1,
}

impl crate::rl_session::FastActorSessionV1 {
    /// FlatV9 sibling of `encode_current_flat_scoring_decision_owned_v4`
    /// (module documentation). The unsanitized observation is read before
    /// the relaxed encode; both describe the same live decision.
    pub(crate) fn encode_current_flat_scoring_decision_owned_v9(
        &self,
        expected: crate::rl_session::FastActorDecisionV1,
        encoder: &mut crate::flat_policy_v4::FlatDecisionEncoderV4,
        buffers: &mut crate::flat_policy_v2::FlatScoringOwnedBuffersV2<'_>,
    ) -> Result<FlatDecisionV9, FlatDecisionErrorV2> {
        let observation = self.flat_policy_observation_raw_v4(expected)?;
        let extension = extension_v1(self.game_state(), &observation)?;
        let columns = columns_v1(&observation, &extension)?;
        let v4 = {
            let _relaxed = relaxed_scope_v1();
            self.encode_current_flat_scoring_decision_owned_v4(expected, encoder, buffers)?
        };
        Ok(FlatDecisionV9 {
            v4,
            extension,
            columns,
        })
    }

    /// The unsanitized observation with its FlatV9 extension under
    /// `extensions.flat_v9`, as the Python reference reads it.
    pub(crate) fn flat_v9_observation_json_v1(
        &self,
        expected: crate::rl_session::FastActorDecisionV1,
    ) -> Result<serde_json::Value, FlatDecisionErrorV2> {
        let mut observation = self.flat_policy_observation_raw_v4(expected)?;
        let extension = extension_v1(self.game_state(), &observation)?;
        observation.visible_projection_hash = 0;
        let mut value = serde_json::to_value(&observation)
            .map_err(|_| FlatDecisionErrorV2::ObservationContract)?;
        value["extensions"]["flat_v9"] = serde_json::to_value(&extension)
            .map_err(|_| FlatDecisionErrorV2::ObservationContract)?;
        // A London pregame decision's facts, as the FlatV7 reference reads
        // them.
        if let Some(view) =
            crate::rl_session::pauper_london_v1::london_pregame_view_v1(self.game_state())
                .map_err(|_| FlatDecisionErrorV2::ObservationContract)?
        {
            use crate::rl_session::pauper_london_v1::LondonPregamePhaseV1;
            value["extensions"]["london_pregame"] = serde_json::json!({
                "phase": if view.phase == LondonPregamePhaseV1::Bottom { "bottom" } else { "announce" },
                "mulligans_taken": view.counts,
                "kept": view.kept,
                "announced_mulligan": view.mulligan_this_round,
            });
        }
        Ok(value)
    }
}

#[cfg(test)]
#[path = "flat_policy_v6_tests.rs"]
mod tests;
