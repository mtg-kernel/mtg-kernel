//! Recover the count at entry to a staged draw from its generating effect and committed events.
use super::*;

/// Mark the beginning of the generating batch. Its zone changes, including
/// replacements to exile, determine the count independently of the draw frame.
pub(crate) fn begin_draw_input_batch(
    state: &mut GameState,
    pending: &EffectContinuation,
    path: &[u16],
    player: PlayerId,
    inputs: impl Iterator<Item = ObjectId> + Clone,
) {
    if !crate::eternal_cards_v1::owned_draws_v1::staged::tracks(state, pending, player) {
        return;
    }
    if crate::eternal_legacy_e5b_v1::next_draw_may_offer_miracle(state, player)
        || crate::eternal_vintage_e7b_v1::dredge::may_need_draw_choice(state, player)
        || inputs.clone().any(|id| {
            let card = state.objects.get(id);
            card.owner == player
                && crate::eternal_vintage_e7b_v1::dredge::dredge_count(
                    crate::card_def::CARD_DEFS[usize::from(card.card_def)].name,
                )
                .is_some()
        })
    {
        use crate::eternal_cards_v1::owned_draws_v1::{record, JournalV1};
        record(
            state,
            JournalV1::DrawInputBatchV1 {
                item: pending.resolving_item.v4.stack_item_id,
                path: path.to_vec(),
                objects: inputs.collect(),
            },
        );
    }
}

pub(super) fn staged_batch_count(
    state: &GameState,
    ctx: &ExecCtx,
    path: &[u16],
    end: usize,
    zone: Zone,
    owner: Option<PlayerId>,
    maximum: u32,
) -> Option<u32> {
    use crate::eternal_cards_v1::owned_draws_v1::JournalV1;
    use event::CommittedEvent;
    let history = state.engine.event_history.get(..end)?;
    let batch = history.iter().enumerate().rev().find_map(|(at, row)| {
        let CommittedEvent::EternalOwnedDrawV1 { journal } = row else {
            return None;
        };
        let JournalV1::DrawInputBatchV1 {
            item,
            path: recorded,
            objects,
        } = journal.as_ref()
        else {
            return None;
        };
        (Some(*item) == ctx.stack_item_id && recorded == path).then_some((at, objects))
    });
    let moved = |row: &CommittedEvent| match row {
        CommittedEvent::ZoneChange {
            object, from, to, ..
        } if *from == zone
            && from != to
            && owner.is_none_or(|player| {
                state
                    .objects
                    .try_get(*object)
                    .is_some_and(|card| card.owner == player)
            }) =>
        {
            Some(*object)
        }
        _ => None,
    };
    // Unmarked saves used the immediately preceding batch of departures.
    let count = match batch {
        Some((at, objects)) => history[at + 1..]
            .iter()
            .filter_map(moved)
            .filter(|object| objects.contains(object))
            .count(),
        None => (history
            .iter()
            .rev()
            .filter(|row| {
                matches!(
                    row,
                    CommittedEvent::ZoneChange { .. }
                        | CommittedEvent::Draw { .. }
                        | CommittedEvent::SpellCast { .. }
                        | CommittedEvent::Damage { .. }
                        | CommittedEvent::LifeLoss { .. }
                        | CommittedEvent::LifeGain { .. }
                        | CommittedEvent::ManaAdded { .. }
                        | CommittedEvent::Tap { .. }
                        | CommittedEvent::Sacrificed { .. }
                )
            })
            .take_while(|row| moved(row).is_some())
            .count())
        .min(maximum as usize),
    };
    u32::try_from(count).ok().filter(|count| *count <= maximum)
}

pub(super) fn staged_dynamic_count(
    state: &GameState,
    ctx: &ExecCtx,
    value: crate::card_def::DynamicValueDef,
    start: usize,
) -> Option<u32> {
    use crate::card_def::DynamicValueDef::*;
    match value {
        GraveyardCardsNamedV1(id)
            if crate::card_def::CARD_DEFS.get(usize::from(id))?.name == "Accumulated Knowledge" =>
        {
            crate::eternal_cards_v1::owned_draws_v1::knowledge::graveyard_count_at(state, start)
        }
        Fixed(_)
        | ControlledPermanentsWithSubtype(_)
        | DistinctManaValuesAmongControlledNonlandPermanents => {
            Some(crate::engine::evaluate_dynamic_value(state, value, ctx.controller).max(0) as u32)
        }
        _ => None,
    }
}

#[cfg(feature = "eternal-fixtures")]
pub(super) fn draw_total_at(
    state: &GameState,
    player: PlayerId,
    base: u32,
    start: usize,
) -> Option<u32> {
    let hand = crate::eternal_cards_v1::owned_draws_v1::initial_hand_len(state, player, start)?;
    base.checked_add(event::draw_instruction_extra_at_hand_size(
        state, player, base, hand,
    ))
}

/// Old replacement choices predate the generator and attempt journals. When
/// resuming one, retain the number of attempts the released interpreter offered.
/// Subsequent saves charge each attempt against that same offer.
#[cfg(feature = "eternal-fixtures")]
pub(crate) fn retain_legacy_draw_offer(
    state: &mut GameState,
    pending: &EffectContinuation,
    choice: &crate::eternal_cards_v1::e3::E3ChoiceV1,
) {
    use crate::eternal_cards_v1::owned_draws_v1::staged;
    if state.eternal_owned_draw_progress_v1.is_some()
        || !staged::tracks(state, pending, choice.player)
        || !staged::unmarked_before(state, pending, state.engine.event_history.len())
        || definition_has_nonterminal_discard(state, pending)
    {
        return;
    }
    let Ok(root) = validated_definition_owned_root_effect(state, pending) else {
        return;
    };
    if legacy_choice_matches(
        state,
        pending,
        &root,
        &choice.path,
        choice.player,
        state.engine.event_history.len(),
        draw_choice_attempts(choice),
    ) != Some(true)
    {
        return;
    }
    state
        .engine
        .event_history
        .push(event::CommittedEvent::ResolutionDrawChoiceOfferedV1 {
            stack_item_id: pending.resolving_item.v4.stack_item_id,
            path: choice.path.clone(),
            player: choice.player,
            attempts: draw_choice_attempts(choice),
        });
}

/// The draw kinds whose replacement choices the released interpreter paused
/// without per-attempt receipts.
#[cfg(feature = "eternal-fixtures")]
#[derive(Clone, Copy)]
struct LegacySite {
    /// A count fixed by a discard or destruction batch.
    batch: bool,
    /// Sylvan Library's draw, after its controller's turn draw.
    sylvan: bool,
    /// Standstill, first marked at its generator after #1031.
    standstill: bool,
    /// A count generated by its resolving leaf (Peer, Sylvan, Standstill).
    generated: bool,
    /// A printed draw instruction.
    printed: bool,
}

#[cfg(feature = "eternal-fixtures")]
fn legacy_site(
    state: &GameState,
    pending: &EffectContinuation,
    root: &EffectOp,
    path: &[u16],
) -> Option<LegacySite> {
    use crate::eternal_cards_v1::EternalOpV1 as E;
    use crate::eternal_premodern_p1a_v1::P1aOpV1;
    use crate::eternal_premodern_p3h_v1::P3hOpV1;
    use crate::eternal_sideboard_s1b_v1::S1bOpV1;
    use crate::standard_cards_v1::StandardOpV1 as S;
    let leaf = resolved_definition_op_at(root, path, &pending.ctx, state);
    let batch = matches!(
        leaf,
        Some(
            EffectOp::DiscardUpToThenDraw { .. }
                | EffectOp::StandardV1(
                    S::MayDiscardUpToThenDraw { .. }
                        | S::EternalV1(E::P3hV1(P3hOpV1::DecreeOfPainV1))
                )
        )
    );
    let sylvan = path == [0]
        && pending
            .resolving_item
            .v4
            .ability_source_contract
            .is_some_and(|s| {
                crate::card_def::CARD_DEFS[usize::from(s.card_def)].name == "Sylvan Library"
            });
    let standstill = matches!(
        leaf,
        Some(EffectOp::StandardV1(S::EternalV1(E::P1bV1(
            crate::eternal_premodern_p1b_v1::P1bOpV1::StandstillV1 { .. }
        ))))
    );
    let generated = sylvan
        || standstill
        || matches!(
            leaf,
            Some(EffectOp::StandardV1(S::EternalV1(
                E::S1bV1(S1bOpV1::PeerV1) | E::P1aV1(P1aOpV1::SylvanDrawV1)
            )))
        );
    let printed = matches!(
        leaf,
        Some(EffectOp::DrawCards { .. } | EffectOp::DrawCardsDynamic { .. })
    );
    (batch || generated || printed).then_some(LegacySite {
        batch,
        sylvan,
        standstill,
        generated,
        printed,
    })
}

/// Match the released choice against the definition and the events available
/// at the offer. Its non-native loop staged the printed count, with Riddler
/// replacing individual draws. When a response or the turn draw lacks a
/// resolution boundary, try only suffixes of observed completed attempts.
/// `None` when released history cannot identify such an offer at `end`.
#[cfg(feature = "eternal-fixtures")]
fn legacy_choice_matches(
    state: &GameState,
    pending: &EffectContinuation,
    root: &EffectOp,
    path: &[u16],
    drawer: PlayerId,
    end: usize,
    offered: u32,
) -> Option<bool> {
    use crate::eternal_cards_v1::owned_draws_v1::staged;
    let site = legacy_site(state, pending, root, path)?;
    let mut starts = Vec::new();
    let (_, spent) = staged::visit_attempts(state, pending, path, end, true, |at| starts.push(at));
    let item = &pending.resolving_item;
    if !(site.batch
        || site.generated
        || (site.printed
            && (spent > 0 || event::draw_instruction_extra_at_hand_size(state, drawer, 1, 0) > 0)))
    {
        return None;
    }
    if offered == 0 {
        return Some(false);
    }
    starts.push(end);
    for (skip, start) in starts.into_iter().enumerate() {
        if skip > 0
            && (site.batch
                || site.standstill
                || !(site.sylvan || item.v4.stack_item_id.0 < state.engine.next_stack_item_id))
        {
            break;
        }
        let Some((player, base, _)) = draw_site(state, pending, root, path, start) else {
            continue;
        };
        if pending.ctx.resolve_player(player, state) != drawer {
            return Some(false);
        }
        let remaining = base.checked_sub(spent - skip as u32);
        // Contiguous departures give an upper bound when two batches touch.
        if remaining == Some(offered)
            || (site.batch && remaining.is_some_and(|limit| offered <= limit))
        {
            return Some(true);
        }
    }
    Some(false)
}

/// Locate the first current attempt at this site. An immediately preceding
/// retained offer binds the balance for the rest of that legacy resolution.
/// Recheck the offer against the pre-journal prefix even after another restore.
/// A journal resumed from a released choice must keep that offer: without it,
/// the count would fall back to one the released history never supported.
#[cfg(feature = "eternal-fixtures")]
pub(super) fn restored_balance(
    state: &GameState,
    pending: &EffectContinuation,
    root: &EffectOp,
    path: &[u16],
) -> Result<Option<u32>, String> {
    use crate::eternal_cards_v1::owned_draws_v1::staged;
    let journal = staged::journal_attempts(state, pending, path);
    let Some(first) = journal.first else {
        return Ok(None);
    };
    if journal.entered || definition_has_nonterminal_discard(state, pending) {
        return Ok(None);
    }
    let offer = first
        .checked_sub(1)
        .and_then(|at| match &state.engine.event_history[at] {
            event::CommittedEvent::ResolutionDrawChoiceOfferedV1 {
                stack_item_id,
                path: site,
                attempts,
                player,
            } if *stack_item_id == pending.resolving_item.v4.stack_item_id && site == path => {
                Some((at, *attempts, *player))
            }
            _ => None,
        });
    let Some((offer_at, attempts, player)) = offer else {
        return if resumed_released_choice(state, pending, root, path, first) {
            Err("restored draw lost its original replacement offer".into())
        } else {
            Ok(None)
        };
    };
    if !staged::unmarked_before(state, pending, offer_at)
        || legacy_choice_matches(state, pending, root, path, player, offer_at, attempts)
            != Some(true)
    {
        return Err("restored draw lost its original replacement offer".into());
    }
    attempts
        .checked_sub(journal.charged)
        .map(Some)
        .ok_or_else(|| "restored draw exceeded its offered attempts".into())
}

/// Whether an unmarked journal starting at `first` continues a released
/// replacement choice. This engine marks batch draws and #1031's generated
/// draws before any attempt that can pause (`begin_draw_input_batch`,
/// `enter_generated`), and a printed draw before a first-attempt choice
/// (`enter_printed_choice`). An unmarked printed journal therefore starts
/// with an ordinary attempt, and with Riddler only a first Miracle draw
/// pauses it (a possible dredge starts the native instruction). Earlier
/// attempts at this site before the journal also prove a released start.
/// A released Miracle stop marks its item and is resumed by its own count.
#[cfg(feature = "eternal-fixtures")]
fn resumed_released_choice(
    state: &GameState,
    pending: &EffectContinuation,
    root: &EffectOp,
    path: &[u16],
    first: usize,
) -> bool {
    use crate::eternal_cards_v1::owned_draws_v1::{initial_hand_len, staged};
    if !staged::unmarked_before(state, pending, first) {
        return false;
    }
    let Some(site) = legacy_site(state, pending, root, path) else {
        return false;
    };
    if site.batch || (site.generated && !site.standstill) {
        return true;
    }
    if staged::visit_attempts(state, pending, path, first, true, |_| {}).1 > 0 {
        return true;
    }
    if !site.printed {
        return false;
    }
    let Some((player, _, _)) = draw_site(state, pending, root, path, first) else {
        return false;
    };
    let drawer = pending.ctx.resolve_player(player, state);
    let history = &state.engine.event_history;
    let dredged = matches!(
        history.get(first + 1),
        Some(event::CommittedEvent::ZoneChange {
            object,
            from: Zone::Library,
            to: Zone::Graveyard | Zone::Exile,
            ..
        }) if state.objects.try_get(*object).is_some_and(|card| card.owner == drawer)
    );
    let item = pending.resolving_item.v4.stack_item_id;
    let suspended = history[first..].iter().any(|row| {
        matches!(row, event::CommittedEvent::MiracleDrawSuspendedV1 { stack_item_id, path: site, .. }
            if *stack_item_id == item && site == path)
    });
    let riddler = initial_hand_len(state, drawer, first)
        .is_some_and(|hand| event::draw_instruction_extra_at_hand_size(state, drawer, 1, hand) > 0);
    dredged || (riddler && !suspended)
}

/// Mark a newly offered printed draw before its first replacement. A loaded
/// choice resumes through the answer handler and never enters here.
#[cfg(feature = "eternal-fixtures")]
pub(crate) fn enter_printed_choice(
    state: &mut GameState,
    pending: &EffectContinuation,
    choice: &crate::eternal_cards_v1::e3::E3ChoiceV1,
) {
    use crate::eternal_cards_v1::owned_draws_v1::{record, staged, JournalV1};
    if state.eternal_owned_draw_progress_v1.is_some()
        || !staged::tracks(state, pending, choice.player)
        || !staged::unmarked_before(state, pending, state.engine.event_history.len())
    {
        return;
    }
    let Ok(root) = validated_definition_owned_root_effect(state, pending) else {
        return;
    };
    if matches!(
        resolved_definition_op_at(&root, &choice.path, &pending.ctx, state),
        Some(EffectOp::DrawCards { .. } | EffectOp::DrawCardsDynamic { .. })
    ) {
        record(
            state,
            JournalV1::GeneratedDrawV1 {
                item: pending.resolving_item.v4.stack_item_id,
                path: choice.path.clone(),
            },
        );
    }
}

/// Validate both an open replacement choice and its saved answer before the
/// interpreter spends that attempt. Native instructions validate their own cursor.
#[cfg(feature = "eternal-fixtures")]
pub(super) fn validate_staged_draw_choice(
    state: &GameState,
    pending: &EffectContinuation,
) -> Result<(), String> {
    use crate::eternal_cards_v1::e3::E3ChoiceKindV1;
    use crate::eternal_vintage_e7b_v1::E7bChoiceV1;
    if state.eternal_owned_draw_progress_v1.is_some() {
        return Ok(());
    }
    let (choice, frames) = match (&pending.choice, pending.frames.split_last()) {
        (
            Some(PendingEffectChoice::SelectTargets {
                purpose: EffectTargetSelectionPurpose::EternalV1 { choice },
                ..
            }),
            _,
        ) => (choice, pending.frames.as_slice()),
        (None, Some((EffectFrame::EternalSelectionV1 { choice, .. }, rest))) => (choice, rest),
        _ => return Ok(()),
    };
    if !matches!(
        choice.kind,
        E3ChoiceKindV1::E7bV1(E7bChoiceV1::DredgeV1 | E7bChoiceV1::JaceOrHullbreacherV1)
    ) {
        return Ok(());
    }
    if !crate::eternal_cards_v1::owned_draws_v1::staged::tracks(state, pending, choice.player) {
        return Ok(());
    }
    let Ok(root) = validated_definition_owned_root_effect(state, pending) else {
        return Ok(());
    };
    let (first, spent) =
        crate::eternal_cards_v1::owned_draws_v1::staged::attempts(state, pending, &choice.path);
    let start = first.unwrap_or(state.engine.event_history.len());
    let Some((player, base, mut expected)) = draw_site(state, pending, &root, &choice.path, start)
    else {
        return if first.is_some()
            || crate::eternal_cards_v1::owned_draws_v1::staged::generated_path(state, pending)
                .is_some()
        {
            Err("replacement choice lost its draw site".into())
        } else {
            Ok(())
        };
    };
    let drawer = pending.ctx.resolve_player(player, state);
    use crate::eternal_cards_v1::owned_draws_v1::staged;
    let offered = draw_choice_attempts(choice);
    // A released choice identified by its history must carry that count;
    // the native total below would ignore its unjournaled earlier attempts.
    let end = state.engine.event_history.len();
    let released = staged::unmarked_before(state, pending, end)
        .then(|| legacy_choice_matches(state, pending, &root, &choice.path, drawer, end, offered))
        .flatten();
    let outstanding = match released {
        Some(true) => Some(offered),
        Some(false) => return Err("released draw choice changed its remaining draw count".into()),
        None => restored_balance(state, pending, &root, &choice.path)?.or_else(|| {
            draw_total_at(state, drawer, base, start).and_then(|total| total.checked_sub(spent))
        }),
    };
    let after = outstanding
        .and_then(|count| count.checked_sub(1))
        .ok_or("staged draw has no outstanding attempt")?;
    if after > 0 {
        expected.push(EffectFrame::Program {
            op: EffectOp::DrawCards {
                player,
                count: after,
            },
            path: choice.path.clone(),
        });
    }
    if drawer != choice.player || frames != expected || choice.remaining != expected {
        return Err("replacement choice changed its remaining draw count".into());
    }
    Ok(())
}
