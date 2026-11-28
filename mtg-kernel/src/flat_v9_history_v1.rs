//! FlatV9 history (Cube M2 task T6; ruling on DECISIONS-1, "FlatV9 history
//! features", candidates 1 and 2). An opt-in record on the game state,
//! absent in every other game, so their bytes, hashes and goldens are
//! unchanged (`GameState::flat_v9_history_v1`).
//!
//! - **Turn sequence (issue 740).** `turn_sequence` counts the actual turns
//!   begun, the current one included, and advances when each turn begins:
//!   both players' turns of a round, extra turns and consecutive turns of one
//!   player alike. A turn skipped by Time Vault is not begun and does not
//!   count.
//! - **Exact entry and plot stamps.** When a turn begins, every battlefield
//!   incarnation without a stamp entered during the turn that just ended and
//!   is stamped with it; likewise every plotted exiled incarnation. An
//!   incarnation without a stamp therefore entered (or was plotted) this
//!   turn. A zone change makes a new incarnation, so re-entry starts fresh.
//!   Stamps of incarnations that left are dropped at each turn boundary.
//! - **Turn facts.** Turns begun by each seat and whether the current turn's
//!   seat also took the previous turn.
//! - **Tutor marker (candidate 1).** Per library, an identity-free marker of
//!   the most recent card a resolving tutor publicly put on top (Vampiric
//!   Tutor, Imperial Seal, Mystical Tutor, Enlightened Tutor, Sterling
//!   Grove). It follows the card's public position only: a draw takes the
//!   top card, so it marks the card drawn at depth 0 and moves it up
//!   otherwise; every other removal from that library (a mill, an exile, a
//!   search or a private pick, whose position may be secret) invalidates it;
//!   an insertion at or above it moves it down; a shuffle, a partial
//!   randomization covering it, or a reorder or multi-card scry whose result
//!   is private and covers it invalidates it; a one-card scry's public
//!   keep-or-bottom result is followed.
//!   No object identity is stored or read, so two worlds with the same
//!   public history have the same marker even where a private operation
//!   moved the marked card in only one of them.
//!
//! Draws since placement count the owner's draws (`GameState::draw_card`).

// The FlatV9 policy (T6, next change) reads the facts in production; until
// then only the tests do.
#![cfg_attr(not(test), allow(dead_code))]

use crate::ids::PlayerId;
use crate::state::{GameState, ObservationNoteKindV1, Zone};
use serde::{Deserialize, Serialize};

/// An incarnation's exact turn stamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnStampV1 {
    pub object: u32,
    pub zone_change_count: u32,
    /// The turn sequence number of the turn it entered (or was plotted);
    /// 0 before the record began.
    pub turn_sequence: u32,
}

/// Public status of one library's tutor marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TutorStatusV1 {
    /// No tutor has put a card on top of this library.
    #[default]
    None,
    /// The marked card is at public position `depth` (0 is the top).
    Tracked,
    /// The marked card left the library from its public position.
    DrawnOrRemoved,
    /// A shuffle or a private reorder covering it ended the tracking.
    Invalidated,
}

/// One library's identity-free tutor marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TutorMarkerV1 {
    pub status: TutorStatusV1,
    /// Cards publicly above the marked card while tracked, else 0.
    pub depth: u32,
    /// The owner's draw count when the card was placed.
    pub draws_at_placement: u32,
}

/// The FlatV9 history of one game (module documentation).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatV9HistoryV1 {
    pub turn_sequence: u32,
    pub turns_begun: [u32; 2],
    pub current_active: Option<PlayerId>,
    pub previous_active: Option<PlayerId>,
    pub entries: Vec<TurnStampV1>,
    pub plots: Vec<TurnStampV1>,
    pub draws: [u32; 2],
    pub tutor: [TutorMarkerV1; 2],
}

/// Actual turns begun so far, from the engine's own counters, when the
/// record starts mid-game (fixtures, imported states).
fn turns_begun_now_v1(state: &GameState) -> [u32; 2] {
    if let Some(turn) = &state.extra_turn_v1 {
        return turn.own_turns_started;
    }
    let round = state.turn;
    if round == 0 {
        return [0, 0];
    }
    let starter = state.starting_player;
    // A Time Vault skip before the record starts is counted here; the live
    // record never counts one.
    let mut begun = [0; 2];
    begun[starter.index()] = round;
    begun[starter.opponent().index()] = if state.active_player == starter {
        round - 1
    } else {
        round
    };
    begun
}

/// Starts the record. Every incarnation already on the battlefield or
/// plotted gets stamp 0 (before the record).
pub fn enable_v1(state: &mut GameState) {
    let turns_begun = turns_begun_now_v1(state);
    let turn_sequence = turns_begun[0] + turns_begun[1];
    let active = state.active_player;
    let previous_active = match &state.extra_turn_v1 {
        Some(turn) if turn_sequence > 1 => Some(turn.previous_player),
        _ if turn_sequence > 1 => Some(active.opponent()),
        _ => None,
    };
    let mut history = FlatV9HistoryV1 {
        turn_sequence,
        turns_begun,
        current_active: (state.turn > 0).then_some(active),
        previous_active,
        ..FlatV9HistoryV1::default()
    };
    stamp_v1(state, &mut history, 0);
    // An incarnation whose legacy stamp is the current turn key entered (or
    // was plotted) this turn as far as the record can tell: leave it
    // unstamped.
    let key = Some(state.turn_key_v1());
    history.entries.retain(|stamp| {
        state
            .objects
            .get(crate::ids::ObjectId(stamp.object))
            .v4
            .entered_battlefield_turn
            != key
    });
    history.plots.retain(|stamp| {
        state
            .objects
            .get(crate::ids::ObjectId(stamp.object))
            .plotted_turn
            != key
    });
    state.flat_v9_history_v1 = Some(Box::new(history));
}

fn battlefield_incarnations_v1(state: &GameState) -> Vec<(u32, u32)> {
    state
        .players
        .iter()
        .flat_map(|player| player.battlefield.iter())
        .map(|&id| (id.0, state.objects.get(id).zone_change_count))
        .collect()
}

fn plotted_incarnations_v1(state: &GameState) -> Vec<(u32, u32)> {
    state
        .exile
        .iter()
        .filter(|&&id| {
            let object = state.objects.get(id);
            object.zone == Zone::Exile && object.plotted_turn.is_some()
        })
        .map(|&id| (id.0, state.objects.get(id).zone_change_count))
        .collect()
}

/// Drops stamps of incarnations that left and stamps every unstamped one
/// with `turn_sequence`.
fn stamp_v1(state: &GameState, history: &mut FlatV9HistoryV1, turn_sequence: u32) {
    for (stamps, live) in [
        (&mut history.entries, battlefield_incarnations_v1(state)),
        (&mut history.plots, plotted_incarnations_v1(state)),
    ] {
        stamps.retain(|stamp| live.contains(&(stamp.object, stamp.zone_change_count)));
        for (object, zone_change_count) in live {
            if !stamps
                .iter()
                .any(|stamp| (stamp.object, stamp.zone_change_count) == (object, zone_change_count))
            {
                stamps.push(TurnStampV1 {
                    object,
                    zone_change_count,
                    turn_sequence,
                });
            }
        }
        stamps.sort_unstable_by_key(|stamp| (stamp.object, stamp.zone_change_count));
    }
}

/// A turn of `state.active_player` has begun (engine step order wrap).
pub(crate) fn on_turn_begin_v1(state: &mut GameState) {
    let Some(mut history) = state.flat_v9_history_v1.take() else {
        return;
    };
    let ended = history.turn_sequence;
    stamp_v1(state, &mut history, ended);
    history.turn_sequence = ended.saturating_add(1);
    let active = state.active_player;
    history.turns_begun[active.index()] = history.turns_begun[active.index()].saturating_add(1);
    history.previous_active = history.current_active;
    history.current_active = Some(active);
    state.flat_v9_history_v1 = Some(history);
}

/// `player` drew the top card of their library (`GameState::draw_card`,
/// whose own removal note the record does not see).
pub(crate) fn on_draw_v1(state: &mut GameState, player: PlayerId) {
    if let Some(history) = &mut state.flat_v9_history_v1 {
        history.draws[player.index()] = history.draws[player.index()].saturating_add(1);
        let marker = &mut history.tutor[player.index()];
        if marker.status == TutorStatusV1::Tracked {
            if marker.depth == 0 {
                marker.status = TutorStatusV1::DrawnOrRemoved;
            } else {
                marker.depth -= 1;
            }
        }
    }
}

/// The cards of `owner`'s library at `range` were randomized without a
/// note (`GameState::randomize_library_top_v1` and `_bottom_v1`).
pub(crate) fn on_partial_randomization_v1(
    state: &mut GameState,
    owner: PlayerId,
    range: std::ops::Range<usize>,
) {
    if let Some(history) = &mut state.flat_v9_history_v1 {
        let marker = &mut history.tutor[owner.index()];
        if marker.status == TutorStatusV1::Tracked && range.contains(&(marker.depth as usize)) {
            invalidate_v1(marker);
        }
    }
}

/// An aborted cast returned a plotted card to exile as a new incarnation
/// (`materialize_aborted_cast_v1`): its plot stamp carries over from the
/// incarnation it was cast from.
pub(crate) fn on_plot_restored_v1(
    state: &mut GameState,
    object: crate::ids::ObjectId,
    exile_zone_change_count: u32,
    zone_change_count: u32,
) {
    if let Some(history) = &mut state.flat_v9_history_v1 {
        let carried = history
            .plots
            .iter()
            .find(|stamp| {
                (stamp.object, stamp.zone_change_count) == (object.0, exile_zone_change_count)
            })
            .copied();
        if let Some(stamp) = carried {
            history.plots.push(TurnStampV1 {
                zone_change_count,
                ..stamp
            });
            history
                .plots
                .sort_unstable_by_key(|stamp| (stamp.object, stamp.zone_change_count));
        }
    }
}

fn invalidate_v1(marker: &mut TutorMarkerV1) {
    marker.status = TutorStatusV1::Invalidated;
    marker.depth = 0;
}

/// A resolving tutor publicly put a card on top of `owner`'s library.
pub(crate) fn on_tutor_placement_v1(state: &mut GameState, owner: PlayerId) {
    if let Some(history) = &mut state.flat_v9_history_v1 {
        history.tutor[owner.index()] = TutorMarkerV1 {
            status: TutorStatusV1::Tracked,
            depth: 0,
            draws_at_placement: history.draws[owner.index()],
        };
    }
}

/// A public library note (`GameState::note_observation`), before it is
/// journalled. `library_len` is the owner's library size after the event.
pub(crate) fn on_observation_note_v1(
    history: &mut FlatV9HistoryV1,
    kind: &ObservationNoteKindV1,
    library_len: impl Fn(PlayerId) -> usize,
) {
    use ObservationNoteKindV1 as K;
    let owner = match kind {
        K::LibraryRandomized { owner }
        | K::LibraryReordered { owner, .. }
        | K::LibraryScried { owner, .. }
        | K::LibraryRemoval { owner, .. }
        | K::LibraryInsertion { owner, .. } => *owner,
        K::LibraryLooked { .. } | K::HandCardRevealed { .. } => return,
    };
    let marker = &mut history.tutor[owner.index()];
    if marker.status != TutorStatusV1::Tracked {
        return;
    }
    let depth = marker.depth;
    match kind {
        K::LibraryRandomized { .. } => invalidate_v1(marker),
        K::LibraryReordered { ordered, .. } => {
            if ordered.len() > 1 && (depth as usize) < ordered.len() {
                invalidate_v1(marker);
            }
        }
        K::LibraryScried {
            retained_top,
            ordered_bottom,
            ..
        } => {
            let prefix = retained_top.len() + ordered_bottom.len();
            if (depth as usize) >= prefix {
                marker.depth = depth - ordered_bottom.len() as u32;
            } else if prefix == 1 {
                // A one-card scry's keep-or-bottom result is public.
                if ordered_bottom.len() == 1 {
                    marker.depth = library_len(owner).saturating_sub(1) as u32;
                }
            } else {
                invalidate_v1(marker);
            }
        }
        // Draws reach `on_draw_v1` instead. Any other removal's position may
        // be a private choice (a search, a pick among looked-at cards), so it
        // never moves the marker by position.
        K::LibraryRemoval { .. } => invalidate_v1(marker),
        K::LibraryInsertion { position, .. } => {
            if *position <= depth {
                marker.depth = depth.saturating_add(1);
            }
        }
        K::LibraryLooked { .. } | K::HandCardRevealed { .. } => {}
    }
}

/// The public facts FlatV9 reads from the record.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlatV9HistoryFactsV1 {
    pub(crate) turn_sequence: u32,
    /// By absolute seat.
    pub(crate) turns_begun: [u32; 2],
    pub(crate) consecutive_turn: bool,
    /// By absolute seat: declared attackers this turn.
    pub(crate) attacked: [bool; 2],
    pub(crate) creature_died: bool,
    /// By absolute seat.
    pub(crate) lost_life: [bool; 2],
    /// By absolute library owner.
    pub(crate) tutor: [FlatV9TutorFactsV1; 2],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlatV9TutorFactsV1 {
    pub(crate) status: TutorStatusV1,
    pub(crate) depth: u32,
    pub(crate) draws_since: u32,
}

impl FlatV9HistoryV1 {
    /// Turns since `object`'s incarnation entered the battlefield: 0 this
    /// turn.
    pub(crate) fn entered_turns_ago_v1(&self, object: u32, zone_change_count: u32) -> u32 {
        Self::age_v1(&self.entries, self.turn_sequence, object, zone_change_count)
    }

    /// Turns since `object`'s incarnation was plotted: 0 this turn.
    pub(crate) fn plotted_turns_ago_v1(&self, object: u32, zone_change_count: u32) -> u32 {
        Self::age_v1(&self.plots, self.turn_sequence, object, zone_change_count)
    }

    fn age_v1(stamps: &[TurnStampV1], now: u32, object: u32, zone_change_count: u32) -> u32 {
        stamps
            .iter()
            .find(|stamp| (stamp.object, stamp.zone_change_count) == (object, zone_change_count))
            .map_or(0, |stamp| now.saturating_sub(stamp.turn_sequence))
    }

    pub(crate) fn facts_v1(&self, state: &GameState) -> FlatV9HistoryFactsV1 {
        let seats = [PlayerId::P0, PlayerId::P1];
        FlatV9HistoryFactsV1 {
            turn_sequence: self.turn_sequence,
            turns_begun: self.turns_begun,
            consecutive_turn: self.previous_active.is_some()
                && self.previous_active == self.current_active,
            // Only the active player attacks; the declaration count is kept
            // in every rules profile.
            attacked: seats.map(|seat| {
                seat == state.active_player && state.creatures_attacked_this_turn_v1() > 0
            }),
            creature_died: state.creature_died_this_turn_v1(),
            lost_life: seats.map(|seat| state.player_lost_life_this_turn_v1(seat)),
            tutor: seats.map(|seat| {
                let marker = self.tutor[seat.index()];
                FlatV9TutorFactsV1 {
                    status: marker.status,
                    depth: marker.depth,
                    draws_since: if marker.status == TutorStatusV1::None {
                        0
                    } else {
                        self.draws[seat.index()].saturating_sub(marker.draws_at_placement)
                    },
                }
            }),
        }
    }
}

#[cfg(test)]
#[path = "flat_v9_history_v1_tests.rs"]
mod tests;
