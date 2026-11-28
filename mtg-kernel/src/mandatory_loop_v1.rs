//! Rules 104.4b and 731.4: a game that enters a loop of mandatory actions no
//! player can break is a draw (issue #957).
//!
//! **Where a mandatory loop runs.** The policy surface (`HarnessSurfaceV2`)
//! answers some engine decisions itself: a priority window whose player has
//! no real option or whose step the surface never offers, the same caster's
//! window after their own cast, the combat one-action throttle and an empty
//! attack declaration. These forced transitions carry no player choice.
//! When two permanents' mandatory triggers keep re-triggering each other and
//! no player can act, every window is forced and one `next_decision` call
//! never returns. [`ForcedLoopDetectorV1`] watches those forced transitions
//! within one call and ends the game as a draw when a position repeats.
//!
//! **What "the same position" means.** The whole game state and the
//! surface's own pass bookkeeping (`public_context`) must repeat, except for
//! three records that change on every repetition without changing the game:
//!
//! - the priority-round counter (the surface's round marks are compared
//!   against it, so they are kept as offsets from it);
//! - stack-item incarnation ids, which are compared by age relative to the
//!   allocator (`StackItemIdAgeHashV1`), with the allocator itself left out;
//! - the event history: the events committed during the watched calls are
//!   kept as the set of distinct events (by age-relative hash). A loop adds
//!   the same events again; a new kind of event, or any event that differs,
//!   is a change. Events committed before watching began are common to every
//!   compared position. This assumes no rule acts on how many times an
//!   identical event has already repeated: per-turn trigger uses are state
//!   and are compared, and counting text such as "the Nth card you draw"
//!   moves a library, so its positions differ.
//!
//! Everything else is compared as it is: life totals, counters, objects and
//! their zone-change counts, tokens, the stack's contents, pending triggers,
//! turn-scoped trigger uses, randomness. A sequence that changes any of them
//! between repetitions, such as a mandatory life drain that ends the game, is
//! never adjudicated. Positions are compared by two independent 64-bit
//! hashes of that canonical form.
//!
//! **What is not adjudicated.** A loop that passes through any decision the
//! surface hands to a policy or a person is optional: every repetition
//! offers a real choice (Zirda, the Dawnwaker with Basalt Monolith; any
//! window an engine-window or action-window seat is shown). The detector
//! lives only inside one `next_decision` call, so every surfaced decision
//! starts it afresh. The census watchdog, the session caps
//! (`max_physical_decisions`, `max_policy_steps`, the optional
//! `max_stack_objects`) and any deterministic step bound are unchanged: a
//! loop whose stack grows still meets the stack cap, since its positions
//! differ.
//!
//! **Cost.** Nothing is hashed for the first
//! [`UNCHECKED_FORCED_TRANSITIONS_V1`] forced transitions of a call, which
//! ordinary games never reach, so their play, bytes and timing are unchanged.
//! After that, Brent's cycle search hashes one position per transition and
//! finds a loop of period `p` within a few multiples of `p` more transitions.
//!
//! **The draw.** The surface records [`MandatoryLoopDrawV1`] on the state
//! (`GameState::mandatory_loop_draw_v1`) and returns
//! `Decision::GameOver { winner: None }`. The engine reports that terminal
//! for the state from then on, so every session reports a natural draw.

use crate::event::CommittedEvent;
use crate::event_history_v1::EventHistoryV1;
use crate::ids::StackItemIdAgeHashV1;
use crate::state::GameState;
use crate::surface_v2::HarnessSurfacePublicContextV2;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// Forced transitions within one surface call before positions are hashed.
/// A performance gate only: no game ends here, and a loop is still found,
/// just after this many transitions.
pub const UNCHECKED_FORCED_TRANSITIONS_V1: u32 = 1024;

/// The terminal record of a mandatory-loop draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MandatoryLoopDrawV1 {
    /// Forced transitions in one repetition of the loop.
    pub period: u32,
}

/// Two independent 64-bit lanes: FNV-1a and SipHash-1-3 with fixed keys.
/// In-process only; never written to an artifact.
struct TwoLaneHasherV1 {
    fnv: u64,
    sip: std::hash::DefaultHasher,
}

impl TwoLaneHasherV1 {
    fn new() -> Self {
        Self {
            fnv: 0xcbf2_9ce4_8422_2325,
            sip: std::hash::DefaultHasher::new(),
        }
    }

    fn finish_pair(&self) -> (u64, u64) {
        (self.fnv, self.sip.finish())
    }
}

impl Hasher for TwoLaneHasherV1 {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.fnv ^= u64::from(byte);
            self.fnv = self.fnv.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.sip.write(bytes);
    }

    fn finish(&self) -> u64 {
        self.fnv
    }
}

/// The distinct events committed since watching began.
struct EventTrackV1 {
    /// The history as last scanned: later histories must extend it.
    scanned: EventHistoryV1,
    distinct: HashSet<(u64, u64)>,
}

impl EventTrackV1 {
    fn new(history: &EventHistoryV1) -> Self {
        Self {
            scanned: history.clone(),
            distinct: HashSet::new(),
        }
    }

    /// Adds the events committed since the last scan. False when the history
    /// no longer extends the scanned one (it was cut or rewritten).
    fn scan(&mut self, history: &EventHistoryV1, stack_item_base: u64) -> bool {
        let scanned = self.scanned.len();
        if history.len() < scanned || history.common_prefix_len_v2(&self.scanned) != scanned {
            return false;
        }
        if history.len() > scanned {
            let _ages = StackItemIdAgeHashV1::begin(stack_item_base);
            for event in &history.range_v2(scanned..) {
                self.distinct.insert(event_hash(event));
            }
        }
        self.scanned = history.clone();
        true
    }
}

fn event_hash(event: &CommittedEvent) -> (u64, u64) {
    let mut hasher = TwoLaneHasherV1::new();
    event.hash(&mut hasher);
    hasher.finish_pair()
}

/// Watches the forced transitions of one surface call (module doc).
#[derive(Default)]
pub(crate) struct ForcedLoopDetectorV1 {
    observed: u32,
    events: Option<EventTrackV1>,
    saved: Option<(u64, u64)>,
    power: u32,
    since_saved: u32,
}

impl ForcedLoopDetectorV1 {
    /// Called once per forced transition, on the position it reached.
    /// Returns the draw when this position repeats an earlier one.
    pub(crate) fn observe(
        &mut self,
        state: &mut GameState,
        surface: impl FnOnce() -> HarnessSurfacePublicContextV2,
    ) -> Option<MandatoryLoopDrawV1> {
        self.observed = self.observed.saturating_add(1);
        if self.observed <= UNCHECKED_FORCED_TRANSITIONS_V1 {
            return None;
        }
        let surface = surface();
        let stack_item_base = state.engine.next_stack_item_id;
        let tracked = match self.events.as_mut() {
            Some(track) => track.scan(&state.engine.event_history, stack_item_base),
            None => false,
        };
        if !tracked {
            // First watched position, or the history was rewritten: start the
            // search again from here.
            self.events = Some(EventTrackV1::new(&state.engine.event_history));
            self.saved = None;
        }
        let distinct_events = self.events.as_ref().map_or(0, |track| track.distinct.len());
        let position = position_hash(state, surface, distinct_events);
        match self.saved {
            None => {
                self.saved = Some(position);
                self.power = 1;
                self.since_saved = 0;
                None
            }
            Some(saved) => {
                self.since_saved += 1;
                if saved == position {
                    return Some(MandatoryLoopDrawV1 {
                        period: self.since_saved,
                    });
                }
                if self.since_saved == self.power {
                    self.saved = Some(position);
                    self.power = self.power.saturating_mul(2);
                    self.since_saved = 0;
                }
                None
            }
        }
    }
}

/// The canonical position (module doc), hashed in two lanes. The state is
/// restored exactly before returning.
fn position_hash(
    state: &mut GameState,
    mut surface: HarnessSurfacePublicContextV2,
    distinct_events: usize,
) -> (u64, u64) {
    let round = state.engine.priority_round;
    surface.combat_priority_round_seen = surface
        .combat_priority_round_seen
        .map(|seen| round.wrapping_sub(seen));
    surface.stack_len_round_seen = surface
        .stack_len_round_seen
        .map(|seen| round.wrapping_sub(seen));
    let stack_item_base = state.engine.next_stack_item_id;
    let history = std::mem::take(&mut state.engine.event_history);
    state.engine.priority_round = 0;
    state.engine.next_stack_item_id = 0;
    let mut hasher = TwoLaneHasherV1::new();
    {
        let _ages = StackItemIdAgeHashV1::begin(stack_item_base);
        state.hash(&mut hasher);
        surface.hash(&mut hasher);
    }
    distinct_events.hash(&mut hasher);
    state.engine.event_history = history;
    state.engine.priority_round = round;
    state.engine.next_stack_item_id = stack_item_base;
    hasher.finish_pair()
}

#[cfg(test)]
mod tests;
