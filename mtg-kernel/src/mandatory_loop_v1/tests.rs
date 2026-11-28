//! Unit checks of the mandatory-loop record and detector. The hand-built
//! two-permanent loop and its optional and finite controls play through the
//! surface and sessions in `rl_session/mandatory_loop_v1_tests.rs`.

use super::*;
use crate::card_def::card_id_by_name;
use crate::engine::{Action, Decision};
use crate::ids::{PlayerId, StackItemId};
use crate::surface_v2::HarnessSurfaceV2;
use std::hash::DefaultHasher;

fn fresh_state() -> GameState {
    let forest = card_id_by_name("Forest").expect("Forest is in every catalog");
    GameState::new_from_libraries(&[forest; 20], &[forest; 20], |_| "Forest".into(), 957)
}

fn context() -> HarnessSurfacePublicContextV2 {
    HarnessSurfaceV2::new().public_context()
}

fn plain_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Feeds the detector the unwatched transitions of a call.
fn skip_gate(detector: &mut ForcedLoopDetectorV1, state: &mut GameState) {
    for _ in 0..UNCHECKED_FORCED_TRANSITIONS_V1 {
        assert_eq!(detector.observe(state, context), None);
    }
}

#[test]
fn an_absent_draw_leaves_state_bytes_and_hash_unchanged() {
    let state = fresh_state();
    let json = serde_json::to_string(&state).unwrap();
    assert!(!json.contains("mandatory_loop_draw_v1"));

    let mut drawn = state.clone();
    drawn.mandatory_loop_draw_v1 = Some(MandatoryLoopDrawV1 { period: 4 });
    assert_ne!(drawn.state_hash(), state.state_hash());
    let drawn_json = serde_json::to_string(&drawn).unwrap();
    assert!(drawn_json.contains(r#""mandatory_loop_draw_v1":{"period":4}"#));
    let restored: GameState = serde_json::from_str(&drawn_json).unwrap();
    assert_eq!(
        restored.mandatory_loop_draw_v1,
        drawn.mandatory_loop_draw_v1
    );
    assert_eq!(restored.state_hash(), drawn.state_hash());
    assert_eq!(serde_json::to_string(&restored).unwrap(), drawn_json);

    drawn.mandatory_loop_draw_v1 = None;
    assert_eq!(drawn.state_hash(), state.state_hash());
    assert_eq!(drawn.diagnostic_state_hash(), state.diagnostic_state_hash());
    assert_eq!(serde_json::to_string(&drawn).unwrap(), json);
}

#[test]
fn a_drawn_state_reports_the_draw_and_refuses_every_action() {
    let mut state = fresh_state();
    state.mandatory_loop_draw_v1 = Some(MandatoryLoopDrawV1 { period: 2 });
    for _ in 0..2 {
        assert_eq!(
            crate::engine::advance_until_decision(&mut state),
            Decision::GameOver { winner: None }
        );
    }
    let before = state.state_hash();
    assert!(crate::engine::step(&mut state, Action::Pass).is_err());
    assert_eq!(state.state_hash(), before);
    let mut surface = HarnessSurfaceV2::new();
    assert_eq!(
        surface.next_decision(&mut state),
        crate::surface_v2::SurfaceDecision::Decision(Decision::GameOver { winner: None })
    );
}

#[test]
fn stack_item_ids_hash_by_age_only_under_the_guard() {
    assert_eq!(plain_hash(&StackItemId(7)), plain_hash(&7u64));
    {
        let _ages = StackItemIdAgeHashV1::begin(10);
        assert_eq!(
            plain_hash(&StackItemId(7)),
            plain_hash(&7u64.wrapping_sub(10))
        );
        {
            let _inner = StackItemIdAgeHashV1::begin(20);
            assert_eq!(
                plain_hash(&StackItemId(17)),
                plain_hash(&7u64.wrapping_sub(10))
            );
        }
        assert_eq!(plain_hash(&StackItemId(17)), plain_hash(&7u64));
    }
    assert_eq!(plain_hash(&StackItemId(7)), plain_hash(&7u64));
}

#[test]
fn nothing_is_compared_before_the_gate_and_a_repeat_after_it_is_a_draw() {
    let mut state = fresh_state();
    let mut detector = ForcedLoopDetectorV1::default();
    skip_gate(&mut detector, &mut state);
    assert_eq!(detector.observe(&mut state, context), None);
    assert_eq!(
        detector.observe(&mut state, context),
        Some(MandatoryLoopDrawV1 { period: 1 })
    );
}

#[test]
fn watching_restores_the_state_exactly() {
    let mut state = fresh_state();
    state.engine.priority_round = 41;
    state.engine.next_stack_item_id = 9;
    state.engine.event_history.push(CommittedEvent::LifeGain {
        player: PlayerId::P0,
        amount: 1,
    });
    let before = state.state_hash();
    let json = serde_json::to_string(&state).unwrap();
    let mut detector = ForcedLoopDetectorV1::default();
    skip_gate(&mut detector, &mut state);
    for _ in 0..3 {
        let _ = detector.observe(&mut state, context);
        assert_eq!(state.state_hash(), before);
        assert_eq!(serde_json::to_string(&state).unwrap(), json);
    }
}

#[test]
fn a_cycle_differing_only_in_bookkeeping_is_found_with_its_period() {
    let mut state = fresh_state();
    let mut detector = ForcedLoopDetectorV1::default();
    skip_gate(&mut detector, &mut state);
    // Period 3 in life totals; the priority round, the stack-item allocator
    // and the event history grow on every transition, repeating the same
    // three events.
    let lives = [20, 21, 19];
    let mut found = None;
    for transition in 0..64u32 {
        let phase = (transition % 3) as usize;
        state.players[0].life = lives[phase];
        state.engine.priority_round += 1;
        state.engine.next_stack_item_id += 1;
        state.engine.event_history.push(CommittedEvent::LifeGain {
            player: PlayerId::P0,
            amount: lives[phase],
        });
        if let Some(draw) = detector.observe(&mut state, context) {
            found = Some(draw);
            break;
        }
    }
    assert_eq!(found, Some(MandatoryLoopDrawV1 { period: 3 }));
}

#[test]
fn a_changing_position_is_never_a_draw() {
    let mut state = fresh_state();
    let mut detector = ForcedLoopDetectorV1::default();
    skip_gate(&mut detector, &mut state);
    // A finite mandatory sequence: a life total keeps moving.
    for transition in 0..10_000 {
        state.players[1].life = 10_000 - transition;
        assert_eq!(detector.observe(&mut state, context), None);
    }
}

#[test]
fn a_new_distinct_event_is_a_change() {
    let mut state = fresh_state();
    let mut detector = ForcedLoopDetectorV1::default();
    skip_gate(&mut detector, &mut state);
    // The visible state repeats, but each transition commits an event that
    // never happened before in this call: the position is not the same.
    for amount in 1..2_000 {
        state.engine.event_history.push(CommittedEvent::LifeGain {
            player: PlayerId::P1,
            amount,
        });
        assert_eq!(detector.observe(&mut state, context), None);
    }
}

#[test]
fn a_rewritten_history_restarts_the_search() {
    let mut state = fresh_state();
    for amount in 0..4 {
        state.engine.event_history.push(CommittedEvent::LifeGain {
            player: PlayerId::P0,
            amount,
        });
    }
    let mut detector = ForcedLoopDetectorV1::default();
    skip_gate(&mut detector, &mut state);
    assert_eq!(detector.observe(&mut state, context), None);
    state.engine.event_history.truncate(2);
    // The cut history is not compared with the saved position.
    assert_eq!(detector.observe(&mut state, context), None);
    assert_eq!(
        detector.observe(&mut state, context),
        Some(MandatoryLoopDrawV1 { period: 1 })
    );
}
