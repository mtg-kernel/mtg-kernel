//! FlatV9 history (`flat_v9_history_v1.rs`): issue 740's turn identity for
//! both starters, consecutive turns, zone re-entry and snapshot restore, and
//! the tutor marker's public state machine, including the hidden-world
//! equality the ruling requires.

use super::*;
use crate::engine::{self, Action};
use crate::event::{self, ProposedEvent};
use crate::ids::ObjectId;
use crate::policy_observation_v6::tests::{advance_to_next_main_for, put, ready_state_for};
use crate::state::ObjectLinkV4;

const P0: PlayerId = PlayerId::P0;
const P1: PlayerId = PlayerId::P1;

fn game_v1(starter: PlayerId) -> GameState {
    let mut state = ready_state_for(starter);
    for owner in [P0, P1] {
        for _ in 0..8 {
            put(&mut state, owner, "Forest", Zone::Library);
        }
    }
    enable_v1(&mut state);
    state
}

fn history_v1(state: &GameState) -> &FlatV9HistoryV1 {
    state.flat_v9_history_v1.as_deref().expect("FlatV9 history")
}

fn entered_v1(state: &GameState, id: ObjectId) -> u32 {
    history_v1(state).entered_turns_ago_v1(id.0, state.objects.get(id).zone_change_count)
}

fn play_land_v1(state: &mut GameState, player: PlayerId) -> ObjectId {
    let land = put(state, player, "Mountain", Zone::Hand);
    engine::step(state, Action::PlayLand(land)).unwrap();
    assert_eq!(state.objects.get(land).zone, Zone::Battlefield);
    land
}

#[test]
fn issue_740_entry_is_earlier_turn_on_the_other_players_turn_for_both_starters() {
    for starter in [P0, P1] {
        let other = starter.opponent();
        let mut state = game_v1(starter);
        assert_eq!(history_v1(&state).turn_sequence, 1);
        let land = play_land_v1(&mut state, starter);
        assert_eq!(entered_v1(&state, land), 0, "this turn");
        let round = state.turn;
        advance_to_next_main_for(&mut state, other);
        // The same round, the other player's turn: the legacy round stamp
        // reads this turn; the exact stamp reads the previous turn.
        assert_eq!(state.turn, round);
        assert_eq!(entered_v1(&state, land), 1, "{starter:?}: earlier turn");
        let facts = history_v1(&state).facts_v1(&state);
        assert_eq!(facts.turn_sequence, 2);
        assert_eq!(facts.turns_begun, [1, 1]);
        assert!(!facts.consecutive_turn);
        let theirs = play_land_v1(&mut state, other);
        assert_eq!(entered_v1(&state, theirs), 0);
        advance_to_next_main_for(&mut state, starter);
        assert_eq!(state.turn, round + 1);
        assert_eq!(entered_v1(&state, land), 2);
        assert_eq!(entered_v1(&state, theirs), 1);
        assert_eq!(history_v1(&state).turn_sequence, 3);
    }
}

#[test]
fn zone_reentry_is_a_new_incarnation_entering_this_turn() {
    for starter in [P0, P1] {
        let mut state = game_v1(starter);
        let land = play_land_v1(&mut state, starter);
        advance_to_next_main_for(&mut state, starter.opponent());
        assert_eq!(entered_v1(&state, land), 1);
        event::propose_and_commit(&mut state, ProposedEvent::zone_change(land, Zone::Hand));
        event::propose_and_commit(
            &mut state,
            ProposedEvent::zone_change(land, Zone::Battlefield),
        );
        assert_eq!(state.objects.get(land).zone, Zone::Battlefield);
        assert_eq!(entered_v1(&state, land), 0, "re-entered this turn");
        advance_to_next_main_for(&mut state, starter);
        assert_eq!(entered_v1(&state, land), 1);
        // Only live incarnations keep stamps.
        let history = history_v1(&state);
        assert!(history.entries.iter().all(|stamp| stamp.object != land.0
            || stamp.zone_change_count == state.objects.get(land).zone_change_count));
    }
}

#[test]
fn snapshot_restore_keeps_the_history_and_its_continuation() {
    let mut state = game_v1(P1);
    let land = play_land_v1(&mut state, P1);
    advance_to_next_main_for(&mut state, P0);
    let snapshot = serde_json::to_string(&state).unwrap();
    let mut restored: GameState = serde_json::from_str(&snapshot).unwrap();
    assert_eq!(restored.flat_v9_history_v1, state.flat_v9_history_v1);
    for game in [&mut state, &mut restored] {
        advance_to_next_main_for(game, P1);
    }
    assert_eq!(restored.flat_v9_history_v1, state.flat_v9_history_v1);
    assert_eq!(entered_v1(&restored, land), 2);
    let hash = |game: &GameState| {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        game.hash(&mut hasher);
        hasher.finish()
    };
    assert_eq!(hash(&restored), hash(&state));
}

#[test]
fn a_game_without_the_record_keeps_its_bytes_and_hash() {
    let plain = ready_state_for(P0);
    let value = serde_json::to_value(&plain).unwrap();
    assert!(value.get("flat_v9_history_v1").is_none());
    let mut enabled = plain.clone();
    enable_v1(&mut enabled);
    let hash = |game: &GameState| {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        game.hash(&mut hasher);
        hasher.finish()
    };
    assert_ne!(hash(&plain), hash(&enabled));
    let mut disabled = enabled.clone();
    disabled.flat_v9_history_v1 = None;
    assert_eq!(hash(&plain), hash(&disabled));
}

fn tracked_v1(depth: u32) -> FlatV9HistoryV1 {
    let mut history = FlatV9HistoryV1::default();
    history.tutor[0] = TutorMarkerV1 {
        status: TutorStatusV1::Tracked,
        depth,
        draws_at_placement: 0,
    };
    history
}

fn note_v1(history: &mut FlatV9HistoryV1, kind: ObservationNoteKindV1) -> TutorMarkerV1 {
    on_observation_note_v1(history, &kind, |_| 20);
    history.tutor[0]
}

#[test]
fn tutor_marker_follows_public_library_notes() {
    use ObservationNoteKindV1 as K;
    let mut history = tracked_v1(0);
    // A placement on top buries it; one below it does not.
    assert_eq!(
        note_v1(
            &mut history,
            K::LibraryInsertion {
                owner: P0,
                position: 0
            }
        )
        .depth,
        1
    );
    assert_eq!(
        note_v1(
            &mut history,
            K::LibraryInsertion {
                owner: P0,
                position: 5
            }
        )
        .depth,
        1,
        "below it"
    );
    // The other library never moves it.
    assert_eq!(
        note_v1(&mut history, K::LibraryRandomized { owner: P1 }).status,
        TutorStatusV1::Tracked
    );
    // A removal that is not a draw may be at a secret position: it
    // invalidates the marker wherever it is, depth cleared.
    let removed = note_v1(
        &mut history,
        K::LibraryRemoval {
            owner: P0,
            position: 3,
        },
    );
    assert_eq!(
        (removed.status, removed.depth),
        (TutorStatusV1::Invalidated, 0)
    );
    // Once invalidated, later notes change nothing.
    assert_eq!(
        note_v1(
            &mut history,
            K::LibraryInsertion {
                owner: P0,
                position: 0
            }
        ),
        removed
    );

    let mut shuffled = tracked_v1(2);
    let after = note_v1(&mut shuffled, K::LibraryRandomized { owner: P0 });
    assert_eq!((after.status, after.depth), (TutorStatusV1::Invalidated, 0));

    let o = |n: u32| ObjectId(n);
    let mut deep = tracked_v1(3);
    let below = note_v1(
        &mut deep,
        K::LibraryScried {
            owner: P0,
            retained_top: vec![o(1)],
            ordered_bottom: vec![o(2)],
        },
    );
    assert_eq!((below.status, below.depth), (TutorStatusV1::Tracked, 2));
    let mut one = tracked_v1(0);
    let bottomed = note_v1(
        &mut one,
        K::LibraryScried {
            owner: P0,
            retained_top: vec![],
            ordered_bottom: vec![o(1)],
        },
    );
    assert_eq!(
        (bottomed.status, bottomed.depth),
        (TutorStatusV1::Tracked, 19)
    );
    let mut kept = tracked_v1(0);
    assert_eq!(
        note_v1(
            &mut kept,
            K::LibraryScried {
                owner: P0,
                retained_top: vec![o(1)],
                ordered_bottom: vec![],
            },
        )
        .depth,
        0
    );
    let mut covered = tracked_v1(1);
    let scried = note_v1(
        &mut covered,
        K::LibraryScried {
            owner: P0,
            retained_top: vec![o(1), o(2)],
            ordered_bottom: vec![],
        },
    );
    assert_eq!(
        (scried.status, scried.depth),
        (TutorStatusV1::Invalidated, 0)
    );
    let mut reordered = tracked_v1(1);
    assert_eq!(
        note_v1(
            &mut reordered,
            K::LibraryReordered {
                owner: P0,
                ordered: vec![o(1)],
                revealed_to: vec![P0, P1],
            },
        )
        .status,
        TutorStatusV1::Tracked,
        "a one-card reorder moves nothing"
    );
    let mut single = tracked_v1(0);
    assert_eq!(
        note_v1(
            &mut single,
            K::LibraryReordered {
                owner: P0,
                ordered: vec![o(1)],
                revealed_to: vec![P0],
            },
        )
        .status,
        TutorStatusV1::Tracked
    );
    let reshuffled = note_v1(
        &mut reordered,
        K::LibraryReordered {
            owner: P0,
            ordered: vec![o(1), o(2), o(3)],
            revealed_to: vec![P0],
        },
    );
    assert_eq!(
        (reshuffled.status, reshuffled.depth),
        (TutorStatusV1::Invalidated, 0)
    );
    // Looks and hand reveals move nothing.
    let mut looked = tracked_v1(0);
    assert_eq!(
        note_v1(
            &mut looked,
            K::LibraryLooked {
                observer: P1,
                owner: P0,
                positions: vec![(0, o(1))],
            },
        ),
        tracked_v1(0).tutor[0]
    );
}

/// A library of `count` Forests for `owner`, the record on, and a tutor
/// placement on top of it.
fn tutored_game_v1(owner: PlayerId, count: usize) -> GameState {
    let mut state = ready_state_for(owner);
    for seat in [P0, P1] {
        for _ in 0..count {
            put(&mut state, seat, "Forest", Zone::Library);
        }
    }
    enable_v1(&mut state);
    on_tutor_placement_v1(&mut state, owner);
    state
}

#[test]
fn tutor_marker_counts_draws_and_sees_the_draw_of_the_card() {
    let mut state = tutored_game_v1(P0, 5);
    let facts = history_v1(&state).facts_v1(&state).tutor[0];
    assert_eq!(
        (facts.status, facts.depth, facts.draws_since),
        (TutorStatusV1::Tracked, 0, 0)
    );
    assert_eq!(
        history_v1(&state).facts_v1(&state).tutor[1].status,
        TutorStatusV1::None
    );
    // A draw while it is buried moves it up; the next draws it.
    on_tutor_placement_v1(&mut state, P0);
    state.flat_v9_history_v1.as_deref_mut().unwrap().tutor[0].depth = 1;
    state.draw_card(P0).unwrap();
    let facts = history_v1(&state).facts_v1(&state).tutor[0];
    assert_eq!((facts.status, facts.depth), (TutorStatusV1::Tracked, 0));
    state.draw_card(P0).unwrap();
    let facts = history_v1(&state).facts_v1(&state).tutor[0];
    assert_eq!(
        (facts.status, facts.draws_since),
        (TutorStatusV1::DrawnOrRemoved, 2)
    );
    state.draw_card(P0).unwrap();
    assert_eq!(history_v1(&state).facts_v1(&state).tutor[0].draws_since, 3);
    // A real shuffle of a newly marked library invalidates it.
    on_tutor_placement_v1(&mut state, P0);
    state.shuffle_library(P0).unwrap();
    assert_eq!(
        history_v1(&state).facts_v1(&state).tutor[0].status,
        TutorStatusV1::Invalidated
    );
}

/// The ruling's hidden-world test: two worlds with the same public history,
/// where a private library operation moves the marked card in only one of
/// them, have the same marker.
#[test]
fn tutor_marker_is_equal_across_worlds_with_the_same_public_history() {
    for private_scry in [true, false] {
        let base = tutored_game_v1(P0, 5);
        let library = base.players[0].library.clone();
        let (marked, next) = (library[0], library[1]);
        let link = |id: ObjectId| ObjectLinkV4 {
            object: id,
            zone_change_count: base.objects.get(id).zone_change_count,
        };
        let mut worlds = [base.clone(), base.clone()];
        for (world, moves) in worlds.iter_mut().zip([false, true]) {
            let order = if moves {
                [next, marked]
            } else {
                [marked, next]
            };
            if private_scry {
                world
                    .apply_scry_result(P0, &[link(marked), link(next)], &order, &[])
                    .unwrap();
            } else {
                world.reorder_library_top(P0, &order, &[P0]).unwrap();
            }
        }
        assert_ne!(worlds[0].players[0].library, worlds[1].players[0].library);
        let facts = worlds
            .each_ref()
            .map(|world| history_v1(world).facts_v1(world));
        assert_eq!(facts[0], facts[1], "private_scry {private_scry}");
        assert_eq!(facts[0].tutor[0].status, TutorStatusV1::Invalidated);
    }
}

/// Issue 740's extra-turn case: after Time Walk the same player takes the
/// next turn in the same round; the first land reads the previous turn and
/// the land played in the extra turn this turn.
#[cfg(feature = "eternal-fixtures")]
#[test]
fn issue_740_extra_turn_land_stamps_for_both_starters() {
    use crate::mana::ManaColor;
    for starter in [P0, P1] {
        let mut state = game_v1(starter);
        let first = play_land_v1(&mut state, starter);
        let walk = put(&mut state, starter, "Time Walk", Zone::Hand);
        state.players[starter.index()].mana_pool[ManaColor::U.pool_index()] = 2;
        engine::step(&mut state, Action::CastSpell(walk)).unwrap();
        let round = state.turn;
        advance_to_next_main_for(&mut state, starter);
        assert_eq!(state.turn, round, "the extra turn keeps the round");
        assert_eq!(entered_v1(&state, first), 1, "{starter:?}");
        let facts = history_v1(&state).facts_v1(&state);
        assert!(facts.consecutive_turn);
        assert_eq!(facts.turn_sequence, 2);
        assert_eq!(facts.turns_begun[starter.index()], 2);
        let second = play_land_v1(&mut state, starter);
        assert_eq!(entered_v1(&state, second), 0);
        assert_eq!(entered_v1(&state, first), 1);
    }
}

/// Issue 740's plot case: Pillage the Bog plotted on the starter's turn
/// reads this turn, then the previous turn at the other player's first
/// observation in the same round while still exiled.
#[cfg(feature = "eternal-fixtures")]
#[test]
fn issue_740_plot_stamps_for_both_starters() {
    use crate::mana::ManaColor;
    for starter in [P0, P1] {
        let mut state = game_v1(starter);
        let pillage = put(&mut state, starter, "Pillage the Bog", Zone::Hand);
        let pool = &mut state.players[starter.index()].mana_pool;
        pool[ManaColor::B.pool_index()] = 1;
        pool[ManaColor::G.pool_index()] = 1;
        pool[ManaColor::C.pool_index()] = 1;
        engine::step(&mut state, Action::PlotSpell(pillage)).unwrap();
        assert_eq!(state.objects.get(pillage).zone, Zone::Exile);
        let plotted = |state: &GameState| {
            history_v1(state)
                .plotted_turns_ago_v1(pillage.0, state.objects.get(pillage).zone_change_count)
        };
        assert_eq!(plotted(&state), 0);
        let round = state.turn;
        advance_to_next_main_for(&mut state, starter.opponent());
        assert_eq!(state.turn, round);
        assert_eq!(state.objects.get(pillage).zone, Zone::Exile);
        assert_eq!(plotted(&state), 1, "{starter:?}");
    }
}

/// The review's hidden-search case: a private search takes the marked card
/// in one world and another card in the other, then shuffles. The public
/// history is the same, so the marker is.
#[test]
fn tutor_marker_is_equal_whichever_card_a_private_search_takes() {
    let base = tutored_game_v1(P0, 5);
    let library = base.players[0].library.clone();
    let mut worlds = [base.clone(), base.clone()];
    for (world, taken) in worlds.iter_mut().zip([library[0], library[3]]) {
        event::propose_and_commit(world, ProposedEvent::zone_change(taken, Zone::Hand));
        assert_eq!(world.objects.get(taken).zone, Zone::Hand);
        let before_shuffle = history_v1(world).facts_v1(world).tutor[0];
        assert_eq!(before_shuffle.status, TutorStatusV1::Invalidated);
        world.shuffle_library(P0).unwrap();
    }
    let facts = worlds
        .each_ref()
        .map(|world| history_v1(world).facts_v1(world));
    assert_eq!(facts[0], facts[1]);
}

/// A partial randomization covering the marked card invalidates it; one
/// elsewhere does not.
#[test]
fn tutor_marker_sees_partial_randomizations() {
    let mut state = tutored_game_v1(P0, 6);
    state.randomize_library_bottom_v1(P0, 3).unwrap();
    assert_eq!(
        history_v1(&state).facts_v1(&state).tutor[0].status,
        TutorStatusV1::Tracked
    );
    state.randomize_library_top_v1(P0, 2).unwrap();
    let facts = history_v1(&state).facts_v1(&state).tutor[0];
    assert_eq!((facts.status, facts.depth), (TutorStatusV1::Invalidated, 0));
}
