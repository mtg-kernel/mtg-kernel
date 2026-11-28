//! Issue #957: a mandatory infinite loop is a natural draw (rules 104.4b and
//! 731.4), and only a mandatory one.
//!
//! The loop is hand-built from two permanents, one for each player: Ajani's
//! Pridemate hosts the test ability "Whenever you gain life, you lose 1 life
//! and each opponent gains 1 life" (existing condition and effects, installed
//! with `trigger::TestTriggersV1`; no supported card pair forms a stateless
//! mandatory loop by itself). P0 plays Scoured Barrens, whose entry gain
//! starts it: each trigger moves one life point to the other player and the
//! other player's copy moves it back, with no player holding a real option,
//! so every window is a forced pass and the position repeats every four
//! forced transitions.
//!
//! Controls: the same loop where P1 keeps an untapped Forest (a mana ability
//! in every window, so every repetition offers a real decision) runs to the
//! session's decision cap; Bloodthirsty Conqueror with Marauding Blight-Priest
//! is a real mandatory drain whose life totals change every repetition, which
//! ends with its winner, and a hand-built drain on one permanent does the same
//! after far more forced transitions than the gate.

use super::*;
use crate::card_def::card_id_by_name;
use crate::effect::{EffectOp, PlayerRef};
use crate::engine::{Action, Decision};
use crate::ids::{ObjectId, PlayerId};
use crate::mandatory_loop_v1::{MandatoryLoopDrawV1, UNCHECKED_FORCED_TRANSITIONS_V1};
use crate::state::{GameObject, GameState, ObjectStateV4, Step, Zone};
use crate::surface_v2::{
    HarnessSurfaceV2, PriorityModeV1, SuppressionCounts, SurfaceAction, SurfaceDecision,
};
use crate::trigger::{TestTriggersV1, TriggerCondition, TriggeredAbilityDef};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

const HOST: &str = "Ajani's Pridemate";

/// "You lose 1 life and each opponent gains 1 life."
fn lose_one_then_each_opponent_gains_one() -> EffectOp {
    EffectOp::Sequence(vec![
        EffectOp::LoseLife {
            player: PlayerRef::Controller,
            amount: 1,
        },
        EffectOp::GainLife {
            player: PlayerRef::Opponent,
            amount: 1,
        },
    ])
}

const HAND_BUILT_TRIGGERS: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::ControllerGainsLife,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    face_index: 0,
    effect: lose_one_then_each_opponent_gains_one,
}];

fn hand_built_triggers() -> TestTriggersV1 {
    TestTriggersV1::install(card_id_by_name(HOST).unwrap(), &HAND_BUILT_TRIGGERS)
}

/// Runs `test` on its own thread with a time bound: an unadjudicated loop
/// would never return.
fn bounded(test: impl FnOnce() + Send + 'static) {
    let (done, finished) = mpsc::channel();
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            test();
            let _ = done.send(());
        })
        .unwrap();
    match finished.recv_timeout(Duration::from_secs(600)) {
        Ok(()) => {}
        Err(RecvTimeoutError::Timeout) => {
            panic!("no terminal within the bound: the loop was not adjudicated")
        }
        Err(RecvTimeoutError::Disconnected) => panic!("the test thread panicked (see above)"),
    }
}

fn put(state: &mut GameState, player: PlayerId, name: &str, zone: Zone) -> ObjectId {
    let card_def = card_id_by_name(name).unwrap_or_else(|| panic!("{name}"));
    let id = state.objects.push(GameObject {
        card_def,
        name: name.into(),
        owner: player,
        controller: player,
        zone,
        tapped: false,
        summoning_sick: false,
        damage: 0,
        counters: Default::default(),
        attachments: Vec::new(),
        v4: ObjectStateV4::from_card_def(card_def),
        spell_copy_origin: None,
        plotted_turn: None,
        zone_change_count: 0,
    });
    match zone {
        Zone::Hand => state.players[player.index()].hand.push(id),
        Zone::Battlefield => state.players[player.index()].battlefield.push(id),
        _ => panic!("helper zone"),
    }
    id
}

/// P0's first main phase: `p0` and `p1` on the battlefield, Scoured Barrens
/// in P0's hand, nothing else in either hand.
fn board(p0: &[&str], p1: &[&str]) -> (GameState, ObjectId) {
    let forest = card_id_by_name("Forest").unwrap();
    let mut state =
        GameState::new_from_libraries(&[forest; 40], &[forest; 40], |_| "Forest".into(), 957);
    state.active_player = PlayerId::P0;
    state.priority_player = PlayerId::P0;
    state.step = Step::Main1;
    for name in p0 {
        put(&mut state, PlayerId::P0, name, Zone::Battlefield);
    }
    for name in p1 {
        put(&mut state, PlayerId::P1, name, Zone::Battlefield);
    }
    let barrens = put(&mut state, PlayerId::P0, "Scoured Barrens", Zone::Hand);
    (state, barrens)
}

fn forced_transitions(counts: SuppressionCounts) -> u64 {
    counts.step_gated
        + counts.no_real_option
        + counts.no_eligible_attacker
        + counts.no_eligible_blockers_for_attacker
        + counts.combat_priority_action_spent
        + counts.stack_top_is_casters_own
}

fn counting_surface() -> HarnessSurfaceV2 {
    HarnessSurfaceV2::new_with_priority_mode_v1(
        PriorityModeV1::HarnessV2,
        SuppressionAuditMode::CountOnly,
    )
}

/// P0 plays Scoured Barrens from the opening priority window.
fn play_barrens(surface: &mut HarnessSurfaceV2, state: &mut GameState, barrens: ObjectId) {
    let SurfaceDecision::Decision(Decision::CastSpellOrPass {
        player, land_drops, ..
    }) = surface.next_decision(state)
    else {
        panic!("expected P0's opening priority window");
    };
    assert_eq!(player, PlayerId::P0);
    assert_eq!(land_drops, vec![barrens]);
    surface
        .apply(state, SurfaceAction::Action(Action::PlayLand(barrens)))
        .unwrap();
}

fn assert_stays_drawn(state: &GameState) {
    let mut state = state.clone();
    assert_eq!(
        crate::engine::advance_until_decision(&mut state),
        Decision::GameOver { winner: None }
    );
    assert!(crate::engine::step(&mut state, Action::Pass).is_err());
    let json = serde_json::to_string(&state).unwrap();
    assert!(json.contains(r#""mandatory_loop_draw_v1":{"period":4}"#));
    let mut restored: GameState = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.state_hash(), state.state_hash());
    assert_eq!(
        crate::engine::advance_until_decision(&mut restored),
        Decision::GameOver { winner: None }
    );
}

fn assert_natural_draw(terminal: &RlSessionTerminalV1) {
    assert_eq!(terminal.terminal_outcome, TerminalOutcomeV1::Draw);
    assert_eq!(
        terminal.terminal_classification,
        TerminalClassificationV1::Natural
    );
    assert_eq!(terminal.terminal_code, TerminalSafeCodeV2::NaturalGameOver);
    assert_eq!(terminal.terminal_reason, "game_over");
    assert_eq!(terminal.winner, None);
    assert_eq!(terminal.terminal_reward, [0, 0]);
}

fn fast_session(state: GameState, max_physical_decisions: u64) -> FastActorSessionV1 {
    let mut session = FastActorSessionV1::reset_with_limits(
        23,
        91,
        max_physical_decisions,
        max_physical_decisions * 4,
    );
    session.state = state;
    session.surface = PolicySurfaceV5::new();
    session.environment_revision = 0;
    session.policy_step_count = 0;
    session.physical_decision_count = 0;
    session.current = None;
    session.flat_action_cache_spare = None;
    session.flat_action_cache_spare_v2 = None;
    session.terminal = None;
    session.flat_action_contract_mode = FlatActionContractModeV1::V2;
    session.advance_to_decision_or_terminal();
    session
}

fn episode_session(state: GameState, max_physical_decisions: u64) -> RlEpisodeSessionV1 {
    let mut session = RlEpisodeSessionV1::reset_with_limits(
        23,
        91,
        max_physical_decisions,
        max_physical_decisions * 4,
    );
    session.state = state;
    session.surface = PolicySurfaceV5::new_with_priority_mode_v1(
        PriorityModeV1::HarnessV2,
        SuppressionAuditMode::Off,
    );
    session.environment_revision = 0;
    session.policy_step_count = 0;
    session.physical_decision_count = 0;
    session.current = None;
    session.terminal = None;
    session.limited_trigger_order = None;
    session.advance_to_decision_or_terminal();
    session
}

fn fast_action(session: &FastActorSessionV1, wanted: impl Fn(&ActionSemanticV1) -> bool) -> u32 {
    session
        .diagnostic_current_action_semantics()
        .expect("current decision")
        .iter()
        .position(wanted)
        .expect("wanted action offered") as u32
}

fn is_play(barrens: ObjectId) -> impl Fn(&ActionSemanticV1) -> bool {
    move |semantic| {
        matches!(semantic, ActionSemanticV1::PlayLand { source, .. }
            if source.arena_id == barrens.0)
    }
}

#[test]
fn the_hand_built_two_permanent_loop_is_a_draw_at_the_surface() {
    bounded(|| {
        let _triggers = hand_built_triggers();
        let (mut state, barrens) = board(&[HOST], &[HOST]);
        let mut surface = counting_surface();
        play_barrens(&mut surface, &mut state, barrens);
        assert_eq!(
            surface.next_decision(&mut state),
            SurfaceDecision::Decision(Decision::GameOver { winner: None })
        );
        assert_eq!(
            state.mandatory_loop_draw_v1,
            Some(MandatoryLoopDrawV1 { period: 4 })
        );
        // The loop ran past the unwatched transitions and no further than a
        // few repetitions beyond them.
        let forced = forced_transitions(surface.suppression_counts());
        assert!(
            forced > u64::from(UNCHECKED_FORCED_TRANSITIONS_V1),
            "{forced}"
        );
        assert!(
            forced < u64::from(UNCHECKED_FORCED_TRANSITIONS_V1) + 64,
            "{forced}"
        );
        // One life point is in flight between the players.
        let lives = [state.players[0].life, state.players[1].life];
        assert_eq!(lives.iter().sum::<i32>(), 41, "{lives:?}");
        assert_eq!(state.engine.halted, None);
        assert_stays_drawn(&state);
        assert_eq!(
            surface.next_decision(&mut state),
            SurfaceDecision::Decision(Decision::GameOver { winner: None })
        );
    });
}

#[test]
fn every_session_reports_the_loop_as_a_natural_draw() {
    bounded(|| {
        let _triggers = hand_built_triggers();
        let (state, barrens) = board(&[HOST], &[HOST]);

        let mut fast = fast_session(state.clone(), 64);
        let FastActorResponseV1::Decision(decision) = fast.current_response() else {
            panic!("expected P0's opening decision");
        };
        let play = fast_action(&fast, is_play(barrens));
        let FastActorResponseV1::Terminal(terminal) =
            fast.step(decision.episode_id, decision.step, play).unwrap()
        else {
            panic!("the flat actor session exposed a decision inside the loop");
        };
        assert_natural_draw(&terminal);
        assert_eq!(terminal.physical_decision_count, 1);
        assert_eq!(
            fast.game_state().mandatory_loop_draw_v1,
            Some(MandatoryLoopDrawV1 { period: 4 })
        );
        assert_stays_drawn(fast.game_state());
        let FastActorResponseV1::Terminal(again) = fast.current_response() else {
            panic!("the drawn session exposed a decision");
        };
        assert_natural_draw(&again);

        let mut episode = episode_session(state, 64);
        let RlSessionResponseV1::Decision(decision) = episode.current_response() else {
            panic!("expected P0's opening decision");
        };
        let play = decision
            .legal_actions
            .iter()
            .find(|action| is_play(barrens)(&action.semantic))
            .expect("Scoured Barrens offered");
        let RlSessionResponseV1::Terminal(terminal) = episode
            .step(
                decision.episode_id,
                decision.step,
                play.selected_index,
                &play.stable_id,
            )
            .unwrap()
        else {
            panic!("the episode session exposed a decision inside the loop");
        };
        assert_natural_draw(&terminal);
        assert_eq!(
            episode.game_state().mandatory_loop_draw_v1,
            Some(MandatoryLoopDrawV1 { period: 4 })
        );
        assert_stays_drawn(episode.game_state());
    });
}

#[test]
fn the_same_loop_with_a_real_option_every_repetition_is_never_adjudicated() {
    bounded(|| {
        let _triggers = hand_built_triggers();
        // P1's untapped Forest is a mana ability in every P1 window, so each
        // repetition hands P1 a real decision.
        let (state, barrens) = board(&[HOST], &[HOST, "Forest"]);
        let cap = 2 * u64::from(UNCHECKED_FORCED_TRANSITIONS_V1) + 2;
        let mut session = fast_session(state, cap);
        let FastActorResponseV1::Decision(decision) = session.current_response() else {
            panic!("expected P0's opening decision");
        };
        let play = fast_action(&session, is_play(barrens));
        let mut response = session
            .step(decision.episode_id, decision.step, play)
            .unwrap();
        let terminal = loop {
            match response {
                FastActorResponseV1::Decision(decision) => {
                    assert_eq!(decision.acting_player, PlayerSeatV1::P1);
                    let pass = fast_action(&session, |semantic| {
                        matches!(semantic, ActionSemanticV1::Pass { .. })
                    });
                    response = session
                        .step(decision.episode_id, decision.step, pass)
                        .unwrap();
                }
                FastActorResponseV1::Terminal(terminal) => break terminal,
            }
        };
        assert_eq!(
            terminal.terminal_classification,
            TerminalClassificationV1::Truncated
        );
        assert_eq!(
            terminal.terminal_reason,
            format!("physical_decision_cap_reached:{cap}")
        );
        assert_eq!(session.game_state().mandatory_loop_draw_v1, None);
        let lives = [
            session.game_state().players[0].life,
            session.game_state().players[1].life,
        ];
        assert_eq!(lives.iter().sum::<i32>(), 41, "{lives:?}");
    });
}

#[test]
fn the_real_two_permanent_drain_ends_with_its_winner() {
    bounded(|| {
        let (mut state, barrens) =
            board(&["Marauding Blight-Priest", "Bloodthirsty Conqueror"], &[]);
        let mut surface = counting_surface();
        play_barrens(&mut surface, &mut state, barrens);
        assert_eq!(
            surface.next_decision(&mut state),
            SurfaceDecision::Decision(Decision::GameOver {
                winner: Some(PlayerId::P0)
            })
        );
        assert_eq!(state.mandatory_loop_draw_v1, None);
        assert_eq!(state.players[1].life, 0);
        assert!(state.players[0].life > 20, "{}", state.players[0].life);
    });
}

/// "Each opponent loses 1 life and you gain 1 life."
fn drain_one() -> EffectOp {
    EffectOp::Sequence(vec![
        EffectOp::LoseLife {
            player: PlayerRef::Opponent,
            amount: 1,
        },
        EffectOp::GainLife {
            player: PlayerRef::Controller,
            amount: 1,
        },
    ])
}

const HAND_BUILT_DRAIN: [TriggeredAbilityDef; 1] = [TriggeredAbilityDef {
    condition: TriggerCondition::ControllerGainsLife,
    home_zone: Zone::Battlefield,
    intervening_if_kicked: false,
    intervening_if_controls_another_source_card: false,
    face_index: 0,
    effect: drain_one,
}];

#[test]
fn a_finite_mandatory_drain_ends_with_its_winner_past_the_gate() {
    // Bloodthirsty Conqueror revalidates every earlier loss batch of the turn
    // on each advance, so a long real drain is slow; the same mandatory
    // re-trigger on one hand-built permanent ("Whenever you gain life, each
    // opponent loses 1 life and you gain 1 life") runs from 1,000 life.
    bounded(|| {
        let _drain = TestTriggersV1::install(card_id_by_name(HOST).unwrap(), &HAND_BUILT_DRAIN);
        let (mut state, barrens) = board(&[HOST], &[]);
        state.players[1].life = 1_000;
        let mut surface = counting_surface();
        play_barrens(&mut surface, &mut state, barrens);
        assert_eq!(
            surface.next_decision(&mut state),
            SurfaceDecision::Decision(Decision::GameOver {
                winner: Some(PlayerId::P0)
            })
        );
        assert_eq!(state.mandatory_loop_draw_v1, None);
        assert_eq!(state.players[1].life, 0);
        assert!(state.players[0].life > 1_000, "{}", state.players[0].life);
        let forced = forced_transitions(surface.suppression_counts());
        assert!(
            forced > u64::from(UNCHECKED_FORCED_TRANSITIONS_V1),
            "{forced}"
        );
    });
}
