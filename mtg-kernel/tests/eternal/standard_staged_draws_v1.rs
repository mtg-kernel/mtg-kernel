//! #926 items 7-10: Standard leaves that generate draws (Kaito, Bane of
//! Nightmares' +0, Braided Quipu, Shanna, Purifying Blade, Ertai Resurrected)
//! stage those draws so Dredge and the Jace or Hullbreacher replacement order
//! apply in cross-format Eternal play. Each leaf finishes its own effects
//! first and the staged draw keeps them across a save.
use mtg_kernel::card_def::{card_id_by_name, CARD_DEFS};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::event::{self, ProposedEvent};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::ManaColor;
use mtg_kernel::state::{GameObject, GameState, ObjectStateV4, Step, Target, Zone};

const P0: PlayerId = PlayerId::P0;
const P1: PlayerId = PlayerId::P1;

fn id(name: &str) -> u16 {
    card_id_by_name(name).unwrap_or_else(|| panic!("{name}"))
}

/// A first main phase with P0 active and holding priority. Each library
/// holds `library` in order, top first.
fn game(library: &[&str]) -> GameState {
    let ids: Vec<u16> = library.iter().map(|name| id(name)).collect();
    let mut state = GameState::new_from_libraries(
        &ids,
        &ids,
        |id| CARD_DEFS[id as usize].object_name.into(),
        0x0926_5d70_0001,
    );
    state.active_player = P0;
    state.priority_player = P0;
    state.step = Step::Main1;
    state
}

/// `name`, owned by `owner` and controlled by `controller`, in `zone`.
fn put_controlled(
    state: &mut GameState,
    owner: PlayerId,
    controller: PlayerId,
    name: &str,
    zone: Zone,
) -> ObjectId {
    let card_def = id(name);
    let object = state.objects.push(GameObject {
        card_def,
        name: CARD_DEFS[card_def as usize].object_name.into(),
        owner,
        controller,
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
    let seat = &mut state.players[controller.index()];
    match zone {
        Zone::Hand => seat.hand.push(object),
        Zone::Battlefield => seat.battlefield.push(object),
        Zone::Graveyard => seat.graveyard.push(object),
        Zone::Library => seat.library.insert(0, object),
        _ => panic!("helper zone"),
    }
    object
}

fn put(state: &mut GameState, player: PlayerId, name: &str, zone: Zone) -> ObjectId {
    put_controlled(state, player, player, name, zone)
}

/// Puts `name` onto the battlefield through a real zone change, so it enters
/// with its loyalty and static abilities.
fn enter(state: &mut GameState, player: PlayerId, name: &str) -> ObjectId {
    let object = put(state, player, name, Zone::Hand);
    event::propose_and_commit(state, ProposedEvent::zone_change(object, Zone::Battlefield));
    assert_eq!(state.objects.get(object).zone, Zone::Battlefield);
    object
}

fn pool(state: &mut GameState, player: PlayerId, colored: &[ManaColor], colorless: u8) {
    let pool = &mut state.players[player.index()].mana_pool;
    for color in colored {
        pool[color.pool_index()] += 1;
    }
    pool[ManaColor::C.pool_index()] += colorless;
}

fn zone(state: &GameState, object: ObjectId) -> Zone {
    state.objects.get(object).zone
}

fn treasures(state: &GameState, player: PlayerId) -> usize {
    state.players[player.index()]
        .battlefield
        .iter()
        .filter(|&&object| {
            CARD_DEFS[state.objects.get(object).card_def as usize].name == "Treasure Token"
        })
        .count()
}

fn next(state: &mut GameState) -> Decision {
    let decision = engine::advance_until_decision(state);
    assert!(
        !matches!(decision, Decision::Halted { .. }),
        "{decision:?}\nstack: {:#?}\npending: {:#?}",
        state.stack,
        state.engine.pending_effect
    );
    decision
}

fn act(state: &mut GameState, action: Action) {
    engine::step(state, action.clone()).unwrap_or_else(|error| panic!("{action:?}: {error}"));
}

/// Passes priority and orders triggers until the stack and pending triggers
/// are empty, returning the first other decision if one interrupts.
fn settle(state: &mut GameState) -> Option<Decision> {
    for _ in 0..200 {
        match next(state) {
            Decision::OrderTriggers { pending, .. } => {
                act(state, Action::OrderTriggers((0..pending.len()).collect()))
            }
            Decision::CastSpellOrPass { .. }
                if state.stack.is_empty() && state.engine.pending_triggers.is_empty() =>
            {
                return None
            }
            Decision::CastSpellOrPass { .. } => act(state, Action::Pass),
            other => return Some(other),
        }
    }
    panic!("did not settle");
}

fn settled(state: &mut GameState) {
    if let Some(other) = settle(state) {
        panic!("unexpected choice: {other:?}");
    }
}

/// A save and load of the current state, with its pending choice validated.
fn restored(state: &GameState) -> GameState {
    let restored: GameState = serde_json::from_slice(&serde_json::to_vec(state).unwrap()).unwrap();
    assert_eq!(state.state_hash(), restored.state_hash());
    mtg_kernel::effect::validate_pending_effect_choice(&restored).unwrap();
    restored
}

/// The drawer's Jace, Wielder of Mysteries and the opponent's Hullbreacher
/// compete for a draw from `drawer`'s empty library.
fn competing_replacements(state: &mut GameState, drawer: PlayerId) -> ObjectId {
    assert!(state.players[drawer.index()].library.is_empty());
    put(state, drawer.opponent(), "Hullbreacher", Zone::Battlefield);
    enter(state, drawer, "Jace, Wielder of Mysteries")
}

/// Settles to the drawer's Jace or Hullbreacher choice, through a save.
fn jace_choice(state: &mut GameState, drawer: PlayerId, jace: ObjectId) {
    match settle(state) {
        Some(Decision::ChooseEffectTargets {
            player,
            legal_targets,
            can_finish: true,
            ..
        }) => {
            assert_eq!(player, drawer);
            assert_eq!(legal_targets, vec![Target::Object(jace)]);
        }
        other => panic!("expected the Jace or Hullbreacher choice, got {other:?}"),
    }
    *state = restored(state);
}

/// Choosing Jace wins the game for the drawer before anything else happens.
fn jace_wins(state: &GameState, drawer: PlayerId, jace: ObjectId) {
    let mut win = restored(state);
    act(&mut win, Action::ChooseEffectTarget(Target::Object(jace)));
    assert!(
        matches!(next(&mut win), Decision::GameOver { winner: Some(winner) } if winner == drawer)
    );
    assert_eq!(
        treasures(&win, drawer.opponent()),
        treasures(state, drawer.opponent())
    );
}

// ---- Kaito, Bane of Nightmares (item 7) ---------------------------------

fn kaito_zero(state: &mut GameState) -> ObjectId {
    let kaito = enter(state, P0, "Kaito, Bane of Nightmares");
    state.objects.get_mut(kaito).summoning_sick = false;
    assert!(
        matches!(next(state), Decision::CastSpellOrPass { activatable_abilities, .. }
        if activatable_abilities.contains(&(kaito, 1)))
    );
    act(state, Action::ActivateAbility(kaito, 1));
    kaito
}

#[test]
fn kaito_draw_orders_jace_and_hullbreacher_after_its_condition() {
    let mut state = game(&[]);
    let jace = competing_replacements(&mut state, P0);
    event::propose_and_commit(&mut state, ProposedEvent::life_loss(P1, 2));
    kaito_zero(&mut state);
    jace_choice(&mut state, P0, jace);
    jace_wins(&state, P0, jace);
    act(&mut state, Action::FinishEffectSelection);
    settled(&mut state);
    assert_eq!(treasures(&state, P1), 1);
    assert!(!state.players[0].has_lost);
}

#[test]
fn kaito_draws_nothing_when_no_opponent_lost_life() {
    let mut state = game(&[]);
    competing_replacements(&mut state, P0);
    kaito_zero(&mut state);
    settled(&mut state);
    assert_eq!(treasures(&state, P1), 0);
    assert!(!state.players[0].has_lost);
    assert!(state.engine.halted.is_none());
}

// ---- Braided Quipu (item 8) ---------------------------------------------

/// Crafts Braided Net into Braided Quipu, leaving Quipu and Clay-Fired
/// Bricks as P0's two artifacts.
fn quipu(state: &mut GameState) -> ObjectId {
    let braided = put(state, P0, "Braided Net", Zone::Battlefield);
    let trail = put(state, P0, "Candy Trail", Zone::Battlefield);
    put(state, P0, "Clay-Fired Bricks", Zone::Battlefield);
    pool(state, P0, &[ManaColor::U, ManaColor::U], 0);
    act(state, Action::ActivateAbility(braided, 1));
    for _ in 0..20 {
        match settle(state) {
            Some(Decision::ChooseTargets { .. }) => {
                act(state, Action::ChooseTarget(Target::Object(trail)))
            }
            Some(Decision::ChooseCostTargets { .. }) => act(state, Action::ChooseCostTarget(trail)),
            None => break,
            Some(other) => panic!("unexpected craft decision: {other:?}"),
        }
    }
    assert_eq!(zone(state, trail), Zone::Exile);
    assert_eq!(state.objects.get(braided).v4.face_index, 1);
    pool(state, P0, &[ManaColor::U; 4], 0);
    act(state, Action::ActivateAbility(braided, 2));
    braided
}

#[test]
fn braided_quipu_dredges_its_counted_draws_before_going_third_from_top() {
    let mut state = game(&["Island"; 10]);
    let thug = put(&mut state, P0, "Golgari Thug", Zone::Graveyard);
    let hand = state.players[0].hand.len();
    let braided = quipu(&mut state);
    match settle(&mut state) {
        Some(Decision::ChooseEffectTargets {
            player,
            legal_targets,
            ..
        }) => {
            assert_eq!(player, P0);
            assert_eq!(legal_targets, vec![Target::Object(thug)]);
        }
        other => panic!("expected the first draw's Dredge choice, got {other:?}"),
    }
    state = restored(&state);
    // Quipu stays on the battlefield until both counted draws finish.
    assert_eq!(zone(&state, braided), Zone::Battlefield);
    act(&mut state, Action::ChooseEffectTarget(Target::Object(thug)));
    settled(&mut state);
    assert_eq!(zone(&state, thug), Zone::Hand);
    // The Thug and one ordinary draw, both counted before Quipu moved.
    assert_eq!(state.players[0].hand.len(), hand + 2);
    assert_eq!(zone(&state, braided), Zone::Library);
    assert_eq!(state.players[0].library[2], braided);
    assert_eq!(state.players[0].library.len(), 10 - 4 - 1 + 1);
}

#[test]
fn braided_quipu_offers_jace_or_hullbreacher_on_each_counted_draw() {
    let mut state = game(&[]);
    let jace = competing_replacements(&mut state, P0);
    let braided = quipu(&mut state);
    for _ in 0..2 {
        jace_choice(&mut state, P0, jace);
        assert_eq!(zone(&state, braided), Zone::Battlefield);
        jace_wins(&state, P0, jace);
        act(&mut state, Action::FinishEffectSelection);
    }
    settled(&mut state);
    assert_eq!(treasures(&state, P1), 2);
    assert_eq!(zone(&state, braided), Zone::Library);
}

// ---- Shanna, Purifying Blade (item 9) -----------------------------------

#[test]
fn shanna_keeps_its_completed_payment_across_staged_replacement_choices() {
    let mut state = game(&[]);
    state.step = Step::Main2;
    put(&mut state, P0, "Shanna, Purifying Blade", Zone::Battlefield);
    let forests: Vec<_> = (0..3)
        .map(|_| put(&mut state, P0, "Forest", Zone::Battlefield))
        .collect();
    let jace = competing_replacements(&mut state, P0);
    event::propose_and_commit(&mut state, ProposedEvent::life_gain(P0, 4));
    // Pass to the end step, where Shanna offers to pay up to the life gained.
    let mut offered = false;
    for _ in 0..30 {
        match next(&mut state) {
            Decision::CastSpellOrPass { .. } => act(&mut state, Action::Pass),
            Decision::ChooseEffectOption { option_count, .. } => {
                assert_eq!(option_count, 4);
                offered = true;
                break;
            }
            other => panic!("unexpected Shanna decision: {other:?}"),
        }
    }
    assert!(offered);
    act(&mut state, Action::ChooseEffectOption(2));
    let tapped = |state: &GameState| {
        forests
            .iter()
            .filter(|&&f| state.objects.get(f).tapped)
            .count()
    };
    for _ in 0..2 {
        jace_choice(&mut state, P0, jace);
        // The payment was made once, before either draw.
        assert_eq!(tapped(&state), 2);
        jace_wins(&state, P0, jace);
        act(&mut state, Action::FinishEffectSelection);
    }
    settled(&mut state);
    assert_eq!(tapped(&state), 2);
    assert_eq!(treasures(&state, P1), 2);
}

// ---- Ertai Resurrected (item 10) ----------------------------------------

/// Casts Ertai and chooses `mode` for its entering trigger, targeting `target`.
fn ertai(state: &mut GameState, mode: u8, target: Target) {
    let ertai = put(state, P0, "Ertai Resurrected", Zone::Hand);
    pool(state, P0, &[ManaColor::U, ManaColor::B], 2);
    assert!(
        matches!(next(state), Decision::CastSpellOrPass { castable_spells, .. }
        if castable_spells.contains(&ertai))
    );
    act(state, Action::CastSpell(ertai));
    loop {
        match settle(state) {
            Some(Decision::ChooseTriggerMode { legal_modes, .. }) => {
                assert!(legal_modes.contains(&mode));
                act(state, Action::ChooseTriggerMode(mode));
            }
            Some(Decision::ChooseTargets { legal_targets, .. }) => {
                assert!(legal_targets.contains(&target), "{legal_targets:?}");
                act(state, Action::ChooseTarget(target));
                return;
            }
            other => panic!("unexpected Ertai decision: {other:?}"),
        }
    }
}

#[test]
fn ertai_destroy_draw_uses_the_controller_saved_before_destruction() {
    // P1 controls P0's creature. After it dies it is P0's graveyard card, but
    // P1 was its controller and draws, choosing between P1's Jace and P0's
    // Hullbreacher.
    let mut state = game(&[]);
    let jace = competing_replacements(&mut state, P1);
    let stolen = put_controlled(&mut state, P0, P1, "Llanowar Elves", Zone::Battlefield);
    ertai(&mut state, 1, Target::Object(stolen));
    jace_choice(&mut state, P1, jace);
    assert_eq!(zone(&state, stolen), Zone::Graveyard);
    assert!(state.players[0].graveyard.contains(&stolen));
    jace_wins(&state, P1, jace);
    act(&mut state, Action::FinishEffectSelection);
    settled(&mut state);
    assert_eq!(treasures(&state, P0), 1);
    assert!(!state.players[1].has_lost);
}

#[test]
fn ertai_counter_draw_uses_the_countered_spells_controller() {
    let mut state = game(&[]);
    let jace = competing_replacements(&mut state, P0);
    let bolt = put(&mut state, P0, "Lightning Bolt", Zone::Hand);
    pool(&mut state, P0, &[ManaColor::R], 0);
    assert!(
        matches!(next(&mut state), Decision::CastSpellOrPass { castable_spells, .. }
        if castable_spells.contains(&bolt))
    );
    act(&mut state, Action::CastSpell(bolt));
    match next(&mut state) {
        Decision::ChooseTargets { legal_targets, .. } => {
            assert!(legal_targets.contains(&Target::Player(P1)));
            act(&mut state, Action::ChooseTarget(Target::Player(P1)));
        }
        other => panic!("expected Bolt's target, got {other:?}"),
    }
    let spell = state.stack.last().unwrap().v4.stack_item_id;
    ertai(&mut state, 0, Target::StackItem(spell));
    jace_choice(&mut state, P0, jace);
    assert_eq!(zone(&state, bolt), Zone::Graveyard);
    assert_eq!(state.players[1].life, 20);
    jace_wins(&state, P0, jace);
    act(&mut state, Action::FinishEffectSelection);
    settled(&mut state);
    assert_eq!(treasures(&state, P1), 1);
}

// ---- Items 3-6 ----------------------------------------------------------

/// Inspiring Call, Incinerating Blast (the Crystal-owned discard and draw),
/// Garruk's Uprising and Vampire Gourmand are FDN-only. Eternal builds append
/// the Standard registry instead of FDN's, so these direct draws never meet
/// Dredge, Jace or Hullbreacher. Adding one to an Eternal catalog must first
/// stage its draw (#926 items 3-6).
#[test]
fn fdn_only_direct_draw_sites_stay_outside_eternal_catalogs() {
    for name in [
        "Inspiring Call",
        "Incinerating Blast",
        "Crystal Barricade",
        "Twinflame Tyrant",
        "Garruk's Uprising",
        "Vampire Gourmand",
    ] {
        assert!(
            card_id_by_name(name).is_none(),
            "{name} entered an Eternal build"
        );
    }
    for name in [
        "Hullbreacher",
        "Jace, Wielder of Mysteries",
        "Golgari Thug",
        "Kaito, Bane of Nightmares",
        "Braided Net",
        "Shanna, Purifying Blade",
        "Ertai Resurrected",
    ] {
        assert!(card_id_by_name(name).is_some(), "{name}");
    }
}
