//! Phasing (702.26) with Talon Gates of Madara (#946): a phased-out permanent
//! keeps its battlefield incarnation (702.26d) but is treated as though it
//! does not exist (702.26b). Stack items whose source or target phased out,
//! permanents with Auras and Equipment (indirect phasing, 702.26g), linked
//! exile, last known information and the schema 5 public record
//! (`phased_out`) are followed through real engine play.
#![cfg(feature = "eternal-fixtures")]

use mtg_kernel::card_def::{card_id_by_name, CARD_DEFS};
use mtg_kernel::engine::{self, Action, Decision};
use mtg_kernel::event::{propose_and_commit, ProposedEvent};
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::mana::ManaColor;
use mtg_kernel::policy_surface_v5::PolicySurfaceV5;
use mtg_kernel::state::{GameObject, GameState, ObjectLinkV4, ObjectStateV4, Step, Target, Zone};
use serde_json::{json, Value};

const P0: PlayerId = PlayerId::P0;
const P1: PlayerId = PlayerId::P1;

fn id(name: &str) -> u16 {
    card_id_by_name(name).unwrap_or_else(|| panic!("{name}"))
}

/// P0's first main phase, P0 holding priority, under the schema 5 public
/// contract when `public`.
fn game(public: bool) -> GameState {
    let ids = vec![id("Island"); 30];
    let mut state = GameState::new_from_libraries(
        &ids,
        &ids,
        |id| CARD_DEFS[id as usize].object_name.into(),
        0x5048_4153_4531,
    );
    state.active_player = P0;
    state.priority_player = P0;
    state.step = Step::Main1;
    if public {
        mtg_kernel::rl::enable_public_shroud_v1(&mut state);
    }
    state
}

fn put(state: &mut GameState, player: PlayerId, name: &str, zone: Zone) -> ObjectId {
    let card_def = id(name);
    let object = state.objects.push(GameObject {
        card_def,
        name: CARD_DEFS[card_def as usize].object_name.into(),
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
    let seat = &mut state.players[player.index()];
    match zone {
        Zone::Hand => seat.hand.push(object),
        Zone::Battlefield => seat.battlefield.push(object),
        Zone::Graveyard => seat.graveyard.push(object),
        _ => panic!("helper zone"),
    }
    object
}

/// Attaches `attachment` to `host` (both on the battlefield).
fn attach(state: &mut GameState, attachment: ObjectId, host: ObjectId) {
    let link = ObjectLinkV4 {
        object: host,
        zone_change_count: state.objects.get(host).zone_change_count,
    };
    state.objects.get_mut(host).attachments.push(attachment);
    state.objects.get_mut(attachment).v4.attached_to = Some(link);
}

fn pool(state: &mut GameState, player: PlayerId, colored: &[ManaColor], colorless: u8) {
    let pool = &mut state.players[player.index()].mana_pool;
    for color in colored {
        pool[color.pool_index()] += 1;
    }
    pool[ManaColor::C.pool_index()] += colorless;
}

fn next(state: &mut GameState) -> Decision {
    let decision = engine::advance_until_decision(state);
    assert!(
        !matches!(decision, Decision::Halted { .. }),
        "{decision:?}\nstack: {:#?}",
        state.stack
    );
    decision
}

fn act(state: &mut GameState, action: Action) {
    engine::step(state, action.clone()).unwrap_or_else(|error| panic!("{action:?}: {error}"));
}

/// The answer to a decision no test cares about.
fn routine(decision: Decision) -> Action {
    match decision {
        Decision::CastSpellOrPass { .. } => Action::Pass,
        Decision::OrderTriggers { pending, .. } => {
            Action::OrderTriggers((0..pending.len()).collect())
        }
        Decision::DeclareAttackers { .. } => Action::DeclareAttackers(Vec::new()),
        Decision::DeclareBlockers { .. } => Action::DeclareBlockers(Vec::new()),
        Decision::ChooseTargets {
            can_finish: true, ..
        } => Action::FinishEffectSelection,
        Decision::ChooseTargets { legal_targets, .. } => Action::ChooseTarget(legal_targets[0]),
        Decision::ChooseEffectTargets {
            can_finish: true, ..
        } => Action::FinishEffectSelection,
        Decision::ChooseEffectTargets { legal_targets, .. } => {
            Action::ChooseEffectTarget(legal_targets[0])
        }
        Decision::ChooseEffectOption { .. } => Action::ChooseEffectOption(0),
        Decision::ChooseEffectBoolean { .. } => Action::ChooseEffectBoolean(false),
        Decision::Discard { count, choices, .. } => {
            Action::Discard(choices.into_iter().take(count as usize).collect())
        }
        other => panic!("unexpected decision: {other:?}"),
    }
}

/// Answers routine decisions until `stop` holds at a decision.
fn run_until(state: &mut GameState, stop: impl Fn(&GameState, &Decision) -> bool) -> Decision {
    for _ in 0..400 {
        let decision = next(state);
        if stop(state, &decision) {
            return decision;
        }
        act(state, routine(decision));
    }
    panic!("never reached the expected decision");
}

fn priority_for(player: PlayerId) -> impl Fn(&GameState, &Decision) -> bool {
    move |_, decision| matches!(decision, Decision::CastSpellOrPass { player: p, .. } if *p == player)
}

/// Passes until the stack is empty at a priority decision.
fn settle(state: &mut GameState) {
    run_until(state, |state, decision| {
        matches!(decision, Decision::CastSpellOrPass { .. })
            && state.stack.is_empty()
            && state.engine.pending_triggers.is_empty()
    });
}

/// `player`'s next first main phase, its untap step behind it. (`turn` is
/// the round, which both players' turns share.)
fn to_main_of(state: &mut GameState, player: PlayerId) {
    if state.active_player == player {
        run_until(state, move |state, _| state.active_player != player);
    }
    run_until(state, move |state, decision| {
        matches!(decision, Decision::CastSpellOrPass { .. })
            && state.active_player == player
            && state.step == Step::Main1
            && state.stack.is_empty()
    });
}

fn phased_out(state: &GameState, object: ObjectId) -> bool {
    mtg_kernel::state::object_is_phased_out(state, object)
}

/// `player` activates Talon Gates of Madara from their hand ({4}: put it
/// onto the battlefield) at their next priority, and its entry trigger phases
/// `target` out. Returns at the first decision after `target` phased out.
fn phase_out_with_gates(state: &mut GameState, player: PlayerId, target: ObjectId) -> ObjectId {
    let gates = put(state, player, "Talon Gates of Madara", Zone::Hand);
    run_until(state, priority_for(player));
    pool(state, player, &[], 4);
    act(state, Action::ActivateAbility(gates, 0));
    for _ in 0..60 {
        match next(state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert!(legal_targets.contains(&Target::Object(target)));
                act(state, Action::ChooseTarget(Target::Object(target)));
            }
            _ if phased_out(state, target) => return gates,
            decision => act(state, routine(decision)),
        }
    }
    panic!("the target never phased out");
}

/// The public schema 5 observation at the current decision.
fn public_view(state: &GameState) -> Value {
    let mut state = state.clone();
    let mut surface = PolicySurfaceV5::new();
    let decision = surface.next_decision(&mut state).unwrap();
    let actor = decision
        .actor(&state)
        .unwrap_or_else(|| panic!("no actor: {decision:?}"));
    let observation = mtg_kernel::rl::observe_policy_v5(&state, &surface, actor, 0, 0, 0, 1)
        .unwrap_or_else(|error| panic!("schema 5 observation: {error:?}"));
    serde_json::to_value(&observation).unwrap()
}

/// The error a non-schema-5 policy observation gives at the current decision.
fn legacy_refusal(state: &GameState) -> String {
    let mut state = state.clone();
    state.public_shroud_v1 = false;
    let mut surface = PolicySurfaceV5::new();
    let decision = surface.next_decision(&mut state).unwrap();
    let actor = decision.actor(&state).unwrap();
    format!(
        "{:?}",
        mtg_kernel::rl::observe_policy_v5(&state, &surface, actor, 0, 0, 0, 1)
            .expect_err("the legacy observation must refuse a phased-out permanent")
    )
}

/// `object`'s public reference by its battlefield incarnation.
fn battlefield_ref(state: &GameState, object: ObjectId) -> Value {
    let live = state.objects.get(object);
    let seat = |player: PlayerId| if player == P0 { "p0" } else { "p1" };
    json!({"arena_id": object.0, "card_db_id": live.card_def,
        "controller": seat(live.controller), "owner": seat(live.owner),
        "zone": "Battlefield", "zone_change_count": live.zone_change_count})
}

fn phased_out_record(observation: &Value) -> &Vec<Value> {
    observation["projection"]["phased_out"]
        .as_array()
        .unwrap_or_else(|| panic!("no phased_out record: {}", observation["projection"]))
}

fn relations(observation: &Value) -> &Vec<Value> {
    observation["projection"]["object_relations"]
        .as_array()
        .unwrap()
}

/// Census stop `ability source contract is structurally malformed`: Walking
/// Ballista's "{4}: put a +1/+1 counter on this creature" waits while Talon
/// Gates phases Ballista out. The ability resolves without its source
/// (113.7a), which nothing can affect while phased out (702.26b), and
/// Ballista phases in with its counters and incarnation unchanged.
#[test]
fn ability_whose_source_phased_out_resolves_without_it() {
    let mut state = game(true);
    let ballista = put(&mut state, P0, "Walking Ballista", Zone::Battlefield);
    state.objects.get_mut(ballista).counters.plus1_plus1 = 1;
    let incarnation = state.objects.get(ballista).zone_change_count;
    pool(&mut state, P0, &[], 4);
    act(&mut state, Action::ActivateAbility(ballista, 0));
    phase_out_with_gates(&mut state, P0, ballista);
    assert_eq!(state.stack.len(), 1, "Ballista's ability still waits");
    assert!(!state.players[0].battlefield.contains(&ballista));

    let observation = public_view(&state);
    let record = phased_out_record(&observation);
    assert_eq!(record.len(), 1);
    assert_eq!(
        record[0]["card"]["stable"],
        battlefield_ref(&state, ballista)
    );
    assert_eq!(record[0]["card"]["counters"]["plus1_plus1"], 1);
    assert!(record[0].get("phased_out_with").is_none());
    let stack = observation["projection"]["stack"].as_array().unwrap();
    assert_eq!(stack[0]["source"], battlefield_ref(&state, ballista));
    assert!(legacy_refusal(&state).contains("phased-out permanents require"));

    settle(&mut state);
    assert_eq!(
        state.objects.get(ballista).counters.plus1_plus1,
        1,
        "no counter"
    );
    assert!(public_view(&state)["projection"]["phased_out"].is_array());

    to_main_of(&mut state, P1);
    assert!(
        phased_out(&state, ballista),
        "phases in on its controller's untap"
    );
    to_main_of(&mut state, P0);
    assert!(state.players[0].battlefield.contains(&ballista));
    assert_eq!(state.objects.get(ballista).zone, Zone::Battlefield);
    assert_eq!(state.objects.get(ballista).zone_change_count, incarnation);
    assert_eq!(state.objects.get(ballista).counters.plus1_plus1, 1);
    assert!(public_view(&state)["projection"]
        .get("phased_out")
        .is_none());
}

/// Census stop `stack target contract is structurally malformed`: P1's
/// Lightning Bolt targets P0's creature and P0 phases the creature out in
/// response. The Bolt's only target is illegal while phased out (702.26b):
/// it does nothing, and the creature phases in undamaged.
#[test]
fn spell_whose_target_phased_out_does_nothing() {
    let mut state = game(true);
    let ballista = put(&mut state, P0, "Walking Ballista", Zone::Battlefield);
    state.objects.get_mut(ballista).counters.plus1_plus1 = 2;
    let bolt = put(&mut state, P1, "Lightning Bolt", Zone::Hand);
    run_until(&mut state, priority_for(P1));
    pool(&mut state, P1, &[ManaColor::R], 0);
    act(&mut state, Action::CastSpell(bolt));
    match next(&mut state) {
        Decision::ChooseTargets { .. } => {
            act(&mut state, Action::ChooseTarget(Target::Object(ballista)))
        }
        other => panic!("expected the Bolt's target, got {other:?}"),
    }
    phase_out_with_gates(&mut state, P0, ballista);
    assert_eq!(state.stack.len(), 1, "the Bolt waits");

    let observation = public_view(&state);
    let target = &observation["projection"]["stack"][0]["targets"][0];
    assert_eq!(target["object"], battlefield_ref(&state, ballista));
    assert_eq!(
        phased_out_record(&observation)[0]["card"]["stable"],
        battlefield_ref(&state, ballista)
    );

    settle(&mut state);
    assert_eq!(state.objects.get(bolt).zone, Zone::Graveyard);
    assert_eq!(state.objects.get(ballista).damage, 0);
    assert_eq!(state.players[1].life, 20);
    to_main_of(&mut state, P1);
    to_main_of(&mut state, P0);
    assert_eq!(state.objects.get(ballista).zone, Zone::Battlefield);
    assert_eq!(state.objects.get(ballista).counters.plus1_plus1, 2);
}

/// Census stop `InvalidEffectContinuation`: a creature with an Aura and an
/// Equipment phases out. Both phase out with it, indirectly, still attached
/// (702.26g); they do not phase in on their own, and they phase in with it,
/// attached. Schema 5 lists all three, the attachments with the creature
/// they phased out with, and keeps their attachment relations.
#[test]
fn auras_and_equipment_phase_out_and_in_with_their_creature() {
    let mut state = game(true);
    let raker = put(&mut state, P0, "Glaring Fleshraker", Zone::Battlefield);
    let spear = put(&mut state, P0, "Shadowspear", Zone::Battlefield);
    let aura = put(&mut state, P0, "Lifelink", Zone::Battlefield);
    attach(&mut state, spear, raker);
    attach(&mut state, aura, raker);
    state.validate_attachment_relations().unwrap();
    let power = engine::effective_power(&state, raker);

    phase_out_with_gates(&mut state, P0, raker);
    for object in [raker, spear, aura] {
        assert!(phased_out(&state, object), "{object:?}");
        assert!(!state.players[0].battlefield.contains(&object));
    }
    assert_eq!(state.objects.get(raker).attachments, vec![spear, aura]);
    state.validate_attachment_relations().unwrap();

    let observation = public_view(&state);
    let record = phased_out_record(&observation);
    assert_eq!(record.len(), 3);
    assert_eq!(record[0]["card"]["stable"], battlefield_ref(&state, raker));
    for (entry, attachment) in record[1..].iter().zip([spear, aura]) {
        assert_eq!(entry["card"]["stable"], battlefield_ref(&state, attachment));
        assert_eq!(entry["phased_out_with"], battlefield_ref(&state, raker));
        assert!(relations(&observation).contains(&json!({
            "relation_kind": "attached_to",
            "object": battlefield_ref(&state, attachment),
            "attached_to": battlefield_ref(&state, raker),
        })));
    }

    settle(&mut state);
    to_main_of(&mut state, P1);
    for object in [raker, spear, aura] {
        assert!(
            phased_out(&state, object),
            "P1's untap step changes nothing"
        );
    }
    to_main_of(&mut state, P0);
    for object in [raker, spear, aura] {
        assert_eq!(state.objects.get(object).zone, Zone::Battlefield);
        assert!(state.players[0].battlefield.contains(&object));
    }
    assert_eq!(state.objects.get(raker).attachments, vec![spear, aura]);
    state.validate_attachment_relations().unwrap();
    assert_eq!(engine::effective_power(&state, raker), power);
}

/// Census stop `live linked-exile source lost its public object relation`:
/// Deep-Cavern Bat holds a card in linked exile and phases out. It has not
/// left the battlefield (702.26d): the card stays exiled, linked to the Bat,
/// and schema 5 shows that link to the phased-out Bat. Once the Bat phases
/// in and then leaves, the card returns.
#[test]
fn deep_cavern_bat_keeps_its_linked_exile_while_phased_out() {
    let mut state = game(true);
    let held = put(&mut state, P1, "Lightning Bolt", Zone::Hand);
    let bat = put(&mut state, P0, "Deep-Cavern Bat", Zone::Hand);
    pool(&mut state, P0, &[ManaColor::B], 1);
    act(&mut state, Action::CastSpell(bat));
    run_until(&mut state, |state, decision| match decision {
        Decision::ChooseTargets { .. } | Decision::ChooseEffectTargets { .. } => true,
        Decision::CastSpellOrPass { .. } => {
            state.stack.is_empty() && state.objects.get(bat).zone == Zone::Battlefield
        }
        _ => false,
    });
    choose_bat_exile(&mut state, held);
    settle(&mut state);
    assert_eq!(state.objects.get(held).zone, Zone::Exile);

    phase_out_with_gates(&mut state, P0, bat);
    settle(&mut state);
    assert_eq!(state.objects.get(held).zone, Zone::Exile, "still exiled");
    let observation = public_view(&state);
    assert!(
        relations(&observation).iter().any(|relation| {
            relation["relation_kind"] == "exiled_by"
                && relation["object"]["arena_id"] == held.0
                && relation["exiled_by"] == battlefield_ref(&state, bat)
        }),
        "{:#?}",
        relations(&observation)
    );

    to_main_of(&mut state, P1);
    to_main_of(&mut state, P0);
    assert_eq!(state.objects.get(bat).zone, Zone::Battlefield);
    assert_eq!(state.objects.get(held).zone, Zone::Exile);
    public_view(&state);
    propose_and_commit(&mut state, ProposedEvent::zone_change(bat, Zone::Graveyard));
    settle(&mut state);
    assert_eq!(state.objects.get(held).zone, Zone::Hand, "returned");
}

/// Answers the Bat's entry trigger: P1 as its target, then `card` to exile.
fn choose_bat_exile(state: &mut GameState, card: ObjectId) {
    for _ in 0..60 {
        match next(state) {
            Decision::ChooseTargets { .. } => act(state, Action::ChooseTarget(Target::Player(P1))),
            Decision::ChooseEffectTargets { legal_targets, .. } => {
                assert!(legal_targets.contains(&Target::Object(card)));
                act(state, Action::ChooseEffectTarget(Target::Object(card)));
                return;
            }
            decision => act(state, routine(decision)),
        }
    }
    panic!("the Bat never offered its exile");
}

/// Review finding: the Bat phases out while its entry trigger waits. An
/// "until this leaves the battlefield" exile whose source has only phased
/// out still exiles, linked to it (610.3c does not apply: it has not left),
/// and the card returns when the Bat later leaves.
#[test]
fn deep_cavern_bat_entry_trigger_after_it_phased_out_exiles_with_its_link() {
    let mut state = game(true);
    let held = put(&mut state, P1, "Lightning Bolt", Zone::Hand);
    let bat = put(&mut state, P0, "Deep-Cavern Bat", Zone::Hand);
    pool(&mut state, P0, &[ManaColor::B], 1);
    act(&mut state, Action::CastSpell(bat));
    run_until(&mut state, |state, decision| match decision {
        Decision::ChooseTargets { .. } => true,
        _ => state.objects.get(bat).zone == Zone::Battlefield && !state.stack.is_empty(),
    });
    if let Decision::ChooseTargets { .. } = next(&mut state) {
        act(&mut state, Action::ChooseTarget(Target::Player(P1)));
    }
    run_until(&mut state, priority_for(P0));
    assert_eq!(state.stack.len(), 1, "the Bat's entry trigger waits");
    phase_out_with_gates(&mut state, P0, bat);
    public_view(&state);
    choose_bat_exile(&mut state, held);
    settle(&mut state);
    assert_eq!(state.objects.get(held).zone, Zone::Exile);
    assert_eq!(
        state.objects.get(held).v4.exiled_by,
        Some(ObjectLinkV4 {
            object: bat,
            zone_change_count: state.objects.get(bat).zone_change_count,
        })
    );
    public_view(&state);

    to_main_of(&mut state, P1);
    to_main_of(&mut state, P0);
    propose_and_commit(&mut state, ProposedEvent::zone_change(bat, Zone::Graveyard));
    settle(&mut state);
    assert_eq!(state.objects.get(held).zone, Zone::Hand, "returned");
}

/// Review finding: damage from a source that phased out uses its last known
/// information (608.2h). Walking Ballista equipped with Shadowspear (lifelink)
/// pings P1 and phases out, with the Shadowspear, before the ping resolves:
/// the damage has lifelink.
#[test]
fn damage_from_a_phased_out_source_has_its_last_known_lifelink() {
    let mut state = game(true);
    let ballista = put(&mut state, P0, "Walking Ballista", Zone::Battlefield);
    state.objects.get_mut(ballista).counters.plus1_plus1 = 2;
    let spear = put(&mut state, P0, "Shadowspear", Zone::Battlefield);
    attach(&mut state, spear, ballista);
    act(&mut state, Action::ActivateAbility(ballista, 1));
    match next(&mut state) {
        Decision::ChooseTargets { legal_targets, .. } => {
            assert!(legal_targets.contains(&Target::Player(P1)));
            act(&mut state, Action::ChooseTarget(Target::Player(P1)));
        }
        other => panic!("expected the ping's target, got {other:?}"),
    }
    phase_out_with_gates(&mut state, P0, ballista);
    assert!(phased_out(&state, spear));
    public_view(&state);
    settle(&mut state);
    assert_eq!(state.players[1].life, 19);
    assert_eq!(state.players[0].life, 21, "lifelink from the Shadowspear");
}

/// Review finding: a phased-out permanent is in the schema 5 observation, so
/// two states that differ only in its counters read differently.
#[test]
fn a_phased_out_permanents_counters_are_public() {
    let mut state = game(true);
    let ballista = put(&mut state, P0, "Walking Ballista", Zone::Battlefield);
    state.objects.get_mut(ballista).counters.plus1_plus1 = 1;
    phase_out_with_gates(&mut state, P0, ballista);
    settle(&mut state);
    let mut other = state.clone();
    other.objects.get_mut(ballista).counters.plus1_plus1 = 3;
    let (one, three) = (public_view(&state), public_view(&other));
    assert_ne!(
        one["visible_projection_hash"],
        three["visible_projection_hash"]
    );
    assert_eq!(
        phased_out_record(&three)[0]["card"]["counters"]["plus1_plus1"],
        3
    );
    let mut stripped = [one, three];
    for observation in &mut stripped {
        observation["projection"]["phased_out"] = Value::Null;
        observation["visible_projection_hash"] = Value::Null;
    }
    assert_eq!(
        stripped[0], stripped[1],
        "the record is the only difference"
    );
}

/// Review finding: Shifting Woodland phased out while it is a copy of a
/// creature card still stops being a copy at the cleanup step (its "until end
/// of turn" ends, 702.26e), so it phases in as Shifting Woodland.
#[test]
fn shifting_woodland_copy_ends_at_cleanup_while_phased_out() {
    let mut state = game(true);
    let woodland = put(&mut state, P0, "Shifting Woodland", Zone::Battlefield);
    let raker = put(&mut state, P0, "Glaring Fleshraker", Zone::Graveyard);
    for card in ["Forest", "Lightning Bolt", "Sol Ring", "Demonic Tutor"] {
        put(&mut state, P0, card, Zone::Graveyard);
    }
    pool(&mut state, P0, &[ManaColor::G, ManaColor::G], 2);
    act(&mut state, Action::ActivateAbility(woodland, 0));
    match next(&mut state) {
        Decision::ChooseTargets { legal_targets, .. } => {
            assert!(legal_targets.contains(&Target::Object(raker)));
            act(&mut state, Action::ChooseTarget(Target::Object(raker)));
        }
        other => panic!("expected Woodland's target, got {other:?}"),
    }
    settle(&mut state);
    assert_eq!(
        state.objects.get(woodland).card_def,
        id("Glaring Fleshraker")
    );
    phase_out_with_gates(&mut state, P0, woodland);
    settle(&mut state);
    to_main_of(&mut state, P1);
    assert!(phased_out(&state, woodland));
    assert_eq!(
        state.objects.get(woodland).card_def,
        id("Shifting Woodland")
    );
    to_main_of(&mut state, P0);
    assert_eq!(state.objects.get(woodland).zone, Zone::Battlefield);
    assert_eq!(
        state.objects.get(woodland).card_def,
        id("Shifting Woodland")
    );
}

/// Census stop `InvalidEffectContinuation` with Animate Dead: the reanimated
/// creature phases out and takes Animate Dead with it (702.26g). Both phase
/// in together, and Animate Dead's link to the creature survived: when Animate
/// Dead later leaves, the creature is sacrificed.
#[test]
fn animate_dead_phases_out_with_its_creature_and_keeps_its_link() {
    let mut state = game(true);
    let corpse = put(&mut state, P0, "Glaring Fleshraker", Zone::Graveyard);
    let animate = put(&mut state, P0, "Animate Dead", Zone::Hand);
    pool(&mut state, P0, &[ManaColor::B], 1);
    act(&mut state, Action::CastSpell(animate));
    for _ in 0..60 {
        match next(&mut state) {
            Decision::ChooseTargets { legal_targets, .. } => {
                assert!(legal_targets.contains(&Target::Object(corpse)));
                act(&mut state, Action::ChooseTarget(Target::Object(corpse)));
            }
            Decision::CastSpellOrPass { .. }
                if state.stack.is_empty()
                    && state.engine.pending_triggers.is_empty()
                    && state.objects.get(corpse).zone == Zone::Battlefield =>
            {
                break
            }
            decision => act(&mut state, routine(decision)),
        }
    }
    assert_eq!(state.objects.get(corpse).zone, Zone::Battlefield);
    assert_eq!(state.objects.get(corpse).attachments, vec![animate]);

    phase_out_with_gates(&mut state, P0, corpse);
    assert!(phased_out(&state, animate), "phased out indirectly");
    settle(&mut state);
    let observation = public_view(&state);
    let record = phased_out_record(&observation);
    assert_eq!(record.len(), 2);
    assert_eq!(
        record[1]["phased_out_with"],
        battlefield_ref(&state, corpse)
    );

    to_main_of(&mut state, P1);
    to_main_of(&mut state, P0);
    assert_eq!(state.objects.get(corpse).zone, Zone::Battlefield);
    assert_eq!(state.objects.get(animate).zone, Zone::Battlefield);
    assert_eq!(state.objects.get(corpse).attachments, vec![animate]);
    propose_and_commit(
        &mut state,
        ProposedEvent::zone_change(animate, Zone::Graveyard),
    );
    settle(&mut state);
    assert_eq!(
        state.objects.get(corpse).zone,
        Zone::Graveyard,
        "sacrificed when Animate Dead leaves"
    );
}

/// Review finding: an ability whose source phased out halted when it read
/// its source (taking the initiative). White Plume Adventurer phases out
/// while its entry trigger waits; the trigger still resolves (113.7a) and its
/// controller takes the initiative.
#[test]
fn taking_the_initiative_from_a_phased_out_source() {
    let mut state = game(true);
    let plume = put(&mut state, P0, "White Plume Adventurer", Zone::Hand);
    pool(&mut state, P0, &[ManaColor::W], 2);
    act(&mut state, Action::CastSpell(plume));
    run_until(&mut state, |state, _| {
        state.objects.get(plume).zone == Zone::Battlefield && !state.stack.is_empty()
    });
    phase_out_with_gates(&mut state, P0, plume);
    assert_eq!(state.stack.len(), 1, "the entry trigger waits");
    public_view(&state);
    settle(&mut state);
    assert_eq!(state.initiative, Some(P0));
    to_main_of(&mut state, P1);
    to_main_of(&mut state, P0);
    assert_eq!(state.objects.get(plume).zone, Zone::Battlefield);
}
