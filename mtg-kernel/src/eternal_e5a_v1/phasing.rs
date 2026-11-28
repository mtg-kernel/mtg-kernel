//! Phasing (702.26) for Talon Gates of Madara ("up to one target creature
//! phases out") and Vision Charm ("target artifact phases out").
//!
//! A phased-out permanent is treated as though it doesn't exist (702.26b).
//! It leaves every zone list and takes the command-zone marker without a zone
//! change (702.26d), so it keeps its incarnation (`zone_change_count`),
//! counters, damage, tapped state, summoning sickness and controller; nothing
//! sees it as a battlefield permanent, and effects that remember it by
//! incarnation apply again once it returns. During its controller's next
//! untap step, before permanents untap, it phases in under that player's
//! control (502.1). Phasing in or out is not a zone change, so no enter or
//! leave abilities trigger.
//!
//! Auras and Equipment attached to a permanent that phases out phase out with
//! it, indirectly, and stay attached to it (702.26g). They never phase in on
//! their own: they phase in along with that permanent, still attached, under
//! the control of whoever controlled them. A permanent that phases out
//! directly while attached to something (Vision Charm on an Equipment) is no
//! longer attached while phased out; it phases in attached to that object if
//! it is still the same battlefield permanent, and unattached otherwise
//! (702.26i).

use super::{e5a_state, live_link, read_state};
use crate::ids::{ObjectId, PlayerId};
use crate::state::{GameState, ObjectLinkV4, Zone};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// One phased-out permanent and the player who controlled it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhasedOutV1 {
    pub object: ObjectLinkV4,
    pub controller: PlayerId,
    /// Phased out indirectly (702.26g): the permanent it is attached to and
    /// phases in with. `None` for a permanent that phased out directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub with: Option<ObjectLinkV4>,
    /// Phased out directly while attached (702.26i): the permanent it was
    /// attached to, to which it phases in attached if that permanent is still
    /// the same battlefield incarnation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attached_to: Option<ObjectLinkV4>,
    /// Last known information (608.2h): it had lifelink just before it
    /// phased out, so damage it deals while phased out (a resolving ability
    /// of it) has lifelink.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub lifelink: bool,
}

// A record without attachment facts hashes exactly as before they existed.
impl Hash for PhasedOutV1 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.object.hash(state);
        self.controller.hash(state);
        if let Some(with) = self.with {
            b"phased-out-indirectly/v1".hash(state);
            with.hash(state);
        }
        if let Some(attached_to) = self.attached_to {
            b"phased-out-attached-to/v1".hash(state);
            attached_to.hash(state);
        }
        if self.lifelink {
            b"phased-out-lifelink/v1".hash(state);
        }
    }
}

impl PhasedOutV1 {
    /// Whether this record names the live object at its recorded incarnation,
    /// still in the command-zone marker.
    fn holds(&self, state: &GameState) -> bool {
        state
            .objects
            .try_get(self.object.object)
            .is_some_and(|live| {
                live.zone == Zone::Command
                    && live.zone_change_count == self.object.zone_change_count
            })
    }
}

/// `object` phases out directly, and the permanents attached to it, and to
/// those, phase out indirectly with it.
pub(crate) fn phase_out(state: &mut GameState, object: ObjectId) {
    if state.objects.get(object).zone != Zone::Battlefield || phased_out(state, object) {
        return;
    }
    // 702.26i: while phased out it is attached to nothing.
    let attached_to = state.objects.get(object).v4.attached_to;
    if let Some(host) = attached_to {
        if state.objects.try_get(host.object).is_some() {
            state
                .objects
                .get_mut(host.object)
                .attachments
                .retain(|&attached| attached != object);
        }
        state.objects.get_mut(object).v4.attached_to = None;
    }
    phase_out_one(state, object, None, attached_to);
    // 702.26g, 702.26h: everything attached phases out indirectly only.
    let mut hosts = vec![object];
    while let Some(host) = hosts.pop() {
        let link = live_link(state, host);
        let attached = state.objects.get(host).attachments.clone();
        for id in attached {
            if state.objects.get(id).zone == Zone::Battlefield {
                phase_out_one(state, id, Some(link), None);
                hosts.push(id);
            }
        }
    }
}

fn phase_out_one(
    state: &mut GameState,
    id: ObjectId,
    with: Option<ObjectLinkV4>,
    attached_to: Option<ObjectLinkV4>,
) {
    let live = state.objects.get(id);
    if live.zone != Zone::Battlefield {
        return;
    }
    let controller = live.controller;
    let link = live_link(state, id);
    // Read while it and its attachments are still on the battlefield.
    let lifelink =
        crate::engine::has_effective_keyword(state, id, crate::card_def::Keywords::LIFELINK);
    for seat in [PlayerId::P0, PlayerId::P1] {
        state.players[seat.index()]
            .battlefield
            .retain(|&object| object != id);
    }
    state.engine.combat.remove_from_combat(id);
    state.objects.get_mut(id).zone = Zone::Command;
    e5a_state(state).phased_out.push(PhasedOutV1 {
        object: link,
        controller,
        with,
        attached_to,
        lifelink,
    });
}

/// Whether `object` is phased out.
pub(crate) fn phased_out(state: &GameState, object: ObjectId) -> bool {
    read_state(state).is_some_and(|e5a| {
        e5a.phased_out
            .iter()
            .any(|phased| phased.object.object == object && phased.holds(state))
    })
}

/// For a phased-out `object`, whether it had lifelink when it phased out:
/// damage it deals now uses that last known information (608.2h). `None`
/// for a permanent that is not phased out.
pub(crate) fn last_known_lifelink(state: &GameState, object: ObjectId) -> Option<bool> {
    read_state(state)?
        .phased_out
        .iter()
        .find(|phased| phased.object.object == object && phased.holds(state))
        .map(|phased| phased.lifelink)
}

/// Whether any permanent is phased out.
pub(crate) fn any(state: &GameState) -> bool {
    read_state(state).is_some_and(|e5a| e5a.phased_out.iter().any(|phased| phased.holds(state)))
}

/// The phased-out permanents still holding their incarnation, in the order
/// they phased out.
pub(crate) fn records(state: &GameState) -> Vec<PhasedOutV1> {
    read_state(state).map_or_else(Vec::new, |e5a| {
        e5a.phased_out
            .iter()
            .copied()
            .filter(|phased| phased.holds(state))
            .collect()
    })
}

/// `player`'s untap step: the permanents that phased out directly under
/// their control phase in, each with the permanents that phased out
/// indirectly with it.
pub(crate) fn phase_in(state: &mut GameState, player: PlayerId) {
    let Some(eternal) = state.eternal_v1.as_mut() else {
        return;
    };
    if eternal.e5a.phased_out.is_empty() {
        return;
    }
    let (mine, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut eternal.e5a.phased_out)
        .into_iter()
        .partition(|phased| phased.with.is_none() && phased.controller == player);
    eternal.e5a.phased_out = rest;
    for phased in mine {
        let returned = phase_in_one(state, phased, player);
        if returned {
            if let Some(host) = phased.attached_to {
                reattach(state, phased.object, host);
            }
        }
        phase_in_attached(state, phased.object, returned);
    }
    // A Shapeshifter that phases in reads its controller's graveyard again.
    let _ = crate::full_text_v1::donor_refusal_at(
        state,
        crate::full_text_v1::DonorCheckSiteV1::PhaseIn,
    );
}

/// Returns `phased` to `controller`'s battlefield if it still holds its
/// incarnation.
fn phase_in_one(state: &mut GameState, phased: PhasedOutV1, controller: PlayerId) -> bool {
    if !phased.holds(state) {
        return false;
    }
    let id = phased.object.object;
    let live = state.objects.get_mut(id);
    live.zone = Zone::Battlefield;
    live.controller = controller;
    state.players[controller.index()].battlefield.push(id);
    true
}

/// The permanents that phased out indirectly with `host` phase in with it,
/// still attached (702.26g), each under its own controller. If `host` did not
/// return they phase in unattached.
fn phase_in_attached(state: &mut GameState, host: ObjectLinkV4, host_returned: bool) {
    let mut hosts = vec![(host, host_returned)];
    while let Some((host, host_returned)) = hosts.pop() {
        let Some(eternal) = state.eternal_v1.as_mut() else {
            return;
        };
        let (with_host, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut eternal.e5a.phased_out)
            .into_iter()
            .partition(|phased| phased.with == Some(host));
        eternal.e5a.phased_out = rest;
        for phased in with_host {
            let returned = phase_in_one(state, phased, phased.controller);
            if returned && !host_returned {
                detach(state, phased.object.object);
            }
            hosts.push((phased.object, returned));
        }
    }
}

/// Restores the attachment a directly phased-out attachment had, if its
/// host is still that battlefield incarnation (702.26i). State-based actions
/// then apply as usual.
fn reattach(state: &mut GameState, attachment: ObjectLinkV4, host: ObjectLinkV4) {
    let host_live = state.objects.try_get(host.object).is_some_and(|live| {
        live.zone == Zone::Battlefield && live.zone_change_count == host.zone_change_count
    });
    if !host_live || host.object == attachment.object {
        return;
    }
    state
        .objects
        .get_mut(host.object)
        .attachments
        .push(attachment.object);
    state.objects.get_mut(attachment.object).v4.attached_to = Some(host);
}

/// Clears `object`'s attachment to a host that did not phase in with it.
fn detach(state: &mut GameState, object: ObjectId) {
    let Some(host) = state.objects.get_mut(object).v4.attached_to.take() else {
        return;
    };
    if state.objects.try_get(host.object).is_some() {
        state
            .objects
            .get_mut(host.object)
            .attachments
            .retain(|&attached| attached != object);
    }
}

#[cfg(all(test, feature = "eternal-fixtures"))]
mod tests {
    use super::*;
    use crate::card_def::{card_id_by_name, CARD_DEFS};
    use crate::state::{GameObject, ObjectStateV4};

    fn game() -> GameState {
        let ids = vec![card_id_by_name("Island").unwrap(); 10];
        GameState::new_from_libraries(
            &ids,
            &ids,
            |id| CARD_DEFS[id as usize].object_name.into(),
            0x5048_4153_4532,
        )
    }

    fn put(state: &mut GameState, player: PlayerId, name: &str) -> ObjectId {
        let card_def = card_id_by_name(name).unwrap();
        let object = state.objects.push(GameObject {
            card_def,
            name: CARD_DEFS[card_def as usize].object_name.into(),
            owner: player,
            controller: player,
            zone: Zone::Battlefield,
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
        state.players[player.index()].battlefield.push(object);
        object
    }

    fn attach(state: &mut GameState, attachment: ObjectId, host: ObjectId) {
        let link = live_link(state, host);
        state.objects.get_mut(host).attachments.push(attachment);
        state.objects.get_mut(attachment).v4.attached_to = Some(link);
    }

    fn on_battlefield(state: &GameState, player: PlayerId, object: ObjectId) -> bool {
        state.objects.get(object).zone == Zone::Battlefield
            && state.players[player.index()].battlefield.contains(&object)
    }

    /// 702.26g: an opponent's Aura and an Equipment on a creature phase out
    /// with it, stay attached, and phase in only with it, each under its own
    /// controller; an Aura on that Equipment follows too.
    #[test]
    fn attachments_phase_out_indirectly_and_phase_in_with_their_host() {
        let mut state = game();
        let creature = put(&mut state, PlayerId::P0, "Glaring Fleshraker");
        let spear = put(&mut state, PlayerId::P0, "Shadowspear");
        let aura = put(&mut state, PlayerId::P1, "Lifelink");
        attach(&mut state, spear, creature);
        attach(&mut state, aura, creature);
        let incarnations =
            [creature, spear, aura].map(|id| state.objects.get(id).zone_change_count);

        phase_out(&mut state, creature);
        for id in [creature, spear, aura] {
            assert!(phased_out(&state, id));
        }
        assert_eq!(state.objects.get(creature).attachments, vec![spear, aura]);
        state.validate_attachment_relations().unwrap();
        let rows = records(&state);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].with, None);
        assert_eq!(rows[1].with, Some(live_link(&state, creature)));
        assert_eq!(rows[2].with, Some(live_link(&state, creature)));

        // The Aura's controller's untap step does not phase it in.
        phase_in(&mut state, PlayerId::P1);
        assert!(phased_out(&state, aura));
        state.validate_attachment_relations().unwrap();

        phase_in(&mut state, PlayerId::P0);
        assert!(on_battlefield(&state, PlayerId::P0, creature));
        assert!(on_battlefield(&state, PlayerId::P0, spear));
        assert!(on_battlefield(&state, PlayerId::P1, aura));
        assert_eq!(state.objects.get(creature).attachments, vec![spear, aura]);
        state.validate_attachment_relations().unwrap();
        assert!(records(&state).is_empty());
        assert_eq!(
            [creature, spear, aura].map(|id| state.objects.get(id).zone_change_count),
            incarnations
        );
    }

    /// 702.26i: an Equipment that phases out directly is attached to nothing
    /// while phased out, and phases in attached to the same creature, or
    /// unattached once that creature is gone.
    #[test]
    fn a_directly_phased_out_attachment_phases_in_attached_or_unattached() {
        for creature_left in [false, true] {
            let mut state = game();
            let creature = put(&mut state, PlayerId::P0, "Glaring Fleshraker");
            let spear = put(&mut state, PlayerId::P0, "Shadowspear");
            attach(&mut state, spear, creature);

            phase_out(&mut state, spear);
            assert!(phased_out(&state, spear));
            assert!(!phased_out(&state, creature));
            assert!(state.objects.get(creature).attachments.is_empty());
            assert_eq!(state.objects.get(spear).v4.attached_to, None);
            state.validate_attachment_relations().unwrap();
            assert_eq!(
                records(&state)[0].attached_to,
                Some(live_link(&state, creature))
            );

            if creature_left {
                state.players[0].battlefield.retain(|&id| id != creature);
                let live = state.objects.get_mut(creature);
                live.zone = Zone::Graveyard;
                live.zone_change_count += 1;
                state.players[0].graveyard.push(creature);
            }
            phase_in(&mut state, PlayerId::P0);
            assert!(on_battlefield(&state, PlayerId::P0, spear));
            state.validate_attachment_relations().unwrap();
            if creature_left {
                assert_eq!(state.objects.get(spear).v4.attached_to, None);
            } else {
                assert_eq!(
                    state.objects.get(spear).v4.attached_to,
                    Some(live_link(&state, creature))
                );
                assert_eq!(state.objects.get(creature).attachments, vec![spear]);
            }
        }
    }

    /// A half-phased attachment pair (one end phased out, the other not) is
    /// malformed.
    #[test]
    fn an_attachment_split_across_phasing_is_rejected() {
        let mut state = game();
        let creature = put(&mut state, PlayerId::P0, "Glaring Fleshraker");
        let spear = put(&mut state, PlayerId::P0, "Shadowspear");
        attach(&mut state, spear, creature);
        phase_out(&mut state, creature);
        let mut split = state.clone();
        split.objects.get_mut(spear).zone = Zone::Battlefield;
        split.players[0].battlefield.push(spear);
        split
            .eternal_v1
            .as_mut()
            .unwrap()
            .e5a
            .phased_out
            .retain(|phased| phased.object.object != spear);
        assert!(split.validate_attachment_relations().is_err());
        state.validate_attachment_relations().unwrap();
    }
}
