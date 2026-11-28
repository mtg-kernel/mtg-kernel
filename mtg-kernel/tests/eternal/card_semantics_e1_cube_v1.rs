//! M2 T8: arm E (`card_semantics_e1`) on the full Vintage Cube table: the
//! declared table and scales pins, zero missing rows over the active cards,
//! the fetch family, the column map and the `PilotPin.table_sha256` binding.

use mtg_kernel::card_def::{CardType, CARD_DEFS, KERNEL_CARDDB_HASH};
use mtg_kernel::card_semantics_e1_cube_v1::*;
use mtg_kernel::cube_draft_v1::{basic_land_ids, land_fetch_bits};
use mtg_kernel::cube_pilot_adapter_v1::{verify_pilot_pin_match, M3_SAMPLE_TEMP1_WIDE_V1};
use mtg_kernel::cube_training_driver_v1::PilotPin;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const CUBE_LIST: &str = include_str!("../../../data/cube_v1/mtgo_vintage_cube_20261007.json");
/// The signed active fence (`fence_v3.json`, A3 amendment 2): 15 cards.
const FENCE_V3: &str = include_str!("../data/card_semantics_e1_cube_v1/fence_v3.json");
const FENCE_V3_SHA256: &str = "8d02e845f1be12d636d4e6b3f1347b4d42f002e6c7ce67ce339114bcf6775e5c";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate sits in the repository root")
        .to_path_buf()
}

fn table_bytes() -> Vec<u8> {
    std::fs::read(repo_root().join(TABLE_PATH_V1)).expect("committed cube table")
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn active() -> BTreeSet<u16> {
    active_ids_v1(CUBE_LIST, FENCE_V3).expect("active ids")
}

#[test]
fn fence_fixture_is_the_signed_fence_v3() {
    assert_eq!(sha256(FENCE_V3.as_bytes()), FENCE_V3_SHA256);
}

#[test]
fn active_cards_are_525_non_basic_plus_5_basics() {
    let ids = active();
    let basics = basic_land_ids().unwrap();
    assert_eq!(ids.len(), 530);
    assert!(basics.iter().all(|id| ids.contains(id)));
    assert_eq!(ids.iter().filter(|id| !basics.contains(id)).count(), 525);
}

#[test]
fn committed_table_is_the_declared_table() {
    assert_eq!(sha256(&table_bytes()), TABLE_SHA256_V1);
}

#[test]
fn arm_e_reader_loads_the_declared_cube_table() {
    let config = declared_config_v1(&repo_root());
    let binding = reader_binding_v1(&config).expect("arm E reader accepts the cube table");
    let scales = scales_v1(&table_bytes()).unwrap();
    assert_eq!(binding, scales.binding);
}

#[test]
fn every_active_card_has_a_nonzero_row() {
    let missing = missing_rows_v1(&table_bytes(), &active()).unwrap();
    assert!(
        missing.is_empty(),
        "active cards without a row: {missing:?}"
    );
}

#[test]
fn a_missing_or_zero_row_is_reported() {
    let mut table: serde_json::Value = serde_json::from_slice(&table_bytes()).unwrap();
    let ids = active();
    let cards = table["cards"].as_array_mut().unwrap();
    let id_at =
        |cards: &[serde_json::Value], i: usize| cards[i]["card_id"].as_u64().unwrap() as u16;
    let active_rows: Vec<usize> = (0..cards.len())
        .filter(|&i| ids.contains(&id_at(cards, i)))
        .take(2)
        .collect();
    let (removed, zeroed) = (id_at(cards, active_rows[0]), id_at(cards, active_rows[1]));
    for v in cards[active_rows[1]]["vector"].as_array_mut().unwrap() {
        *v = 0.into();
    }
    cards.remove(active_rows[0]);
    let count = cards.len();
    table["card_count"] = count.into();
    let bytes = serde_json::to_vec(&table).unwrap();
    let mut expected = vec![removed, zeroed];
    expected.sort_unstable();
    assert_eq!(missing_rows_v1(&bytes, &ids).unwrap(), expected);
}

#[test]
fn committed_scales_are_the_readers_divisors_and_binding() {
    let bytes = std::fs::read(repo_root().join(SCALES_PATH_V1)).expect("committed scales");
    assert_eq!(sha256(&bytes), SCALES_SHA256_V1);
    let text = String::from_utf8(bytes).unwrap();
    assert_eq!(text, scales_json_v1(&table_bytes()).unwrap());
    let parsed: ScalesV1 = serde_json::from_str(&text).unwrap();
    assert_eq!(parsed, scales_v1(&table_bytes()).unwrap());
    assert_eq!(parsed.schema, SCALES_SCHEMA_V1);
    assert_eq!(parsed.table_sha256, TABLE_SHA256_V1);
    assert!(parsed.column_divisors.iter().all(|d| d.is_power_of_two()));
    let reader = reader_binding_v1(&declared_config_v1(&repo_root())).unwrap();
    assert_eq!(parsed.binding, reader);
}

#[test]
fn scales_binding_tracks_the_reader_on_a_small_table() {
    let root = std::env::temp_dir().join(format!("cse1-cube-scales-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let table = serde_json::json!({
        "schema": "card_semantics_table/v1",
        "kernel_carddb_hash": format!("{:016x}", KERNEL_CARDDB_HASH),
        "vector_width": 3, "card_count": 2,
        "cards": [{"card_id": 9, "vector": [5, -3, 0]}, {"card_id": 4, "vector": [-1, 1, 0]}]
    });
    let bytes = serde_json::to_vec(&table).unwrap();
    let path = root.join("table.json");
    std::fs::write(&path, &bytes).unwrap();
    let config = CardSemanticsConfigV1 {
        table_path: path,
        table_sha256: sha256(&bytes),
    };
    let reader = reader_binding_v1(&config).unwrap();
    let scales = scales_v1(&bytes).unwrap();
    std::fs::remove_dir_all(&root).unwrap();
    assert_eq!(scales.column_divisors, [8, 4, 1]);
    assert_eq!(scales.card_count, 2);
    assert_eq!(scales.binding, reader);
}

/// Under arm E, a collector's `PilotPin.table_sha256` is the declared table
/// digest, and a pin built from the reader's configuration matches the
/// declared pin on every field.
#[test]
fn collector_pin_table_sha256_equals_the_declared_table_digest() {
    let config = declared_config_v1(&repo_root());
    let table_sha256 = pilot_table_sha256_v1(&config).unwrap();
    assert_eq!(table_sha256, TABLE_SHA256_V1);
    let pin = |table_sha256: String| PilotPin {
        checkpoint_sha256: "11".repeat(32),
        after_state_sha256: "22".repeat(32),
        card_database_sha256: format!("{:016x}", KERNEL_CARDDB_HASH),
        table_sha256,
        training_math: "training-math".into(),
        source_binary_sha256: "44".repeat(32),
        decoding: M3_SAMPLE_TEMP1_WIDE_V1.into(),
    };
    let declared = pin(TABLE_SHA256_V1.to_owned());
    verify_pilot_pin_match(&declared, &pin(table_sha256)).unwrap();
    // A table the reader refuses never yields a pin.
    let stale = CardSemanticsConfigV1 {
        table_sha256: "00".repeat(32),
        ..config
    };
    assert!(pilot_table_sha256_v1(&stale).is_err());
}

/// The fetch family (5 columns) equals the engine's fetchable basic land
/// types on every land row, and no other row carries it.
#[test]
fn fetch_family_matches_the_engine() {
    let table: serde_json::Value = serde_json::from_slice(&table_bytes()).unwrap();
    let tokens: Vec<&str> = table["vocabulary"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["token"].as_str().unwrap())
        .collect();
    let columns: Vec<usize> = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
        .iter()
        .map(|basic| {
            let token = format!("fetch:subtype={basic}");
            tokens.iter().position(|t| *t == token).expect(&token)
        })
        .collect();
    assert_eq!(
        tokens
            .iter()
            .filter(|t| t.starts_with("fetch:subtype="))
            .count(),
        5
    );
    let mut fetchers = 0;
    for row in table["cards"].as_array().unwrap() {
        let id = row["card_id"].as_u64().unwrap() as u16;
        let vector = row["vector"].as_array().unwrap();
        let land = CARD_DEFS[usize::from(id)].types.contains(&CardType::Land);
        let bits = if land { land_fetch_bits(id) } else { 0 };
        for (k, &column) in columns.iter().enumerate() {
            let expected = i64::from((bits >> k) & 1);
            assert_eq!(
                vector[column].as_i64().unwrap(),
                expected,
                "{} fetch column {k}",
                row["name"]
            );
        }
        fetchers += usize::from(bits != 0);
    }
    assert!(fetchers > 0, "the cube has fetchlands");
}

/// The provisional table M4 pins maps column for column into the full
/// table: same token and kind at the same index.
#[test]
fn column_map_keeps_every_provisional_column() {
    let root = repo_root();
    let read = |path: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(root.join(path)).unwrap()).unwrap()
    };
    let provisional = read(PROVISIONAL_TABLE_PATH_V1);
    let full = read(TABLE_PATH_V1);
    let map = read(COLUMN_MAP_PATH_V1);
    assert_eq!(map["old_content_sha256"], provisional["content_sha256"]);
    assert_eq!(map["new_content_sha256"], full["content_sha256"]);
    let old = provisional["vocabulary"].as_array().unwrap();
    let new = full["vocabulary"].as_array().unwrap();
    assert_eq!(map["old_width"], old.len());
    assert_eq!(map["new_width"], new.len());
    let old_to_new = map["old_to_new"].as_array().unwrap();
    assert_eq!(old_to_new.len(), old.len());
    for (i, (column, target)) in old.iter().zip(old_to_new).enumerate() {
        assert_eq!(target.as_u64(), Some(i as u64));
        assert_eq!(column["token"], new[i]["token"]);
        assert_eq!(column["kind"], new[i]["kind"]);
    }
}

/// The scales and the missing-row check refuse every table the arm E reader
/// refuses, so neither can certify a table the trainer would not load.
#[test]
fn scales_and_row_check_refuse_what_the_reader_refuses() {
    let root = std::env::temp_dir().join(format!("cse1-cube-refusals-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let base = serde_json::json!({
        "schema": "card_semantics_table/v1",
        "kernel_carddb_hash": format!("{:016x}", KERNEL_CARDDB_HASH),
        "vector_width": 2, "card_count": 2,
        "cards": [{"card_id": 9, "vector": [5, -3]}, {"card_id": 4, "vector": [-1, 1]}]
    });
    let mutations: [fn(&mut serde_json::Value); 7] = [
        |t| t["schema"] = "card_semantics_table/v2".into(),
        |t| t["kernel_carddb_hash"] = "0000000000000000".into(),
        |t| t["card_count"] = 3.into(),
        |t| t["cards"][1]["card_id"] = 9.into(),
        |t| t["cards"][1]["vector"] = serde_json::json!([1]),
        |t| t["cards"][0]["vector"][0] = 16_777_217.into(),
        |t| t["cards"][0]["card_id"] = 65_536.into(),
    ];
    let ids: BTreeSet<u16> = [4, 9].into();
    for (k, mutate) in mutations.iter().enumerate() {
        let mut table = base.clone();
        mutate(&mut table);
        let bytes = serde_json::to_vec(&table).unwrap();
        let path = root.join(format!("table-{k}.json"));
        std::fs::write(&path, &bytes).unwrap();
        let config = CardSemanticsConfigV1 {
            table_path: path,
            table_sha256: sha256(&bytes),
        };
        assert!(
            reader_binding_v1(&config).is_err(),
            "reader accepts mutation {k}"
        );
        assert!(scales_v1(&bytes).is_err(), "scales accept mutation {k}");
        assert!(
            missing_rows_v1(&bytes, &ids).is_err(),
            "row check accepts mutation {k}"
        );
    }
    let bytes = serde_json::to_vec(&base).unwrap();
    std::fs::remove_dir_all(&root).unwrap();
    assert!(scales_v1(&bytes).is_ok());
    assert_eq!(missing_rows_v1(&bytes, &ids).unwrap(), Vec::<u16>::new());
}

#[test]
fn active_ids_refuse_a_duplicate_or_unlisted_fenced_name() {
    let list = |names: &[&str]| {
        let cards: Vec<_> = names
            .iter()
            .map(|n| serde_json::json!({"name": n}))
            .collect();
        serde_json::json!({"schema": "cube_list/v1", "cards": cards}).to_string()
    };
    let fence = |names: &[&str]| {
        let cards: Vec<_> = names
            .iter()
            .map(|n| serde_json::json!({"name": n, "reason": "test"}))
            .collect();
        serde_json::json!({"cards": cards}).to_string()
    };
    let ok = active_ids_v1(
        &list(&["Lightning Bolt", "Counterspell"]),
        &fence(&["Counterspell"]),
    )
    .unwrap();
    assert_eq!(ok.len(), 1 + 5);
    assert!(active_ids_v1(
        &list(&["Lightning Bolt", "Counterspell", "Counterspell"]),
        &fence(&["Counterspell"])
    )
    .is_err());
    assert!(active_ids_v1(&list(&["Lightning Bolt"]), &fence(&["Counterspell"])).is_err());
}
