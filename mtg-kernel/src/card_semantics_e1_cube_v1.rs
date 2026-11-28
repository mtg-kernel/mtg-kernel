//! M2 task T8: arm E (`card_semantics_e1`, `+cse1`) on the Vintage Cube card
//! table.
//!
//! The trainer's arm E reader ([`CardSemanticsConfigV1`], re-exported here)
//! reads any `card_semantics_table/v1` file by path and SHA-256. This module
//! declares the Cube binding of that reader:
//!
//! * [`TABLE_PATH_V1`] and [`TABLE_SHA256_V1`]: the full Cube table the M3
//!   Gate 1 manifest pins (every Cube list card with a definition plus the
//!   five basics, with the fetch family). This is the value
//!   `PilotPin.table_sha256` binds under arm E; a collector obtains it through
//!   [`pilot_table_sha256_v1`], which loads the table through the reader
//!   first, so a pin can only name a table the reader accepts.
//! * [`SCALES_PATH_V1`] and [`SCALES_SHA256_V1`]: the reader's power-of-two
//!   column divisors and its binding digest for that table, written by the
//!   table generator (`card_semantics_table_v1 --cube --write`) from
//!   [`scales_json_v1`]. The tests prove the divisors reproduce the reader's
//!   binding bit for bit.
//! * [`missing_rows_v1`]: the active cards (Cube list minus the fence, plus
//!   the five basics) that have no row, or only a zero row, in a table. The
//!   arm E reader gives a card without a row a zero row silently, so the
//!   Gate 1 receipt requires this list to be empty.

pub use crate::native_policy_train_step_v1::card_projection_e1::CardSemanticsConfigV1;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;

/// The full Cube table, relative to the repository root.
pub const TABLE_PATH_V1: &str = "data/card_semantics_v1/cube_table_v1_full.json";
/// SHA-256 of the file bytes at [`TABLE_PATH_V1`]: the declared
/// `PilotPin.table_sha256` under arm E.
pub const TABLE_SHA256_V1: &str =
    "25328ecb634570ffcc153d1ff70dd2d02b61c306c36e96acf0f2a23b192ef0cc";
/// The reader's column divisors and binding for [`TABLE_PATH_V1`].
pub const SCALES_PATH_V1: &str = "data/card_semantics_v1/cube_table_v1_full_e1_scales.json";
/// SHA-256 of the file bytes at [`SCALES_PATH_V1`].
pub const SCALES_SHA256_V1: &str =
    "e1dc6f998f6024b683b1f98d70ac78086bf50b8dc8c722a9314819a2ac361569";
/// The provisional cube table M4 D0 to D8 pin until D15; the full table
/// keeps its columns at their indices.
pub const PROVISIONAL_TABLE_PATH_V1: &str = "data/card_semantics_v1/cube_table_v1.json";
/// The provisional to full column map (identity on the provisional columns).
pub const COLUMN_MAP_PATH_V1: &str =
    "data/card_semantics_v1/cube_table_v1_full_column_map_778c12a1.json";
/// Schema tag of the scales document.
pub const SCALES_SCHEMA_V1: &str = "card_semantics_e1_scales/v1";

/// The arm E reader configuration of the declared Cube table under
/// `repo_root`.
pub fn declared_config_v1(repo_root: &Path) -> CardSemanticsConfigV1 {
    CardSemanticsConfigV1 {
        table_path: repo_root.join(TABLE_PATH_V1),
        table_sha256: TABLE_SHA256_V1.to_owned(),
    }
}

/// The `PilotPin.table_sha256` a collector built on `config` must carry
/// under arm E: the table file's SHA-256, returned only after the arm E
/// reader has loaded the table (file digest, schema, card database hash,
/// dimensions and integers checked), so the pin names exactly the table the
/// trainer reads.
pub fn pilot_table_sha256_v1(config: &CardSemanticsConfigV1) -> Result<String, String> {
    config.load_v1()?;
    Ok(config.table_sha256.clone())
}

/// Loads `config` through the arm E reader and returns its binding digest
/// (the value the checkpoint's `card_semantics` section carries).
pub fn reader_binding_v1(config: &CardSemanticsConfigV1) -> Result<String, String> {
    Ok(config.load_v1()?.binding_v1().to_owned())
}

#[derive(Deserialize)]
struct TableRowsV1 {
    schema: String,
    kernel_carddb_hash: String,
    vector_width: usize,
    card_count: usize,
    cards: Vec<TableRowV1>,
}

#[derive(Deserialize)]
struct TableRowV1 {
    card_id: usize,
    vector: Vec<i32>,
}

/// Parses `table_bytes` and applies every check the arm E reader applies
/// (`card_projection_e1`): schema, compiled card database hash, a vector
/// width of 1 to 65,536, a non-empty row list matching `card_count`, every
/// card id below the card vocabulary, no duplicate id, every row of the
/// declared width and every integer exact in f32. Returns the parsed table
/// and the row span (largest id plus one).
fn parse_checked_v1(table_bytes: &[u8]) -> Result<(TableRowsV1, usize), String> {
    let table: TableRowsV1 = serde_json::from_slice(table_bytes).map_err(|e| e.to_string())?;
    if table.schema != "card_semantics_table/v1"
        || table.kernel_carddb_hash != format!("{:016x}", crate::card_def::KERNEL_CARDDB_HASH)
    {
        return Err("card semantics schema or kernel_carddb_hash differs".into());
    }
    let width = table.vector_width;
    if width == 0
        || width > 65_536
        || table.cards.is_empty()
        || table.card_count != table.cards.len()
    {
        return Err("invalid card semantics dimensions".into());
    }
    let span = table
        .cards
        .iter()
        .map(|r| r.card_id)
        .max()
        .expect("non-empty")
        .checked_add(1)
        .filter(|&n| n < crate::native_policy_value_net_v1::CARD_VOCAB_SIZE_V1)
        .ok_or("card semantics card id outside vocabulary")?;
    let mut seen = BTreeSet::new();
    for row in &table.cards {
        if row.vector.len() != width || !seen.insert(row.card_id) {
            return Err("card semantics row width or duplicate card id".into());
        }
        if row
            .vector
            .iter()
            .any(|&v| (v as f32) as f64 != f64::from(v))
        {
            return Err("card semantics integer not exact in f32".into());
        }
    }
    Ok((table, span))
}

/// The arm E reader's scaling of one table: every column is divided by the
/// next power of two at or above the largest absolute value in that column
/// (at least 1), so each scaled value is exact in f32.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalesV1 {
    pub schema: String,
    /// SHA-256 of the table file bytes the divisors were computed from.
    pub table_sha256: String,
    pub vector_width: usize,
    /// Number of table rows.
    pub card_count: usize,
    /// One power-of-two divisor per column, in column order.
    pub column_divisors: Vec<u64>,
    /// The reader's binding digest (the `card_semantics` checkpoint section
    /// binding): SHA-256 over the ASCII table digest, the dense scaled rows
    /// as f32 little-endian bits over ids `0..=max id`, then the row span and
    /// width as u64 little endian.
    pub binding: String,
}

/// Computes the arm E reader's divisors and binding for `table_bytes`. The
/// arithmetic is the reader's (`card_projection_e1`), restated so the
/// generator can write it; the tests require the binding to equal the
/// reader's.
pub fn scales_v1(table_bytes: &[u8]) -> Result<ScalesV1, String> {
    let digest = format!("{:x}", Sha256::digest(table_bytes));
    let (table, span) = parse_checked_v1(table_bytes)?;
    let width = table.vector_width;
    let mut maxima = vec![1u32; width];
    for row in &table.cards {
        for (max, &v) in maxima.iter_mut().zip(&row.vector) {
            *max = (*max).max(v.unsigned_abs());
        }
    }
    let divisors: Vec<u64> = maxima
        .into_iter()
        .map(|v| u64::from(v).next_power_of_two())
        .collect();
    let mut rows = vec![0.0f32; span * width];
    for row in &table.cards {
        for (j, &v) in row.vector.iter().enumerate() {
            rows[row.card_id * width + j] = v as f32 / divisors[j] as f32;
        }
    }
    let mut hash = Sha256::new();
    hash.update(digest.as_bytes());
    for v in &rows {
        hash.update(v.to_bits().to_le_bytes());
    }
    hash.update((span as u64).to_le_bytes());
    hash.update((width as u64).to_le_bytes());
    Ok(ScalesV1 {
        schema: SCALES_SCHEMA_V1.to_owned(),
        table_sha256: digest,
        vector_width: width,
        card_count: table.cards.len(),
        column_divisors: divisors,
        binding: format!("{:x}", hash.finalize()),
    })
}

/// The committed scales document text for `table_bytes`: fixed key order,
/// one line per field, a trailing newline.
pub fn scales_json_v1(table_bytes: &[u8]) -> Result<String, String> {
    let scales = scales_v1(table_bytes)?;
    let mut out = String::new();
    let divisors: Vec<String> = scales.column_divisors.iter().map(u64::to_string).collect();
    writeln!(out, "{{").unwrap();
    writeln!(out, "  \"schema\": \"{}\",", scales.schema).unwrap();
    writeln!(out, "  \"table_sha256\": \"{}\",", scales.table_sha256).unwrap();
    writeln!(out, "  \"vector_width\": {},", scales.vector_width).unwrap();
    writeln!(out, "  \"card_count\": {},", scales.card_count).unwrap();
    writeln!(out, "  \"column_divisors\": [{}],", divisors.join(",")).unwrap();
    writeln!(out, "  \"binding\": \"{}\"", scales.binding).unwrap();
    writeln!(out, "}}").unwrap();
    Ok(out)
}

/// The ids in `active` that have no row in `table_bytes`, or a row whose
/// every entry is zero (which the reader cannot tell from a missing row),
/// ascending. A table the arm E reader would refuse is an error.
pub fn missing_rows_v1(table_bytes: &[u8], active: &BTreeSet<u16>) -> Result<Vec<u16>, String> {
    let (table, _) = parse_checked_v1(table_bytes)?;
    let present: BTreeSet<usize> = table
        .cards
        .iter()
        .filter(|row| row.vector.iter().any(|&v| v != 0))
        .map(|row| row.card_id)
        .collect();
    Ok(active
        .iter()
        .copied()
        .filter(|&id| !present.contains(&usize::from(id)))
        .collect())
}

/// The active card ids: every name of the Cube list (`cube_list/v1`) that
/// the fence record (`{"cards": [{"name": ...}]}`) does not name, plus the
/// five basic lands. Every name must be in the card database and every
/// fenced name on the list.
pub fn active_ids_v1(cube_list: &str, fence: &str) -> Result<BTreeSet<u16>, String> {
    use crate::card_def::card_id_by_name;
    let list: serde_json::Value = serde_json::from_str(cube_list).map_err(|e| e.to_string())?;
    if list["schema"].as_str() != Some("cube_list/v1") {
        return Err("cube list schema is not cube_list/v1".into());
    }
    let names: Vec<&str> = list["cards"]
        .as_array()
        .ok_or("cube list has no cards array")?
        .iter()
        .map(|card| card["name"].as_str().ok_or("cube card without a name"))
        .collect::<Result<_, _>>()?;
    let mut listed = BTreeSet::new();
    if let Some(name) = names.iter().find(|name| !listed.insert(**name)) {
        return Err(format!("{name} is listed twice"));
    }
    let fenced =
        crate::cube_draft_v1::parse_fence_record(fence).map_err(|error| error.to_string())?;
    if let Some(name) = fenced.iter().find(|name| !names.contains(&name.as_str())) {
        return Err(format!("fenced card {name:?} is not on the cube list"));
    }
    let mut ids = BTreeSet::new();
    for name in names.into_iter().filter(|name| !fenced.contains(*name)) {
        let id = card_id_by_name(name).ok_or_else(|| format!("{name} is not in CARD_DEFS"))?;
        if !ids.insert(id) {
            return Err(format!("{name} is listed twice"));
        }
    }
    for id in crate::cube_draft_v1::basic_land_ids().map_err(|error| error.to_string())? {
        if !ids.insert(id) {
            return Err(format!("basic land {id} is also on the cube list"));
        }
    }
    Ok(ids)
}
