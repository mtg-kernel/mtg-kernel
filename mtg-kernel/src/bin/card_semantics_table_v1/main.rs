//! Card semantics feature tables (arm E of the card representation design;
//! `docs/design/card_semantics_table_v1.md`).
//!
//! Walks the engine's own typed definition of every card in the Pauper pool
//! (`data/pauper_support_v1.json`, default build) or the Vintage Cube list
//! (`data/cube_v1/mtgo_vintage_cube_20261007.json`, `eternal-fixtures` build)
//! and writes a fixed-width integer vector per card, bound to
//! `KERNEL_CARDDB_HASH`. Only the declared consumers read the cube tables
//! (`checks::unreferenced`): M4's loader reads the provisional cube table and
//! arm E (`card_semantics_e1_cube_v1`, M2 T8) the full one.
//!
//! Usage (from the repository root or anywhere; paths are fixed):
//!   card_semantics_table_v1 --write          regenerate the Pauper table and dump (default build)
//!   card_semantics_table_v1 --check          verify the Pauper table and dump (default build)
//!   card_semantics_table_v1 --cube --write   regenerate the cube table and dump (eternal-fixtures)
//!   card_semantics_table_v1 --cube --check   verify the cube table and dump (eternal-fixtures)

mtg_kernel::fast_global_allocator!();

mod checks;
mod extract;
mod mapping;
mod table;

use std::io;
use std::path::{Path, PathBuf};

use mtg_kernel::card_def::KERNEL_CARDDB_HASH;

const CUBE_FEATURE_ERR: &str = "the cube table needs the eternal-fixtures build";
const CUBE_REGEN_CMD: &str = "cargo run --release -p mtg-kernel --features eternal-fixtures --bin card_semantics_table_v1 -- --cube --write";
const USAGE: &str = "usage: card_semantics_table_v1 [--cube] (--write | --check)";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate sits in the repository root")
        .to_path_buf()
}

fn read(repo: &Path, relative: &str) -> Result<String, String> {
    std::fs::read_to_string(repo.join(relative)).map_err(|e| format!("{relative}: {e}"))
}

fn read_optional(repo: &Path, relative: &str) -> Result<Option<String>, String> {
    match std::fs::read_to_string(repo.join(relative)) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{relative}: {e}")),
    }
}

struct Generated {
    table_text: String,
    dump_text: String,
    report: String,
    failures: Vec<String>,
}

/// Generates the Pauper table and dump text and runs the twin and census checks.
/// `prior` is the committed vocabulary, if any.
fn generate(repo: &Path, prior: &[(String, String)]) -> Result<Generated, String> {
    let pool_text = read(repo, table::POOL_PATH)?;
    let mapping_text = read(repo, table::MAPPING_PATH)?;
    let mapping = mapping::parse(&mapping_text)?;
    let built = table::build(&pool_text, &mapping_text, &mapping, prior);
    let names = table::pool_names(&pool_text);
    let before = table::rows(&names, None);

    let (census_text, mut failures) = checks::census_report(&before, &built.rows);
    let mapping_text_report = checks::mapping_report(&built.rows, &mapping);
    let (pairs_text, pair_failures) = checks::pairs(&built);
    failures.extend(pair_failures);

    let mut debug_heads = std::collections::BTreeSet::new();
    for row in &built.rows {
        debug_heads.extend(row.tokens.debug_heads.iter().cloned());
    }
    let mut census_block = String::new();
    census_block.push_str(&census_text);
    census_block.push_str(&mapping_text_report);
    census_block.push_str("similarity (Jaccard of nonzero (column, value) pairs):\n");
    census_block.push_str(&pairs_text);
    census_block.push_str(&format!(
        "variant names read from a Debug head (enums without serde): {}\n",
        debug_heads.len()
    ));
    for head in &debug_heads {
        census_block.push_str(&format!("  {head}\n"));
    }

    let table_text = table::render(&built);
    let dump_text = table::render_dump(&built, &census_block);
    Ok(Generated {
        table_text,
        dump_text,
        report: format!(
            "kernel_carddb_hash {KERNEL_CARDDB_HASH:016x}\nvector_width {}\ncards {}\n{census_block}",
            built.vocabulary.len(),
            built.rows.len()
        ),
        failures,
    })
}

/// Generates the Vintage Cube table and dump text and collects hard validation
/// failures before any output file is opened.
fn generate_cube(repo: &Path, cube_prior: &[(String, String)]) -> Result<Generated, String> {
    let cube_text = read(repo, table::CUBE_POOL_PATH)?;
    let mapping_text = read(repo, table::MAPPING_PATH)?;
    let pauper_text = read(repo, table::TABLE_PATH)?;

    let cube_names = table::parse_cube_names(&cube_text)?;
    let mapping = mapping::parse(&mapping_text)?;
    let pauper_vocab = table::try_committed_vocabulary(&pauper_text)?;

    let mut failures = Vec::new();
    if !table::content_hash_holds(&pauper_text) {
        failures.push(format!(
            "{}: content_sha256 does not hold",
            table::TABLE_PATH
        ));
    }
    if pauper_vocab.is_empty() {
        failures.push(format!("{}: vocabulary is empty", table::TABLE_PATH));
    }

    let pauper_vocab_sha256 = table::vocabulary_sha256(&pauper_vocab);
    let effective_prior: &[(String, String)] = if cube_prior.is_empty() {
        &pauper_vocab
    } else if cube_prior.starts_with(&pauper_vocab) {
        cube_prior
    } else {
        failures.push(format!(
            "{}: committed vocabulary does not start with the Pauper vocabulary",
            table::CUBE_TABLE_PATH
        ));
        &pauper_vocab
    };

    let selection = table::select_cube_rows(&cube_names, &mapping)?;
    let built = table::build_from_rows(table::TableBuildInput {
        pool_path: table::CUBE_POOL_PATH,
        pool_text: &cube_text,
        mapping_text: &mapping_text,
        rows: selection.rows,
        prior: effective_prior,
        pauper_vocabulary: Some(table::PauperVocabularyRef {
            path: table::TABLE_PATH,
            sha256: pauper_vocab_sha256.clone(),
            len: pauper_vocab.len(),
        }),
    })?;

    if built.vocabulary.len() < pauper_vocab.len()
        || built
            .vocabulary
            .iter()
            .zip(&pauper_vocab)
            .any(|(entry, (token, kind))| entry.token != *token || entry.kind != kind.as_str())
    {
        failures.push("generated cube vocabulary does not extend the Pauper vocabulary".into());
    }

    let report_block = checks::cube_report(
        &built,
        &selection.before_mapping,
        &mapping,
        pauper_vocab.len(),
        &selection.not_implemented,
    );
    let table_text = table::render(&built);
    if !table::content_hash_holds(&table_text) {
        failures.push("generated cube table failed content_sha256 self-check".into());
    }
    let dump_text = table::render_cube_dump(&built, &report_block);

    Ok(Generated {
        table_text,
        dump_text,
        report: format!(
            "kernel_carddb_hash {KERNEL_CARDDB_HASH:016x}\npauper_vocabulary_sha256 {pauper_vocab_sha256}\n{report_block}"
        ),
        failures,
    })
}

/// The provisional cube table's columns, read once and only when its bytes
/// are the ones M4 pins.
fn provisional_columns(repo: &Path) -> Result<Vec<(String, String)>, String> {
    let path = table::CUBE_PROVISIONAL_TABLE_PATH;
    let bytes = std::fs::read(repo.join(path)).map_err(|e| format!("{path}: {e}"))?;
    if table::sha256_hex(&bytes) != table::CUBE_PROVISIONAL_TABLE_SHA256 {
        return Err(format!(
            "{path} differs from the provisional table M4 pins ({})",
            table::CUBE_PROVISIONAL_TABLE_SHA256
        ));
    }
    let text = String::from_utf8(bytes).map_err(|e| format!("{path}: {e}"))?;
    table::try_committed_vocabulary(&text)
}

/// `--cube --write`: under the cube writer lock, generates the full table,
/// refuses it unless it keeps every provisional column at its index, computes
/// its arm E scales (M2 T8) before any output is opened, then stages the
/// table, dump and scales together and publishes the table last, restoring
/// the dump and scales if a later rename fails.
fn regenerate_cube(repo: &Path) -> Result<(), String> {
    with_writer_lock(repo, table::CUBE_TABLE_PATH, || {
        // The provisional columns are read once: they seed the first full
        // table and bound every later one.
        let provisional = provisional_columns(repo);
        let prior = match read_optional(repo, table::CUBE_TABLE_PATH)? {
            Some(text) => table::try_committed_vocabulary(&text)?,
            None => provisional.clone()?,
        };
        let mut generated = generate_cube(repo, &prior)?;
        let columns = table::try_committed_vocabulary(&generated.table_text)?;
        match provisional {
            Ok(provisional) if columns.starts_with(&provisional) => {}
            Ok(_) => generated.failures.push(format!(
                "generated cube table does not keep the columns of {}",
                table::CUBE_PROVISIONAL_TABLE_PATH
            )),
            Err(e) => generated.failures.push(e),
        }
        print!("{}", generated.report);
        require_checks(&generated.failures)?;
        let scales = cube_scales_text(&generated.table_text)?;
        publish_staged_with(
            repo,
            table::CUBE_TABLE_PATH,
            table::CUBE_DUMP_PATH,
            &generated,
            Some((cube_scales_path(), &scales)),
            rename_file,
        )
    })
}

fn committed_prior_at(repo: &Path, relative: &str) -> Result<Vec<(String, String)>, String> {
    match read_optional(repo, relative)? {
        Some(text) => table::try_committed_vocabulary(&text),
        None => Ok(Vec::new()),
    }
}

/// Every Pauper check, as failures; empty means the committed files are current.
fn check(repo: &Path) -> Result<(String, Vec<String>), String> {
    let committed_table = read(repo, table::TABLE_PATH)?;
    let committed_dump = read(repo, table::DUMP_PATH)?;
    let prior = table::try_committed_vocabulary(&committed_table)?;
    let generated = generate(repo, &prior)?;
    let mut failures = generated.failures;
    if !table::content_hash_holds(&committed_table) {
        failures.push(format!(
            "{}: content_sha256 does not hold",
            table::TABLE_PATH
        ));
    }
    let recorded: serde_json::Value =
        serde_json::from_str(&committed_table).map_err(|e| e.to_string())?;
    let hash = format!("{KERNEL_CARDDB_HASH:016x}");
    if recorded["kernel_carddb_hash"].as_str() != Some(hash.as_str()) {
        failures.push(format!(
            "the table records card database {}, the engine is {hash}: regenerate with --write",
            recorded["kernel_carddb_hash"]
        ));
    }
    if generated.table_text != committed_table {
        failures.push(format!(
            "{} differs from the generator output: regenerate with --write",
            table::TABLE_PATH
        ));
    }
    if generated.dump_text != committed_dump {
        failures.push(format!(
            "{} differs from the generator output: regenerate with --write",
            table::DUMP_PATH
        ));
    }
    for hit in checks::unreferenced(repo) {
        failures.push(format!("{hit} names the card semantics table"));
    }
    Ok((generated.report, failures))
}

/// Every cube check, as failures; empty means the committed files are current.
fn check_cube(repo: &Path) -> Result<(String, Vec<String>), String> {
    let committed_table = read(repo, table::CUBE_TABLE_PATH)?;
    let committed_dump = read(repo, table::CUBE_DUMP_PATH)?;
    let prior = table::try_committed_vocabulary(&committed_table)?;
    let generated = generate_cube(repo, &prior)?;
    let mut failures = generated.failures;
    if !table::content_hash_holds(&committed_table) {
        failures.push(format!(
            "{}: content_sha256 does not hold",
            table::CUBE_TABLE_PATH
        ));
    }
    let recorded: serde_json::Value =
        serde_json::from_str(&committed_table).map_err(|e| e.to_string())?;
    let hash = format!("{KERNEL_CARDDB_HASH:016x}");
    if recorded["kernel_carddb_hash"].as_str() != Some(hash.as_str()) {
        failures.push(format!(
            "the cube table records card database {}, the engine is {hash}: regenerate with `{CUBE_REGEN_CMD}`",
            recorded["kernel_carddb_hash"]
        ));
    }
    let pauper_text = read(repo, table::TABLE_PATH)?;
    let pauper_vocab = table::try_committed_vocabulary(&pauper_text)?;
    let expected_pauper_sha = table::vocabulary_sha256(&pauper_vocab);
    if recorded["pauper_vocabulary_sha256"].as_str() != Some(expected_pauper_sha.as_str()) {
        failures.push(format!(
            "the cube table records Pauper vocabulary {}, the committed Pauper table has {expected_pauper_sha}: regenerate with `{CUBE_REGEN_CMD}`",
            recorded["pauper_vocabulary_sha256"]
        ));
    }
    if generated.table_text != committed_table {
        failures.push(format!(
            "{} differs from the generator output: regenerate with `{CUBE_REGEN_CMD}`",
            table::CUBE_TABLE_PATH
        ));
    }
    if generated.dump_text != committed_dump {
        failures.push(format!(
            "{} differs from the generator output: regenerate with `{CUBE_REGEN_CMD}`",
            table::CUBE_DUMP_PATH
        ));
    }
    failures.extend(cube_scales_failures(repo, &committed_table));
    match provisional_columns(repo) {
        Ok(columns) if prior.starts_with(&columns) => {}
        Ok(_) => failures.push(format!(
            "{} does not keep the provisional table's columns at their indices",
            table::CUBE_TABLE_PATH
        )),
        Err(e) => failures.push(e),
    }
    for hit in checks::unreferenced(repo) {
        failures.push(format!("{hit} names the card semantics table"));
    }
    Ok((generated.report, failures))
}

/// The arm E scales document (M2 T8) of a cube table text: the trainer
/// reader's power-of-two column divisors and binding digest. It refuses a
/// table the reader would refuse.
#[cfg(feature = "eternal-fixtures")]
fn cube_scales_text(table_text: &str) -> Result<String, String> {
    mtg_kernel::card_semantics_e1_cube_v1::scales_json_v1(table_text.as_bytes())
        .map_err(|e| format!("{}: arm E scales: {e}", table::CUBE_TABLE_PATH))
}

#[cfg(not(feature = "eternal-fixtures"))]
fn cube_scales_text(_table_text: &str) -> Result<String, String> {
    Err(CUBE_FEATURE_ERR.to_string())
}

#[cfg(feature = "eternal-fixtures")]
fn cube_scales_path() -> &'static str {
    mtg_kernel::card_semantics_e1_cube_v1::SCALES_PATH_V1
}

#[cfg(not(feature = "eternal-fixtures"))]
fn cube_scales_path() -> &'static str {
    "data/card_semantics_v1/cube_table_v1_full_e1_scales.json"
}

/// The committed scales document must equal the one computed from the
/// committed cube table.
#[cfg(feature = "eternal-fixtures")]
fn cube_scales_failures(repo: &Path, committed_table: &str) -> Vec<String> {
    use mtg_kernel::card_semantics_e1_cube_v1::SCALES_PATH_V1;
    let expected = match cube_scales_text(committed_table) {
        Ok(text) => text,
        Err(e) => return vec![e],
    };
    match read_optional(repo, SCALES_PATH_V1) {
        Ok(Some(text)) if text == expected => Vec::new(),
        Ok(_) => vec![format!(
            "{SCALES_PATH_V1} differs from the committed cube table's scales: regenerate with `{CUBE_REGEN_CMD}`"
        )],
        Err(e) => vec![e],
    }
}

#[cfg(not(feature = "eternal-fixtures"))]
fn cube_scales_failures(_repo: &Path, _committed_table: &str) -> Vec<String> {
    vec![CUBE_FEATURE_ERR.to_string()]
}

fn require_checks(failures: &[String]) -> Result<(), String> {
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} check(s) failed:\n{}",
            failures.len(),
            failures.join("\n")
        ))
    }
}

/// The writer's lock file, inside the table's staging directory. It is never
/// a tracked data file, so locking it blocks no reader.
///
/// The file is permanent: no writer deletes it, so the path names one file
/// for good and every writer that opens it contends for the same lock. (A
/// lock file that is unlinked on release can be locked, already dead, by a
/// writer that opened it earlier, while a later writer locks its replacement.)
/// The name falls under the repository's `*.tmp` ignore rule, so the file
/// that stays behind is never an untracked change.
const LOCK_FILE: &str = "writer.lock.tmp";
/// The dump saved before publication; it survives only a failed rollback or
/// an interrupted publication, and then it is the only copy of the old dump.
const PREVIOUS_DUMP_FILE: &str = "previous.txt";

fn staging_dir(repo: &Path, table_rel: &str) -> PathBuf {
    repo.join(table_rel).with_extension("staging")
}

/// Joins cleanup errors to the primary result without hiding it.
fn with_cleanup_errors<T>(result: Result<T, String>, cleanup: Vec<String>) -> Result<T, String> {
    if cleanup.is_empty() {
        return result;
    }
    let cleanup = cleanup.join("; ");
    match result {
        Ok(_) => Err(format!("cleanup failed: {cleanup}")),
        Err(e) => Err(format!("{e}; cleanup also failed: {cleanup}")),
    }
}

/// Opens one table's writer lock file without locking it, creating the file
/// and its staging directory on the first write. The file is never truncated.
fn open_writer_lock(repo: &Path, table_rel: &str) -> Result<(PathBuf, std::fs::File), String> {
    use std::fs;
    let stage = staging_dir(repo, table_rel);
    fs::create_dir_all(&stage).map_err(|e| format!("{}: {e}", stage.display()))?;
    let lock_path = stage.join(LOCK_FILE);
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| format!("{}: {e}", lock_path.display()))?;
    Ok((lock_path, lock))
}

/// Takes the exclusive lock on an opened writer lock file and runs `body`
/// holding it. However long ago the file was opened, it is still the file at
/// its path, because no writer removes it.
fn run_locked<T>(
    lock_path: &Path,
    lock: std::fs::File,
    body: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    if let Err(e) = lock.try_lock() {
        return Err(format!(
            "cannot lock {} for regeneration (another writer holds it): {e}",
            lock_path.display()
        ));
    }
    let result = body();
    // Closing the file releases the lock. The file and its directory stay.
    drop(lock);
    result
}

/// Runs `body` holding the writer lock of one table. The lock is taken before
/// the committed prior is read and held through publication and cleanup.
fn with_writer_lock<T>(
    repo: &Path,
    table_rel: &str,
    body: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    let (lock_path, lock) = open_writer_lock(repo, table_rel)?;
    run_locked(&lock_path, lock, body)
}

/// Regenerates one table under its writer lock: `produce` reads the committed
/// prior and generates; hard failures stop before any output file is opened.
fn regenerate(
    repo: &Path,
    table_rel: &str,
    dump_rel: &str,
    produce: impl FnOnce() -> Result<Generated, String>,
) -> Result<(), String> {
    with_writer_lock(repo, table_rel, || {
        let generated = produce()?;
        print!("{}", generated.report);
        require_checks(&generated.failures)?;
        write_outputs_at(repo, table_rel, dump_rel, &generated)
    })
}

fn rename_file(from: &Path, to: &Path) -> io::Result<()> {
    std::fs::rename(from, to)
}

fn write_outputs_at(
    repo: &Path,
    table_rel: &str,
    dump_rel: &str,
    generated: &Generated,
) -> Result<(), String> {
    publish_staged(repo, table_rel, dump_rel, generated, rename_file)
}

// Publish the canonical JSON last. A failed rename leaves the previous JSON
// intact; restore its companion dump before removing the staged files.
// No fsync or directory sync: outputs are tracked in git; --check detects a mismatched pair.
// The caller holds the writer lock (`with_writer_lock`).
fn publish_staged(
    repo: &Path,
    table_rel: &str,
    dump_rel: &str,
    generated: &Generated,
    rename: impl FnMut(&Path, &Path) -> io::Result<()>,
) -> Result<(), String> {
    publish_staged_with(repo, table_rel, dump_rel, generated, None, rename)
}

/// The companion file staged with a table (the cube table's arm E scales).
const NEXT_COMPANION_FILE: &str = "companion.json";
/// The companion saved before publication; like [`PREVIOUS_DUMP_FILE`] it
/// survives only a failed rollback, as the only copy of the old companion.
const PREVIOUS_COMPANION_FILE: &str = "previous-companion.json";

/// One published output and the copy saved before publication.
struct Restore<'a> {
    saved: &'a Path,
    target: &'a Path,
    /// Whether a previous file existed (else restoring removes the output).
    had: bool,
}

/// A failed restore: the saved copy left on disk (none when no previous file
/// existed) and the error.
type RestoreFailure<'a> = (Option<&'a Path>, String);

/// Restores every published output, later ones first, and returns the
/// outputs that could not be put back.
fn restore_all<'a>(
    published: &[Restore<'a>],
    rename: &mut impl FnMut(&Path, &Path) -> io::Result<()>,
) -> Vec<RestoreFailure<'a>> {
    let mut failed = Vec::new();
    for output in published.iter().rev() {
        let back = if output.had {
            rename(output.saved, output.target)
        } else {
            std::fs::remove_file(output.target)
        };
        if let Err(e) = back {
            let kept = if output.had {
                output.saved.display().to_string()
            } else {
                "nothing (no previous file existed)".to_string()
            };
            failed.push((
                output.had.then_some(output.saved),
                format!(
                    "could not restore {}: {e}; retained {kept}",
                    output.target.display()
                ),
            ));
        }
    }
    failed
}

/// The publication error: the failed rename, then any failed restore.
fn restore_error(target: &Path, e: io::Error, failed: &[RestoreFailure<'_>]) -> String {
    let mut message = format!("{}: {e}", target.display());
    for (_, failure) in failed {
        message.push_str("; ");
        message.push_str(failure);
    }
    message
}

/// [`publish_staged`] with an optional companion output `(path, text)`.
/// Every output is staged and every previous file saved before the first
/// rename; the dump, then the companion, then the canonical JSON are renamed,
/// and a failed later rename restores the earlier outputs.
fn publish_staged_with(
    repo: &Path,
    table_rel: &str,
    dump_rel: &str,
    generated: &Generated,
    companion: Option<(&str, &str)>,
    mut rename: impl FnMut(&Path, &Path) -> io::Result<()>,
) -> Result<(), String> {
    use std::fs;
    let table = repo.join(table_rel);
    let dump = repo.join(dump_rel);
    let stage = staging_dir(repo, table_rel);
    let next_table = stage.join("table.json");
    let next_dump = stage.join("dump.txt");
    let old_dump = stage.join(PREVIOUS_DUMP_FILE);
    let next_companion = stage.join(NEXT_COMPANION_FILE);
    let old_companion = stage.join(PREVIOUS_COMPANION_FILE);
    let companion_path = companion.map(|(rel, _)| repo.join(rel));
    // Only files this publication owns are checked and removed: without a
    // companion, the companion names are someone else's files.
    let mut retained_files = vec![(&old_dump, &dump, "dump")];
    if let Some(path) = &companion_path {
        retained_files.push((&old_companion, path, "companion"));
    }
    for (retained, destination, of) in retained_files {
        match fs::symlink_metadata(retained) {
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Ok(_) => {
                return Err(format!(
                    "{} holds the {of} retained by an earlier failed or interrupted publication; \
                     reconcile {} and {} with it (restore it as the {of} or delete it), then rerun",
                    retained.display(),
                    destination.display(),
                    table.display()
                ));
            }
            Err(e) => return Err(format!("{}: {e}", retained.display())),
        }
    }
    fs::create_dir_all(&stage).map_err(|e| format!("{}: {e}", stage.display()))?;
    let at = |path: &Path, e: io::Error| format!("{}: {e}", path.display());
    let save = |from: &Path, to: &Path| -> Result<bool, String> {
        match fs::copy(from, to) {
            Ok(_) => Ok(true),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(format!("{} -> {}: {e}", from.display(), to.display())),
        }
    };
    let mut preserve_old_dump = false;
    let mut preserve_old_companion = false;
    let result = (|| -> Result<(), String> {
        fs::write(&next_table, &generated.table_text).map_err(|e| at(&next_table, e))?;
        fs::write(&next_dump, &generated.dump_text).map_err(|e| at(&next_dump, e))?;
        if let Some((_, text)) = companion {
            fs::write(&next_companion, text).map_err(|e| at(&next_companion, e))?;
        }
        let had_dump = save(&dump, &old_dump)?;
        let had_companion = match &companion_path {
            Some(path) => save(path, &old_companion)?,
            None => false,
        };
        rename(&next_dump, &dump).map_err(|e| at(&dump, e))?;
        let dump_restore = Restore {
            saved: &old_dump,
            target: &dump,
            had: had_dump,
        };
        let mut published = vec![dump_restore];
        if let Some(path) = &companion_path {
            if let Err(e) = rename(&next_companion, path) {
                let failed = restore_all(&published, &mut rename);
                preserve_old_dump = failed
                    .iter()
                    .any(|(kept, _)| *kept == Some(old_dump.as_path()));
                return Err(restore_error(path, e, &failed));
            }
            published.push(Restore {
                saved: &old_companion,
                target: path,
                had: had_companion,
            });
        }
        if let Err(e) = rename(&next_table, &table) {
            let failed = restore_all(&published, &mut rename);
            preserve_old_dump = failed
                .iter()
                .any(|(kept, _)| *kept == Some(old_dump.as_path()));
            preserve_old_companion = failed
                .iter()
                .any(|(kept, _)| *kept == Some(old_companion.as_path()));
            return Err(restore_error(&table, e, &failed));
        }
        Ok(())
    })();
    // Attempt every step; only the generator's own staged files are removed.
    // The staging directory stays: it holds the permanent writer lock.
    let mut cleanup = Vec::new();
    let mut owned = vec![&next_table, &next_dump];
    if !preserve_old_dump {
        owned.push(&old_dump);
    }
    if companion.is_some() {
        owned.push(&next_companion);
        if !preserve_old_companion {
            owned.push(&old_companion);
        }
    }
    for file in owned {
        match fs::remove_file(file) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => cleanup.push(at(file, e)),
        }
    }
    with_cleanup_errors(result, cleanup)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Write,
    Check,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cli {
    cube: bool,
    mode: Mode,
}

fn parse_args(args: &[String]) -> Result<Cli, String> {
    let mut cube = false;
    let mut mode: Option<Mode> = None;
    for arg in args {
        match arg.as_str() {
            "--cube" => {
                if cube {
                    return Err(USAGE.to_string());
                }
                cube = true;
            }
            "--write" => {
                if mode.replace(Mode::Write).is_some() {
                    return Err(USAGE.to_string());
                }
            }
            "--check" => {
                if mode.replace(Mode::Check).is_some() {
                    return Err(USAGE.to_string());
                }
            }
            _ => return Err(USAGE.to_string()),
        }
    }
    if cube && !cfg!(feature = "eternal-fixtures") {
        return Err(CUBE_FEATURE_ERR.to_string());
    }
    let mode = mode.ok_or_else(|| USAGE.to_string())?;
    Ok(Cli { cube, mode })
}

fn run_args(repo: &Path, args: &[String]) -> Result<(), String> {
    let cli = parse_args(args)?;
    if cli.cube {
        return match cli.mode {
            Mode::Write => regenerate_cube(repo),
            Mode::Check => {
                let (report, failures) =
                    with_writer_lock(repo, table::CUBE_TABLE_PATH, || check_cube(repo))?;
                print!("{report}");
                require_checks(&failures)?;
                println!("card semantics table v1 (cube): all checks passed");
                Ok(())
            }
        };
    }

    if cfg!(feature = "eternal-fixtures") {
        return match cli.mode {
            Mode::Write => Err(
                "the Pauper table requires the default build; pass --cube --write for the cube table"
                    .to_string(),
            ),
            Mode::Check => {
                println!(
                    "the Pauper table is checked in the default build; pass --cube --check for the cube table"
                );
                Ok(())
            }
        };
    }

    if cfg!(feature = "limited-fdn-fixtures") {
        return Err("card semantics table v1 requires the default Pauper catalog".to_string());
    }

    match cli.mode {
        Mode::Write => regenerate(repo, table::TABLE_PATH, table::DUMP_PATH, || {
            let prior = committed_prior_at(repo, table::TABLE_PATH)?;
            generate(repo, &prior)
        }),
        Mode::Check => {
            let (report, failures) = check(repo)?;
            print!("{report}");
            require_checks(&failures)?;
            println!("card semantics table v1: all checks passed");
            Ok(())
        }
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    run_args(&repo_root(), &args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    fn scratch_repo(label: &str) -> PathBuf {
        let temp = std::env::temp_dir().join(format!("cardsem-{label}-{}", std::process::id()));
        if temp.exists() {
            std::fs::remove_dir_all(&temp).unwrap();
        }
        std::fs::create_dir(&temp).unwrap();
        std::fs::create_dir_all(temp.join(table::DUMP_PATH).parent().unwrap()).unwrap();
        std::fs::create_dir_all(temp.join(table::CUBE_POOL_PATH).parent().unwrap()).unwrap();
        std::fs::copy(
            repo_root().join(table::MAPPING_PATH),
            temp.join(table::MAPPING_PATH),
        )
        .unwrap();
        temp
    }

    fn generated(table_text: &str, dump_text: &str) -> Generated {
        Generated {
            table_text: table_text.into(),
            dump_text: dump_text.into(),
            report: String::new(),
            failures: Vec::new(),
        }
    }

    fn locked_write(repo: &Path, generated: &Generated) -> Result<(), String> {
        with_writer_lock(repo, table::TABLE_PATH, || {
            write_outputs_at(repo, table::TABLE_PATH, table::DUMP_PATH, generated)
        })
    }

    /// Another writer holding the Pauper table's lock, opened as a writer
    /// opens it.
    fn hold_lock(repo: &Path) -> std::fs::File {
        let (_, lock) = open_writer_lock(repo, table::TABLE_PATH).unwrap();
        lock.try_lock().unwrap();
        lock
    }

    /// The names in a table's staging directory, sorted.
    fn staged(repo: &Path, table_rel: &str) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(staging_dir(repo, table_rel))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn table_replacement_failure_restores_the_previous_dump() {
        let temp = scratch_repo("restore");
        std::fs::create_dir(temp.join(table::TABLE_PATH)).unwrap();
        std::fs::write(temp.join(table::DUMP_PATH), "original").unwrap();
        assert!(locked_write(&temp, &generated("new table", "new dump")).is_err());
        assert_eq!(read(&temp, table::DUMP_PATH).unwrap(), "original");
        assert_eq!(staged(&temp, table::TABLE_PATH), [LOCK_FILE]);
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn table_replacement_failure_without_a_previous_dump_removes_the_new_dump() {
        let temp = scratch_repo("restore-none");
        std::fs::create_dir(temp.join(table::TABLE_PATH)).unwrap();
        let error = locked_write(&temp, &generated("new table", "new dump")).unwrap_err();
        assert!(!error.contains("could not restore"), "{error}");
        assert!(!temp.join(table::DUMP_PATH).exists());
        assert!(temp.join(table::TABLE_PATH).is_dir());
        assert_eq!(staged(&temp, table::TABLE_PATH), [LOCK_FILE]);
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn successful_write_leaves_only_the_writer_lock_in_staging() {
        let temp = scratch_repo("fresh");
        locked_write(&temp, &generated("new table", "new dump")).unwrap();
        assert_eq!(read(&temp, table::TABLE_PATH).unwrap(), "new table");
        assert_eq!(read(&temp, table::DUMP_PATH).unwrap(), "new dump");
        assert_eq!(staged(&temp, table::TABLE_PATH), [LOCK_FILE]);
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn successful_write_replaces_existing_outputs_and_drops_the_saved_dump() {
        let temp = scratch_repo("replace");
        std::fs::write(temp.join(table::TABLE_PATH), "old table").unwrap();
        std::fs::write(temp.join(table::DUMP_PATH), "old dump").unwrap();
        let stage = staging_dir(&temp, table::TABLE_PATH);
        let mut renames = 0;
        with_writer_lock(&temp, table::TABLE_PATH, || {
            publish_staged(
                &temp,
                table::TABLE_PATH,
                table::DUMP_PATH,
                &generated("new table", "new dump"),
                |from: &Path, to: &Path| {
                    renames += 1;
                    // The old dump is saved before the first rename, and the
                    // writer lock is still held during publication.
                    assert_eq!(
                        std::fs::read_to_string(stage.join(PREVIOUS_DUMP_FILE)).unwrap(),
                        "old dump"
                    );
                    let nested = with_writer_lock(&temp, table::TABLE_PATH, || Ok(()));
                    assert!(nested.unwrap_err().contains("cannot lock"));
                    std::fs::rename(from, to)
                },
            )
        })
        .unwrap();
        assert_eq!(renames, 2);
        assert_eq!(read(&temp, table::TABLE_PATH).unwrap(), "new table");
        assert_eq!(read(&temp, table::DUMP_PATH).unwrap(), "new dump");
        assert_eq!(staged(&temp, table::TABLE_PATH), [LOCK_FILE]);
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn failed_rollback_retains_the_previous_dump_and_a_retry_refuses_to_touch_it() {
        let temp = scratch_repo("failed-rollback");
        std::fs::write(temp.join(table::TABLE_PATH), "old table").unwrap();
        std::fs::write(temp.join(table::DUMP_PATH), "old dump").unwrap();
        let stage = staging_dir(&temp, table::TABLE_PATH);
        let previous = stage.join(PREVIOUS_DUMP_FILE);
        let mut renames = 0;
        let error = with_writer_lock(&temp, table::TABLE_PATH, || {
            publish_staged(
                &temp,
                table::TABLE_PATH,
                table::DUMP_PATH,
                &generated("new table", "new dump"),
                |from: &Path, to: &Path| {
                    renames += 1;
                    if renames == 1 {
                        std::fs::rename(from, to)
                    } else {
                        Err(io::Error::other("injected rename failure"))
                    }
                },
            )
        })
        .unwrap_err();
        assert_eq!(renames, 3, "publish dump, publish table, restore dump");
        assert!(error.contains("could not restore"), "{error}");
        assert!(error.contains(PREVIOUS_DUMP_FILE), "{error}");
        assert_eq!(read(&temp, table::TABLE_PATH).unwrap(), "old table");
        assert_eq!(read(&temp, table::DUMP_PATH).unwrap(), "new dump");
        assert_eq!(std::fs::read_to_string(&previous).unwrap(), "old dump");
        // The staged table is gone; the retained dump and the lock file stay.
        assert_eq!(
            staged(&temp, table::TABLE_PATH),
            [PREVIOUS_DUMP_FILE, LOCK_FILE]
        );

        // A retry, even one whose own staging would fail, never overwrites or
        // deletes the retained dump and names it.
        std::fs::create_dir(stage.join("table.json")).unwrap();
        let retry = locked_write(&temp, &generated("newer table", "newer dump")).unwrap_err();
        assert!(retry.contains("reconcile"), "{retry}");
        assert!(retry.contains(&previous.display().to_string()), "{retry}");
        assert_eq!(std::fs::read_to_string(&previous).unwrap(), "old dump");
        assert_eq!(read(&temp, table::TABLE_PATH).unwrap(), "old table");
        assert_eq!(read(&temp, table::DUMP_PATH).unwrap(), "new dump");
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn cleanup_continues_past_a_failed_first_step_and_keeps_the_primary_error() {
        let temp = scratch_repo("cleanup-errors");
        let stage = staging_dir(&temp, table::TABLE_PATH);
        // A non-empty directory where the staged table goes. The staged table
        // is the first file written and the first file cleanup removes, so
        // writing it fails (the primary error) and so does the first cleanup
        // step.
        let blocked = stage.join("table.json");
        std::fs::create_dir_all(&blocked).unwrap();
        std::fs::write(blocked.join("keep"), "x").unwrap();
        // A staged dump left by an interrupted run: a later cleanup step.
        std::fs::write(stage.join("dump.txt"), "interrupted").unwrap();
        let write_error = std::fs::write(&blocked, "probe").unwrap_err();
        let remove_error = std::fs::remove_file(&blocked).unwrap_err();

        let error = locked_write(&temp, &generated("new table", "new dump")).unwrap_err();
        // The primary error is kept, with the cleanup error beside it.
        assert_eq!(
            error,
            format!(
                "{path}: {write_error}; cleanup also failed: {path}: {remove_error}",
                path = blocked.display()
            )
        );
        // Cleanup went on after its first step failed: the later staged file
        // is gone, and only the entry that could not be removed is left.
        assert!(!stage.join("dump.txt").exists());
        assert_eq!(staged(&temp, table::TABLE_PATH), ["table.json", LOCK_FILE]);
        assert!(blocked.join("keep").is_file());
        assert!(!temp.join(table::TABLE_PATH).exists());
        assert!(!temp.join(table::DUMP_PATH).exists());
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn publication_error_survives_a_failed_first_cleanup_step_and_later_files_are_removed() {
        let temp = scratch_repo("cleanup-later-files");
        std::fs::write(temp.join(table::TABLE_PATH), "old table").unwrap();
        std::fs::write(temp.join(table::DUMP_PATH), "old dump").unwrap();
        let stage = staging_dir(&temp, table::TABLE_PATH);
        let blocked = stage.join("table.json");
        let mut renames = 0;
        let error = with_writer_lock(&temp, table::TABLE_PATH, || {
            publish_staged(
                &temp,
                table::TABLE_PATH,
                table::DUMP_PATH,
                &generated("new table", "new dump"),
                |_from: &Path, _to: &Path| {
                    renames += 1;
                    // Everything is staged by the first rename.
                    for name in ["table.json", "dump.txt", PREVIOUS_DUMP_FILE] {
                        assert!(stage.join(name).is_file(), "{name} is not staged");
                    }
                    // The staged table, the first file cleanup removes, turns
                    // into a non-empty directory that cannot be removed, and
                    // publication fails (the primary error).
                    std::fs::remove_file(&blocked).unwrap();
                    std::fs::create_dir(&blocked).unwrap();
                    std::fs::write(blocked.join("keep"), "x").unwrap();
                    Err(io::Error::other("injected rename failure"))
                },
            )
        })
        .unwrap_err();
        assert_eq!(renames, 1);
        let remove_error = std::fs::remove_file(&blocked).unwrap_err();
        // The primary error is kept, with the cleanup error beside it.
        assert_eq!(
            error,
            format!(
                "{}: injected rename failure; cleanup also failed: {}: {remove_error}",
                temp.join(table::DUMP_PATH).display(),
                blocked.display()
            )
        );
        // Cleanup went on after its first step failed: the staged dump and
        // the saved previous dump are gone, and nothing was published.
        assert_eq!(staged(&temp, table::TABLE_PATH), ["table.json", LOCK_FILE]);
        assert_eq!(read(&temp, table::TABLE_PATH).unwrap(), "old table");
        assert_eq!(read(&temp, table::DUMP_PATH).unwrap(), "old dump");
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn cleanup_error_after_success_is_reported() {
        assert_eq!(with_cleanup_errors(Ok(()), Vec::new()), Ok(()));
        assert_eq!(
            with_cleanup_errors(Ok(()), vec!["a: busy".into()]),
            Err::<(), _>("cleanup failed: a: busy".into())
        );
        assert_eq!(
            with_cleanup_errors::<()>(Err("primary".into()), vec!["a".into(), "b".into()]),
            Err("primary; cleanup also failed: a; b".into())
        );
    }

    #[test]
    fn failed_write_keeps_foreign_staging_files() {
        let temp = scratch_repo("foreign-failed");
        let stage = staging_dir(&temp, table::TABLE_PATH);
        std::fs::create_dir_all(&stage).unwrap();
        // A publication without a companion owns neither companion name.
        let foreign = ["local.txt", NEXT_COMPANION_FILE, PREVIOUS_COMPANION_FILE];
        for name in foreign {
            std::fs::write(stage.join(name), "local work").unwrap();
        }
        std::fs::create_dir(temp.join(table::TABLE_PATH)).unwrap();
        std::fs::write(temp.join(table::DUMP_PATH), "original").unwrap();
        let error = locked_write(&temp, &generated("new table", "new dump")).unwrap_err();
        assert!(!error.contains("retained"), "{error}");
        assert_eq!(read(&temp, table::DUMP_PATH).unwrap(), "original");
        for name in foreign {
            assert_eq!(
                std::fs::read_to_string(stage.join(name)).unwrap(),
                "local work"
            );
        }
        let mut expected = foreign.map(String::from).to_vec();
        expected.push(LOCK_FILE.into());
        expected.sort();
        assert_eq!(staged(&temp, table::TABLE_PATH), expected);
        // A successful publication without a companion leaves them too.
        std::fs::remove_dir(temp.join(table::TABLE_PATH)).unwrap();
        locked_write(&temp, &generated("new table", "new dump")).unwrap();
        assert_eq!(read(&temp, table::TABLE_PATH).unwrap(), "new table");
        assert_eq!(staged(&temp, table::TABLE_PATH), expected);
        std::fs::remove_dir_all(temp).unwrap();
    }

    /// With a companion, a retained companion copy blocks the next write and
    /// names the companion's own destination; a failed table rename restores
    /// the dump and the companion.
    #[test]
    fn companion_publication_restores_on_failure_and_names_its_destination() {
        let temp = scratch_repo("companion");
        let companion_rel = "data/card_semantics_v1/companion_fixture.json";
        std::fs::write(temp.join(table::DUMP_PATH), "old dump").unwrap();
        std::fs::write(temp.join(companion_rel), "old companion").unwrap();
        std::fs::create_dir(temp.join(table::TABLE_PATH)).unwrap();
        let publish = |temp: &Path| {
            with_writer_lock(temp, table::TABLE_PATH, || {
                publish_staged_with(
                    temp,
                    table::TABLE_PATH,
                    table::DUMP_PATH,
                    &generated("new table", "new dump"),
                    Some((companion_rel, "new companion")),
                    rename_file,
                )
            })
        };
        let error = publish(&temp).unwrap_err();
        assert!(!error.contains("could not restore"), "{error}");
        assert_eq!(read(&temp, table::DUMP_PATH).unwrap(), "old dump");
        assert_eq!(read(&temp, companion_rel).unwrap(), "old companion");
        assert_eq!(staged(&temp, table::TABLE_PATH), [LOCK_FILE]);
        let stage = staging_dir(&temp, table::TABLE_PATH);
        std::fs::write(stage.join(PREVIOUS_COMPANION_FILE), "kept").unwrap();
        std::fs::remove_dir(temp.join(table::TABLE_PATH)).unwrap();
        let retry = publish(&temp).unwrap_err();
        let destination = temp.join(companion_rel).display().to_string();
        assert!(
            retry.contains("companion retained") && retry.contains(&destination),
            "{retry}"
        );
        std::fs::remove_file(stage.join(PREVIOUS_COMPANION_FILE)).unwrap();
        publish(&temp).unwrap();
        assert_eq!(read(&temp, table::TABLE_PATH).unwrap(), "new table");
        assert_eq!(read(&temp, companion_rel).unwrap(), "new companion");
        assert_eq!(staged(&temp, table::TABLE_PATH), [LOCK_FILE]);
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn interrupted_staging_is_reused_after_the_writer_releases_its_lock() {
        let temp = scratch_repo("retry");
        let table = temp.join(table::TABLE_PATH);
        let stage = staging_dir(&temp, table::TABLE_PATH);
        std::fs::create_dir(&stage).unwrap();
        std::fs::write(stage.join("table.json"), "interrupted").unwrap();
        std::fs::write(stage.join("local.txt"), "local work").unwrap();
        let lock = hold_lock(&temp);
        let blocked = locked_write(&temp, &generated("new table", "new dump"));
        assert!(blocked.unwrap_err().contains("cannot lock"));
        assert_eq!(
            std::fs::read_to_string(stage.join("table.json")).unwrap(),
            "interrupted"
        );
        drop(lock);
        locked_write(&temp, &generated("new table", "new dump")).unwrap();
        assert_eq!(std::fs::read_to_string(table).unwrap(), "new table");
        assert_eq!(read(&temp, table::DUMP_PATH).unwrap(), "new dump");
        assert_eq!(
            std::fs::read_to_string(stage.join("local.txt")).unwrap(),
            "local work"
        );
        assert_eq!(staged(&temp, table::TABLE_PATH), ["local.txt", LOCK_FILE]);
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn writer_lock_is_a_staging_file_taken_before_the_prior_is_read() {
        let temp = scratch_repo("lock-order");
        let lock = hold_lock(&temp);
        let mut produced = false;
        let error = regenerate(&temp, table::TABLE_PATH, table::DUMP_PATH, || {
            produced = true;
            Ok(generated("new table", "new dump"))
        })
        .unwrap_err();
        assert!(error.contains("cannot lock"), "{error}");
        assert!(error.contains(LOCK_FILE), "{error}");
        assert!(!produced, "the prior was read without the writer lock");
        assert!(!temp.join(table::TABLE_PATH).exists());
        drop(lock);

        regenerate(&temp, table::TABLE_PATH, table::DUMP_PATH, || {
            // Held while generating; the tracked mapping input stays readable.
            let nested = with_writer_lock(&temp, table::TABLE_PATH, || Ok(()));
            assert!(nested.unwrap_err().contains("cannot lock"));
            let mapping = std::fs::File::open(temp.join(table::MAPPING_PATH)).unwrap();
            mapping.try_lock().unwrap();
            drop(mapping);
            read(&temp, table::MAPPING_PATH).unwrap();
            Ok(generated("new table", "new dump"))
        })
        .unwrap();
        assert_eq!(read(&temp, table::TABLE_PATH).unwrap(), "new table");
        assert_eq!(staged(&temp, table::TABLE_PATH), [LOCK_FILE]);
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn lock_file_is_not_replaced_between_a_writers_open_and_lock() {
        let temp = scratch_repo("lock-identity");
        let stage = staging_dir(&temp, table::TABLE_PATH);
        let lock_path = stage.join(LOCK_FILE);
        // Mark the lock file: a replacement would not carry the mark.
        std::fs::create_dir_all(&stage).unwrap();
        std::fs::write(&lock_path, "the first lock file").unwrap();

        // Writer B opens the lock file while writer A holds it, and is
        // delayed before it locks. A publishes and releases.
        let mut delayed = None;
        with_writer_lock(&temp, table::TABLE_PATH, || {
            delayed = Some(open_writer_lock(&temp, table::TABLE_PATH)?);
            write_outputs_at(
                &temp,
                table::TABLE_PATH,
                table::DUMP_PATH,
                &generated("table a", "dump a"),
            )
        })
        .unwrap();
        let (delayed_path, delayed_lock) = delayed.unwrap();
        assert_eq!(delayed_path, lock_path);
        // A left the lock file alone: the path still names the file B opened.
        assert_eq!(
            std::fs::read_to_string(&lock_path).unwrap(),
            "the first lock file"
        );

        // Writer C starts and takes the lock. B then locks what it opened:
        // that is the file C holds, so B is refused and does not run.
        let held = hold_lock(&temp);
        let mut ran = false;
        let error = run_locked(&delayed_path, delayed_lock, || {
            ran = true;
            Ok(())
        })
        .unwrap_err();
        assert!(error.contains("cannot lock"), "{error}");
        assert!(!ran, "two writers held the writer lock at once");
        drop(held);
        assert_eq!(read(&temp, table::TABLE_PATH).unwrap(), "table a");
        assert_eq!(staged(&temp, table::TABLE_PATH), [LOCK_FILE]);
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn permanent_writer_lock_is_covered_by_a_repository_ignore_rule() {
        // The lock file outlives every run, and builds record whether the
        // work tree has untracked files, so its name has to be an ignored one.
        let ignore = read(&repo_root(), ".gitignore").unwrap();
        let extension = Path::new(LOCK_FILE)
            .extension()
            .and_then(|extension| extension.to_str())
            .expect("the lock file name has an extension");
        let rule = format!("*.{extension}");
        assert!(
            ignore.lines().any(|line| line.trim() == rule),
            ".gitignore has no {rule} rule to cover {LOCK_FILE}"
        );
    }

    #[test]
    fn only_a_missing_committed_table_is_an_empty_prior() {
        let temp = scratch_repo("prior");
        let path = temp.join(table::TABLE_PATH);
        assert_eq!(committed_prior_at(&temp, table::TABLE_PATH), Ok(Vec::new()));
        std::fs::write(&path, [0xffu8, 0xfe, 0x00]).unwrap();
        assert!(committed_prior_at(&temp, table::TABLE_PATH).is_err());
        std::fs::write(&path, "{ not json").unwrap();
        assert!(committed_prior_at(&temp, table::TABLE_PATH).is_err());
        std::fs::write(&path, "{\"vocabulary\": 3}").unwrap();
        assert!(committed_prior_at(&temp, table::TABLE_PATH).is_err());
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(committed_prior_at(&temp, table::TABLE_PATH).is_err());
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn unexpected_arguments_get_the_usage_error() {
        for argv in [
            args(&[]),
            args(&["--write", "--unknown"]),
            args(&["--check", "extra"]),
            args(&["--write", "--check"]),
            args(&["--write", "--write"]),
            args(&["--cube", "--cube", "--write"]),
            args(&["--cube", "--write", "--check"]),
            args(&["--cube", "--unknown"]),
            args(&["write"]),
        ] {
            assert_eq!(parse_args(&argv).unwrap_err(), USAGE, "{argv:?}");
            assert_eq!(run_args(Path::new("unused"), &argv).unwrap_err(), USAGE);
        }
        let write = parse_args(&args(&["--write"])).unwrap();
        assert!(!write.cube && write.mode == Mode::Write);
        let check = parse_args(&args(&["--check"])).unwrap();
        assert!(!check.cube && check.mode == Mode::Check);
        for (argv, mode) in [
            (args(&["--cube", "--write"]), Mode::Write),
            (args(&["--write", "--cube"]), Mode::Write),
            (args(&["--cube", "--check"]), Mode::Check),
            (args(&["--check", "--cube"]), Mode::Check),
        ] {
            match parse_args(&argv) {
                Ok(cli) => {
                    if !cfg!(feature = "eternal-fixtures") {
                        panic!("the default build accepted {argv:?}");
                    }
                    assert!(cli.cube && cli.mode == mode, "{argv:?}");
                }
                Err(error) => {
                    if cfg!(feature = "eternal-fixtures") {
                        panic!("the eternal build refused {argv:?}: {error}");
                    }
                    assert_eq!(error, CUBE_FEATURE_ERR);
                }
            }
        }
    }

    fn tokens(items: &[(&str, extract::TokenKind, i64)]) -> extract::CardTokens {
        let mut out = extract::CardTokens::default();
        for &(token, kind, amount) in items {
            let (count, value) = match kind {
                extract::TokenKind::Count => (amount, 0),
                extract::TokenKind::Value => (1, amount),
            };
            out.tokens.insert(
                token.to_string(),
                extract::Acc {
                    kind,
                    count,
                    value,
                    via_mapping: false,
                    direct: true,
                },
            );
        }
        out
    }

    #[test]
    fn regeneration_keeps_a_reordered_prior_and_appends_only_new_tokens() {
        use super::extract::TokenKind::{Count, Value};
        let entry = |token: &str, kind: &str| (token.to_string(), kind.to_string());
        // Shared prefix, then cube-only columns out of sorted order, then a
        // column no card carries any more.
        let prior = vec![
            entry("type:Creature", "count"),
            entry("kw:FLYING", "count"),
            entry("z:cube_only", "count"),
            entry("a:cube_only", "value"),
            entry("leaf:Retired", "count"),
        ];
        let rows = vec![table::CardRow {
            name: "Fixture".into(),
            id: 0,
            tokens: tokens(&[
                ("type:Creature", Count, 1),
                ("z:cube_only", Count, 2),
                ("a:cube_only", Value, 7),
                ("m:new", Count, 1),
                ("b:new", Count, 3),
            ]),
        }];
        let built = table::build_from_rows(table::TableBuildInput {
            pool_path: table::CUBE_POOL_PATH,
            pool_text: "",
            mapping_text: "",
            rows,
            prior: &prior,
            pauper_vocabulary: None,
        })
        .unwrap();
        let order: Vec<(&str, &str, usize)> = built
            .vocabulary
            .iter()
            .map(|e| (e.token.as_str(), e.kind, e.carriers))
            .collect();
        assert_eq!(
            order,
            vec![
                ("type:Creature", "count", 1),
                ("kw:FLYING", "count", 0),
                ("z:cube_only", "count", 1),
                ("a:cube_only", "value", 1),
                ("leaf:Retired", "count", 0),
                ("b:new", "count", 1),
                ("m:new", "count", 1),
            ]
        );
        assert_eq!(built.vectors, vec![vec![1, 0, 2, 7, 0, 3, 1]]);
    }

    #[test]
    fn no_effect_reasons_state_only_what_the_definition_establishes() {
        use super::checks::{no_effect_token_reason_for as reason, DefinitionFacts};
        let mana_creature = DefinitionFacts {
            creature: true,
            mana_ability: true,
            ..DefinitionFacts::default()
        };
        assert_eq!(
            reason(mana_creature),
            "nonland permanent with only a mana ability in its definition"
        );
        // A mana creature with a keyword (a flying mana bird) is not mana-only.
        let with_keyword = DefinitionFacts {
            keywords: true,
            ..mana_creature
        };
        let flying = reason(with_keyword);
        assert!(!flying.contains("only"), "{flying}");
        assert!(flying.contains("mana ability") && flying.contains("keywords"));
        for other in [
            DefinitionFacts {
                static_rules: true,
                ..mana_creature
            },
            DefinitionFacts {
                abilities: true,
                ..mana_creature
            },
            DefinitionFacts {
                attachment: true,
                ..mana_creature
            },
        ] {
            assert_eq!(reason(other), flying);
        }
        let basic = DefinitionFacts {
            land: true,
            mana_ability: true,
            ..DefinitionFacts::default()
        };
        assert_eq!(
            reason(basic),
            "land with only a mana ability in its definition"
        );
        let entry_rule_land = DefinitionFacts {
            static_rules: true,
            ..basic
        };
        assert!(!reason(entry_rule_land).contains("only"));
        let no_mana_land = DefinitionFacts {
            land: true,
            ..DefinitionFacts::default()
        };
        assert_eq!(
            reason(no_mana_land),
            "land with no mana ability in its definition"
        );
        let vanilla = DefinitionFacts {
            creature: true,
            ..DefinitionFacts::default()
        };
        let keyword_creature = DefinitionFacts {
            keywords: true,
            ..vanilla
        };
        assert!(reason(vanilla).starts_with("creature with no keywords"));
        assert!(reason(keyword_creature).starts_with("creature with keywords"));
    }

    #[test]
    fn no_registry_card_with_other_rules_is_reported_as_mana_only() {
        for (index, def) in mtg_kernel::card_def::CARD_DEFS.iter().enumerate() {
            let id = u16::try_from(index).unwrap();
            let has_triggers = !mtg_kernel::trigger::triggers_for(id).is_empty();
            let facts = checks::DefinitionFacts::of(def, has_triggers);
            let reason = checks::no_effect_token_reason(id);
            // Keywords, static rules, attachments and abilities all count.
            if facts.has_other_rules() {
                assert!(!reason.contains("only"), "{}: {reason}", def.name);
            }
        }
    }

    #[test]
    fn jaccard_is_exact() {
        let j = checks::jaccard(&[1, 0, 3, 2], &[1, 5, 3, 0]);
        assert_eq!((j.shared, j.union), (2, 4));
        assert!(j.at_least(50) && j.at_most(50));
        let k = checks::jaccard(&[2], &[3]);
        assert_eq!((k.shared, k.union), (0, 2));
    }

    #[test]
    fn overlap_and_low_distinctiveness_match_exact_definition() {
        assert_eq!(checks::overlap(&[1, 0, 3, 2], &[1, 5, 3, 2]), (3, 3));
        assert_eq!(checks::overlap(&[1, 5, 3, 2], &[1, 0, 3, 2]), (3, 4));
        assert!(checks::overlap_meets_threshold(3, 4));
        assert!(!checks::overlap_meets_threshold(2, 3));
        assert!(!checks::overlap_meets_threshold(0, 0));

        // 11 identical 4-feature rows -> each has 10 other rows at 4/4 >= 0.75.
        // One extra row matches 3 of their 4 features -> overlaps them at 3/4,
        // and they overlap it at 3/4 (11 neighbors for the 11 rows, 11 for it).
        // One distinct row matches only 2 of 4 -> 0 neighbors >= 0.75.
        let mut rows = Vec::new();
        let mut vectors = Vec::new();
        for i in 0..11 {
            rows.push(table::CardRow {
                name: format!("Clone {i:02}"),
                id: i as u16,
                tokens: extract::CardTokens::default(),
            });
            vectors.push(vec![1, 2, 3, 4, 0]);
        }
        rows.push(table::CardRow {
            name: "ThreeQuarter".into(),
            id: 11,
            tokens: extract::CardTokens::default(),
        });
        vectors.push(vec![1, 2, 3, 0, 5]);
        rows.push(table::CardRow {
            name: "Distinct".into(),
            id: 12,
            tokens: extract::CardTokens::default(),
        });
        vectors.push(vec![1, 2, 9, 9, 9]);

        let synthetic = table::Table {
            pool_path: table::CUBE_POOL_PATH,
            pool_sha256: String::new(),
            mapping_sha256: String::new(),
            pauper_vocabulary: None,
            vocabulary: Vec::new(),
            rows,
            vectors,
        };
        let flagged = checks::low_distinctiveness(&synthetic);
        assert_eq!(flagged.len(), 12);
        assert!(flagged.iter().all(|row| row.name != "Distinct"));
        for row in &flagged {
            assert_eq!(row.neighbors, 11, "{row:?}");
            assert_eq!(row.nonzero, 4, "{row:?}");
        }
    }

    #[test]
    fn vocabulary_sha256_depends_on_order_and_kind() {
        let a = vec![
            ("color:R".into(), "count".into()),
            ("stat:mana_value".into(), "value".into()),
        ];
        let b = vec![
            ("stat:mana_value".into(), "value".into()),
            ("color:R".into(), "count".into()),
        ];
        let c = vec![
            ("color:R".into(), "value".into()),
            ("stat:mana_value".into(), "value".into()),
        ];
        assert_ne!(table::vocabulary_sha256(&a), table::vocabulary_sha256(&b));
        assert_ne!(table::vocabulary_sha256(&a), table::vocabulary_sha256(&c));
    }

    #[test]
    fn mapping_rejects_unjustified_entries() {
        let text = r#"{"schema":"card_semantics_leaf_mapping/v1","rules":[],"entries":[{"source":"leaf:X","emit":[{"token":"leaf:DealDamage"}],"why":" "}]}"#;
        assert!(mapping::parse(text).is_err());
    }

    #[test]
    fn both_committed_tables_keep_counts_within_bound() {
        const MAX_COLUMN_COUNT: i64 = 64;
        let repo = repo_root();
        for path in [table::TABLE_PATH, table::CUBE_TABLE_PATH] {
            let parsed: serde_json::Value =
                serde_json::from_str(&read(&repo, path).unwrap()).unwrap();
            let vocab = parsed["vocabulary"].as_array().unwrap();
            for row in parsed["cards"].as_array().unwrap() {
                let vector = row["vector"].as_array().unwrap();
                assert_eq!(vector.len(), vocab.len(), "{path}: {}", row["name"]);
                for (column, value) in vocab.iter().zip(vector) {
                    if column["kind"] == "count" {
                        assert!(
                            value.as_i64().unwrap() <= MAX_COLUMN_COUNT,
                            "{path}: {} has count {value} for {}",
                            row["name"],
                            column["token"],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn committed_cube_table_encodes_tamiyo_name_list_once() {
        let text = read(&repo_root(), table::CUBE_TABLE_PATH).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        let vocab = parsed["vocabulary"].as_array().unwrap();
        let tamiyo = parsed["cards"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["name"] == "Tamiyo, Collector of Tales")
            .unwrap();
        for (token, expected) in [
            (
                "leaf:StandardV1.EternalV1.C9cV1.TamiyoPlusOneNamedV1=named",
                1,
            ),
            (
                "leaf:StandardV1.EternalV1.C9cV1.TamiyoPlusOneNamedV1.named",
                0,
            ),
            ("op:Choice", 1),
            ("op:Choice.options", 1),
        ] {
            let index = vocab
                .iter()
                .position(|column| column["token"] == token)
                .unwrap();
            assert_eq!(tamiyo["vector"][index], expected, "{token}");
        }
    }

    #[test]
    fn committed_cube_header_and_content_hash_match_inputs() {
        let repo = repo_root();
        let text = read(&repo, table::CUBE_TABLE_PATH).unwrap();
        assert!(table::content_hash_holds(&text), "Cube content SHA-256");
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["schema"], table::TABLE_SCHEMA);
        for (path_key, hash_key, path) in [
            ("pool_path", "pool_sha256", table::CUBE_POOL_PATH),
            ("mapping_path", "mapping_sha256", table::MAPPING_PATH),
        ] {
            assert_eq!(parsed[path_key], path, "{path_key}");
            let bytes = std::fs::read(repo.join(path)).unwrap();
            assert_eq!(parsed[hash_key], table::sha256_hex(&bytes), "{hash_key}");
        }
        let pauper = read(&repo, table::TABLE_PATH).unwrap();
        let vocabulary = table::try_committed_vocabulary(&pauper).unwrap();
        assert_eq!(parsed["pauper_table_path"], table::TABLE_PATH);
        assert_eq!(parsed["pauper_vocabulary_len"], vocabulary.len());
        assert_eq!(
            parsed["pauper_vocabulary_sha256"],
            table::vocabulary_sha256(&vocabulary)
        );
    }

    #[cfg(not(feature = "eternal-fixtures"))]
    #[test]
    fn non_eternal_build_refuses_cube_flag() {
        for argv in [
            args(&["--cube"]),
            args(&["--cube", "--check"]),
            args(&["--cube", "--write"]),
            args(&["--write", "--cube"]),
        ] {
            assert_eq!(
                run_args(Path::new("unused"), &argv).unwrap_err(),
                CUBE_FEATURE_ERR
            );
        }
    }

    #[cfg(all(feature = "limited-fdn-fixtures", not(feature = "eternal-fixtures")))]
    #[test]
    fn extended_catalog_refuses_both_modes_before_reading_the_pool() {
        for mode in ["--check", "--write"] {
            assert_eq!(
                run_args(Path::new("unused"), &args(&[mode])).unwrap_err(),
                "card semantics table v1 requires the default Pauper catalog"
            );
        }
    }

    // The Pauper table binds the default catalog; feature builds that append
    // other registries change `KERNEL_CARDDB_HASH` and are out of its scope.
    #[cfg(not(feature = "limited-fdn-fixtures"))]
    #[test]
    fn write_refuses_a_changed_pair_before_replacing_outputs() {
        let temp = scratch_repo("changed-pair");
        let original = repo_root();
        for path in [table::POOL_PATH, table::TABLE_PATH, table::DUMP_PATH] {
            std::fs::copy(original.join(path), temp.join(path)).unwrap();
        }
        std::fs::write(
            temp.join(table::MAPPING_PATH),
            r#"{"schema":"card_semantics_leaf_mapping/v1","rules":[],"entries":[{"source":"leaf:DealDamage","emit":[{"token":"leaf:ReviewFixture"}],"why":"Changed mapping fixture"}]}"#,
        )
        .unwrap();
        let error = run_args(&temp, &args(&["--write"])).unwrap_err();
        assert!(error.contains("differs from pinned"), "{error}");
        for path in [table::TABLE_PATH, table::DUMP_PATH] {
            assert_eq!(
                std::fs::read(original.join(path)).unwrap(),
                std::fs::read(temp.join(path)).unwrap()
            );
        }
        // Nothing was staged: only the writer lock was taken, and released.
        assert_eq!(staged(&temp, table::TABLE_PATH), [LOCK_FILE]);
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[cfg(not(feature = "limited-fdn-fixtures"))]
    #[test]
    fn write_fails_on_an_unreadable_committed_table_instead_of_starting_over() {
        let temp = scratch_repo("unreadable-prior");
        let original = repo_root();
        for path in [table::POOL_PATH, table::DUMP_PATH] {
            std::fs::copy(original.join(path), temp.join(path)).unwrap();
        }
        for bytes in [[0xffu8, 0xfe, 0x00].as_slice(), b"{ not json".as_slice()] {
            std::fs::write(temp.join(table::TABLE_PATH), bytes).unwrap();
            let error = run_args(&temp, &args(&["--write"])).unwrap_err();
            assert!(!error.contains("check(s) failed"), "{error}");
            assert_eq!(std::fs::read(temp.join(table::TABLE_PATH)).unwrap(), bytes);
            assert_eq!(
                std::fs::read(original.join(table::DUMP_PATH)).unwrap(),
                std::fs::read(temp.join(table::DUMP_PATH)).unwrap()
            );
            assert_eq!(staged(&temp, table::TABLE_PATH), [LOCK_FILE]);
        }
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[cfg(feature = "eternal-fixtures")]
    #[test]
    fn cube_check_refuses_a_busy_publication_lock() {
        let temp = scratch_repo("cube-check-locked");
        let (_, lock) = open_writer_lock(&temp, table::CUBE_TABLE_PATH).unwrap();
        lock.try_lock().unwrap();

        let error = run_args(&temp, &args(&["--cube", "--check"])).unwrap_err();
        assert!(error.contains("cannot lock"), "{error}");

        drop(lock);
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn cube_table_references_outside_declared_consumers_are_flagged() {
        let temp = scratch_repo("cube-declared-consumers");
        // Exercise the separator mix emitted by the Windows directory walk.
        let integration_test = if cfg!(windows) {
            r"mtg-kernel/tests\eternal\cube_draft_build_v1.rs"
        } else {
            "mtg-kernel/tests/eternal/cube_draft_build_v1.rs"
        };
        let allowed = [
            "mtg-kernel/src/cube_draft_build_v1/nested/loader.rs",
            integration_test,
            "mtg-kernel/src/card_semantics_e1_cube_v1.rs",
            "mtg-kernel/tests/eternal/card_semantics_e1_cube_v1.rs",
        ];
        let denied = [
            "mtg-kernel/src/cube_draft_build_v1_extra/loader.rs",
            "mtg-kernel/tests/eternal/cube_draft_build_v1.rs.bak",
            "mtg-kernel/src/card_semantics_e1_cube_v1.rs.bak",
            "mtg-kernel/src/card_semantics_e1_cube_v1/reader.rs",
        ];
        for path in allowed.into_iter().chain(denied) {
            let path = temp.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            // M4's consumers name the provisional table (the full table is
            // covered by arm_e_cube_module_imports_outside_its_declarations_are_flagged).
            std::fs::write(&path, table::CUBE_PROVISIONAL_TABLE_PATH).unwrap();
        }

        let mut expected = denied.map(|path| temp.join(path));
        expected.sort();
        let mut hits: Vec<_> = checks::unreferenced(&temp)
            .into_iter()
            .map(PathBuf::from)
            .collect();
        hits.sort();
        std::fs::remove_dir_all(temp).unwrap();
        assert_eq!(hits, expected);
    }

    /// M2 T8: a consumer that reaches the full table through the arm E cube
    /// module's exported path names no table file, so the module name is a
    /// needle too; only its exact declarations are exempt.
    #[test]
    fn arm_e_cube_module_imports_outside_its_declarations_are_flagged() {
        let temp = scratch_repo("arm-e-module-imports");
        let write = |path: &str, text: &str| {
            let path = temp.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        };
        write(
            "mtg-kernel/src/lib.rs",
            "#[cfg(feature = \"eternal-fixtures\")]\npub mod card_semantics_e1_cube_v1;\n",
        );
        write(
            "mtg-kernel/tests/eternal_integration.rs",
            "#[path = \"eternal/card_semantics_e1_cube_v1.rs\"]\nmod card_semantics_e1_cube_v1;\n",
        );
        write(
            "python/tools/rust_integration_module_seconds.json",
            "{\"note\": \"bonder, card_semantics_e1_cube_v1, cube\", \"card_semantics_e1_cube_v1\": 1}\n",
        );
        let consumer = "mtg-kernel/src/undeclared_reader_v1.rs";
        write(
            consumer,
            "use crate::card_semantics_e1_cube_v1::TABLE_PATH_V1;\n",
        );
        let reexport = "mtg-kernel/examples/reexport_v1.rs";
        write(
            reexport,
            "pub mod card_semantics_e1_cube_v1;\nuse mtg_kernel::card_semantics_e1_cube_v1::TABLE_PATH_V1;\n",
        );
        // M4's provisional-table consumers may name the provisional table but
        // not the full one or the arm E module.
        write(
            "mtg-kernel/src/cube_draft_build_v1/card_index.rs",
            "const TABLE: &str = \"data/card_semantics_v1/cube_table_v1.json\";\n",
        );
        let m4_full = "mtg-kernel/src/cube_draft_build_v1/full_reader.rs";
        write(
            m4_full,
            "use crate::card_semantics_e1_cube_v1::TABLE_PATH_V1;\n",
        );
        let m4_full_path = "mtg-kernel/examples/cube_pod_runner/full_path.rs";
        write(
            m4_full_path,
            "const T: &str = \"data/card_semantics_v1/cube_table_v1_full.json\";\n",
        );
        let mut expected = vec![
            temp.join(reexport),
            temp.join(consumer),
            temp.join(m4_full),
            temp.join(m4_full_path),
        ];
        expected.sort();
        let hits: Vec<_> = checks::unreferenced(&temp)
            .into_iter()
            .map(PathBuf::from)
            .collect();
        // The declarations alone are not hits; a second line in lib.rs is.
        write(
            "mtg-kernel/src/lib.rs",
            "pub mod card_semantics_e1_cube_v1;\npub use card_semantics_e1_cube_v1::TABLE_PATH_V1;\n",
        );
        let with_lib: Vec<_> = checks::unreferenced(&temp)
            .into_iter()
            .map(PathBuf::from)
            .collect();
        std::fs::remove_dir_all(temp).unwrap();
        let mut hits_sorted = hits;
        hits_sorted.sort();
        assert_eq!(hits_sorted, expected);
        assert!(
            with_lib.iter().any(|p| p.ends_with("lib.rs")),
            "{with_lib:?}"
        );
    }

    #[cfg(not(feature = "limited-fdn-fixtures"))]
    #[test]
    fn committed_table_is_current_and_checks_hold() {
        let (report, failures) = check(&repo_root()).expect("committed files readable");
        println!("{report}");
        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[cfg(feature = "eternal-fixtures")]
    #[test]
    fn eternal_build_refuses_pauper_write_and_allows_pauper_check() {
        assert!(run_args(Path::new("unused"), &args(&["--write"])).is_err());
        assert!(run_args(Path::new("unused"), &args(&["--check"])).is_ok());
    }

    #[cfg(feature = "eternal-fixtures")]
    #[test]
    fn cube_write_refuses_hard_failure_before_replacing_outputs() {
        let temp = scratch_repo("cube-hard-fail");
        let original = repo_root();
        for path in [table::CUBE_POOL_PATH, table::TABLE_PATH] {
            std::fs::copy(original.join(path), temp.join(path)).unwrap();
        }
        let sentinel_table =
            r#"{"vocabulary":[{"index":0,"token":"wrong:prefix","kind":"count","carriers":0}]}"#;
        std::fs::write(temp.join(table::CUBE_TABLE_PATH), sentinel_table).unwrap();
        std::fs::write(temp.join(table::CUBE_DUMP_PATH), "sentinel dump").unwrap();
        let err = run_args(&temp, &args(&["--cube", "--write"])).unwrap_err();
        assert!(err.contains("Pauper vocabulary"), "{err}");
        assert_eq!(read(&temp, table::CUBE_TABLE_PATH).unwrap(), sentinel_table);
        assert_eq!(read(&temp, table::CUBE_DUMP_PATH).unwrap(), "sentinel dump");
        assert_eq!(staged(&temp, table::CUBE_TABLE_PATH), [LOCK_FILE]);
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[cfg(feature = "eternal-fixtures")]
    #[test]
    fn cube_write_fails_on_a_malformed_committed_cube_table() {
        let temp = scratch_repo("cube-malformed-prior");
        let original = repo_root();
        for path in [table::CUBE_POOL_PATH, table::TABLE_PATH] {
            std::fs::copy(original.join(path), temp.join(path)).unwrap();
        }
        std::fs::write(temp.join(table::CUBE_TABLE_PATH), "{ not json").unwrap();
        let err = run_args(&temp, &args(&["--cube", "--write"])).unwrap_err();
        assert!(err.contains("committed table JSON"), "{err}");
        assert_eq!(read(&temp, table::CUBE_TABLE_PATH).unwrap(), "{ not json");
        assert!(!temp.join(table::CUBE_DUMP_PATH).exists());
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[cfg(feature = "eternal-fixtures")]
    #[test]
    fn cube_regeneration_keeps_a_reordered_prior_and_appends_only_new_tokens() {
        let repo = repo_root();
        let pauper =
            table::try_committed_vocabulary(&read(&repo, table::TABLE_PATH).unwrap()).unwrap();
        let fresh = generate_cube(&repo, &[]).unwrap();
        let fresh = table::try_committed_vocabulary(&fresh.table_text).unwrap();
        let suffix = &fresh[pauper.len()..];
        assert!(
            suffix.len() >= 3,
            "too few cube-only columns: {}",
            suffix.len()
        );
        // A prior that reorders the cube-only columns, keeps a column no card
        // uses, and lacks one token the cards now carry.
        let new_token = suffix[0].clone();
        let mut prior = pauper.clone();
        prior.extend(suffix[1..].iter().rev().cloned());
        prior.push(("leaf:RetiredCubeFixture".to_string(), "count".to_string()));
        let generated = generate_cube(&repo, &prior).unwrap();
        assert!(generated.failures.is_empty(), "{:#?}", generated.failures);
        let mut expected = prior.clone();
        expected.push(new_token);
        let vocab = table::try_committed_vocabulary(&generated.table_text).unwrap();
        assert_eq!(vocab, expected);
    }

    #[cfg(feature = "eternal-fixtures")]
    #[test]
    fn mana_creature_with_a_keyword_is_not_reported_as_mana_only() {
        let id = mtg_kernel::card_def::card_id_by_name("Birds of Paradise").unwrap();
        let reason = checks::no_effect_token_reason(id);
        assert!(!reason.contains("only"), "{reason}");
        assert!(reason.contains("mana ability"), "{reason}");
    }

    #[cfg(feature = "eternal-fixtures")]
    #[test]
    fn cube_carddb_mismatch_fails_check_naming_regeneration_command() {
        let temp = scratch_repo("cube-carddb-mismatch");
        let original = repo_root();
        for path in [
            table::CUBE_POOL_PATH,
            table::TABLE_PATH,
            table::CUBE_PROVISIONAL_TABLE_PATH,
        ] {
            std::fs::copy(original.join(path), temp.join(path)).unwrap();
        }
        run_args(&temp, &args(&["--cube", "--write"])).unwrap();
        let current_hash = format!("\"kernel_carddb_hash\": \"{KERNEL_CARDDB_HASH:016x}\"");
        let stale_hash = "\"kernel_carddb_hash\": \"0000000000000000\"";
        let table_text = read(&temp, table::CUBE_TABLE_PATH).unwrap();
        assert!(table_text.contains(&current_hash));
        std::fs::write(
            temp.join(table::CUBE_TABLE_PATH),
            table_text.replacen(&current_hash, stale_hash, 1),
        )
        .unwrap();
        let (_, failures) = check_cube(&temp).unwrap();
        assert!(
            failures
                .iter()
                .any(|f| f.contains("card database") && f.contains(CUBE_REGEN_CMD)),
            "{failures:#?}"
        );
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[cfg(feature = "eternal-fixtures")]
    #[test]
    fn cube_write_and_check_roundtrip_preserves_pauper_prefix_and_is_idempotent() {
        let temp = scratch_repo("cube-roundtrip");
        let original = repo_root();
        for path in [
            table::CUBE_POOL_PATH,
            table::TABLE_PATH,
            table::CUBE_PROVISIONAL_TABLE_PATH,
        ] {
            std::fs::copy(original.join(path), temp.join(path)).unwrap();
        }
        run_args(&temp, &args(&["--cube", "--write"])).unwrap();
        let first_table = read(&temp, table::CUBE_TABLE_PATH).unwrap();
        let first_dump = read(&temp, table::CUBE_DUMP_PATH).unwrap();
        assert!(table::content_hash_holds(&first_table));

        let pauper_vocab =
            table::try_committed_vocabulary(&read(&temp, table::TABLE_PATH).unwrap()).unwrap();
        let cube_vocab = table::try_committed_vocabulary(&first_table).unwrap();
        assert!(cube_vocab.len() >= pauper_vocab.len());
        assert_eq!(&cube_vocab[..pauper_vocab.len()], pauper_vocab.as_slice());

        let parsed: serde_json::Value = serde_json::from_str(&first_table).unwrap();
        let card_names: std::collections::BTreeSet<&str> = parsed["cards"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["name"].as_str().unwrap())
            .collect();
        for land in table::BASIC_LANDS {
            assert!(card_names.contains(land), "missing basic land {land}");
        }
        assert_eq!(
            parsed["pauper_vocabulary_sha256"].as_str(),
            Some(table::vocabulary_sha256(&pauper_vocab).as_str())
        );

        let (_, failures) = check_cube(&temp).unwrap();
        assert!(failures.is_empty(), "{failures:#?}");

        // Second regeneration must keep exact column order and produce byte-identical output.
        let first_scales = read(&temp, cube_scales_path()).unwrap();
        run_args(&temp, &args(&["--cube", "--write"])).unwrap();
        assert_eq!(read(&temp, table::CUBE_TABLE_PATH).unwrap(), first_table);
        assert_eq!(read(&temp, table::CUBE_DUMP_PATH).unwrap(), first_dump);
        assert_eq!(read(&temp, cube_scales_path()).unwrap(), first_scales);
        assert_eq!(staged(&temp, table::CUBE_TABLE_PATH), [LOCK_FILE]);
        std::fs::remove_dir_all(temp).unwrap();
    }

    /// M2 T8: the full table keeps every provisional column at its index and
    /// adds the five fetch columns; the fetch family tells apart fetchlands
    /// whose rows the walk alone made identical.
    #[cfg(feature = "eternal-fixtures")]
    #[test]
    fn full_cube_table_keeps_provisional_columns_and_adds_the_fetch_family() {
        let repo = repo_root();
        let provisional = table::try_committed_vocabulary(
            &read(&repo, table::CUBE_PROVISIONAL_TABLE_PATH).unwrap(),
        )
        .unwrap();
        let generated = generate_cube(&repo, &provisional).unwrap();
        assert!(generated.failures.is_empty(), "{:#?}", generated.failures);
        let vocab = table::try_committed_vocabulary(&generated.table_text).unwrap();
        assert_eq!(vocab[..provisional.len()], provisional);
        let fetch: Vec<usize> = table::BASIC_LANDS
            .iter()
            .map(|basic| {
                let token = format!("{}{basic}", table::FETCH_PREFIX);
                vocab.iter().position(|(t, _)| *t == token).expect(&token)
            })
            .collect();
        assert!(fetch.iter().all(|&i| i >= provisional.len()));
        assert!(fetch.iter().all(|&i| vocab[i].1 == "count"));
        let parsed: serde_json::Value = serde_json::from_str(&generated.table_text).unwrap();
        let row = |name: &str| -> Vec<i64> {
            let row = parsed["cards"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["name"] == name)
                .unwrap_or_else(|| panic!("{name}"));
            fetch
                .iter()
                .map(|&i| row["vector"][i].as_i64().unwrap())
                .collect()
        };
        // Plains, Island, Swamp, Mountain, Forest.
        assert_eq!(row("Scalding Tarn"), [0, 1, 0, 1, 0]);
        assert_eq!(row("Polluted Delta"), [0, 1, 1, 0, 0]);
        assert_eq!(row("Arid Mesa"), [1, 0, 0, 1, 0]);
        assert_eq!(row("Island"), [0, 0, 0, 0, 0]);
        assert_eq!(row("Lightning Bolt"), [0, 0, 0, 0, 0]);
    }

    /// M2 T8: `--cube --check` refuses a changed scales file and a changed
    /// provisional table.
    #[cfg(feature = "eternal-fixtures")]
    #[test]
    fn cube_check_flags_changed_scales_and_provisional_table() {
        let temp = scratch_repo("cube-scales-provisional");
        let original = repo_root();
        for path in [
            table::CUBE_POOL_PATH,
            table::TABLE_PATH,
            table::CUBE_PROVISIONAL_TABLE_PATH,
        ] {
            std::fs::copy(original.join(path), temp.join(path)).unwrap();
        }
        run_args(&temp, &args(&["--cube", "--write"])).unwrap();
        let (_, failures) = check_cube(&temp).unwrap();
        assert!(failures.is_empty(), "{failures:#?}");
        let scales = mtg_kernel::card_semantics_e1_cube_v1::SCALES_PATH_V1;
        let text = read(&temp, scales).unwrap();
        std::fs::write(
            temp.join(scales),
            text.replacen("\"binding\": \"", "\"binding\": \"0", 1),
        )
        .unwrap();
        let mut provisional = std::fs::read(temp.join(table::CUBE_PROVISIONAL_TABLE_PATH)).unwrap();
        provisional.push(b'\n');
        std::fs::write(temp.join(table::CUBE_PROVISIONAL_TABLE_PATH), provisional).unwrap();
        let (_, failures) = check_cube(&temp).unwrap();
        std::fs::remove_dir_all(temp).unwrap();
        assert!(failures.iter().any(|f| f.contains(scales)), "{failures:#?}");
        assert!(
            failures
                .iter()
                .any(|f| f.contains("differs from the provisional table M4 pins")),
            "{failures:#?}"
        );
    }

    #[cfg(feature = "eternal-fixtures")]
    #[test]
    fn committed_cube_table_is_current_and_checks_hold() {
        let (report, failures) = check_cube(&repo_root()).expect("committed cube files readable");
        println!("{report}");
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
