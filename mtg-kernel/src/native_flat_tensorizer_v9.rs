//! FlatV9 (Python `features_v12.py`, `data/flat_policy_v9/`): the Cube
//! feature contract (M2 task T6) on FlatV8. Additive sibling of
//! `native_flat_tensorizer_v8.rs`; that module, its identity and
//! `data/flat_policy_v8/*` are never touched here.
//!
//! Every FlatV9 tensor is the FlatV8 tensor of the relaxed decision
//! (`flat_policy_v6.rs`), except:
//! - each object row has `FLAT_V9_OBJECT_EXTRA_DIM_V1` more columns, a
//!   public card's FlatV9 columns or zeros (`FlatV9ColumnsV1::object_row_v1`),
//!   so the object width is 131;
//! - the state row has `FLAT_V9_STATE_EXTRA_DIM_V1` more columns, so the
//!   state width is 270.
//!
//! The V9 identity is never compared against the V8, V7 or V4 one or
//! relabeled as either.


use crate::flat_policy_v6::{
    FlatScoringDecisionViewV9, FLAT_V9_OBJECT_EXTRA_DIM_V1, FLAT_V9_STATE_EXTRA_DIM_V1,
};
use crate::native_flat_tensorizer_v2::{
    object_node_to_raw_v9, NativeFlatDecisionTensorV2, NativeFlatTensorErrorV2,
    NATIVE_FLAT_OBJECT_FEATURE_DIM_V2, NATIVE_FLAT_STATE_FEATURE_DIM_V2,
};
use crate::native_flat_tensorizer_v8::NativeFlatTensorizerV8;
use crate::native_policy_value_net_v1::v8_inclusion_v1::V8_ACTION_FEATURE_DIM_V1;
use crate::native_policy_value_net_v1::{
    NativeEncodedDecisionSchemaV1, NativeEncodedDecisionViewV1,
};

// Generated from features_v12.py by data/flat_policy_v9/generate_contract.py.
#[allow(dead_code)]
mod identity {
    include!("../../data/flat_policy_v9/feature_identity.rs");
}
pub(crate) use identity::*;

/// FlatV9 object width: FlatV8's plus the FlatV9 object columns.
pub(crate) const V9_OBJECT_FEATURE_DIM_V1: usize =
    NATIVE_FLAT_OBJECT_FEATURE_DIM_V2 + FLAT_V9_OBJECT_EXTRA_DIM_V1;
/// FlatV9 state width: FlatV8's plus the FlatV9 state columns.
pub(crate) const V9_STATE_FEATURE_DIM_V1: usize =
    NATIVE_FLAT_STATE_FEATURE_DIM_V2 + FLAT_V9_STATE_EXTRA_DIM_V1;

/// The V9 tensorizer: the V8 fill, then the FlatV9 object and state
/// columns. Poisoned by its first failure, as V8 is.
#[derive(Default)]
pub(crate) struct NativeFlatTensorizerV9 {
    poisoned: bool,
    v8: NativeFlatTensorizerV8,
    v8_tensor: NativeFlatDecisionTensorV2,
    pairs: Vec<i64>,
}

impl NativeFlatTensorizerV9 {
    /// See `NativeFlatTensorizerV2::set_state_digest_v1`.
    pub(crate) fn set_state_digest_v1(&mut self, on: bool) {
        self.v8.set_state_digest_v1(on);
    }

    /// See `NativeFlatTensorizerV2::set_action_digest_v1`.
    pub(crate) fn set_action_digest_v1(&mut self, on: bool) {
        self.v8.set_action_digest_v1(on);
    }

    /// Fills `common` (131-wide object rows, 270-wide state, 196-wide action
    /// rows) and `pairing` transactionally: on an error both keep their
    /// previous decision.
    pub(crate) fn fill(
        &mut self,
        decision: FlatScoringDecisionViewV9<'_>,
        common: &mut NativeFlatDecisionTensorV2,
        pairing: &mut Vec<i64>,
    ) -> Result<(), NativeFlatTensorErrorV2> {
        if self.poisoned {
            return Err(NativeFlatTensorErrorV2::Poisoned);
        }
        self.fill_inner_v1(decision, common, pairing)
            .inspect_err(|_| self.poisoned = true)
    }

    fn fill_inner_v1(
        &mut self,
        decision: FlatScoringDecisionViewV9<'_>,
        common: &mut NativeFlatDecisionTensorV2,
        pairing: &mut Vec<i64>,
    ) -> Result<(), NativeFlatTensorErrorV2> {
        self.v8
            .fill(decision.v4, &mut self.v8_tensor, &mut self.pairs)?;
        let node_to_raw = object_node_to_raw_v9(decision.v4)?;
        let objects = decision.v4.common().objects();
        let width = NATIVE_FLAT_OBJECT_FEATURE_DIM_V2;
        let rows = self.v8_tensor.object_features.len() / width;
        if rows * width != self.v8_tensor.object_features.len()
            || rows != node_to_raw.len().max(1)
            || decision.columns.state.len() != FLAT_V9_STATE_EXTRA_DIM_V1
            || self.v8_tensor.state.len() != NATIVE_FLAT_STATE_FEATURE_DIM_V2
        {
            return Err(NativeFlatTensorErrorV2::OutputInvariant);
        }
        let mut object_features = Vec::new();
        object_features
            .try_reserve_exact(rows * V9_OBJECT_FEATURE_DIM_V1)
            .map_err(|_| NativeFlatTensorErrorV2::AllocationFailed)?;
        for (node, row) in self
            .v8_tensor
            .object_features
            .chunks_exact(width)
            .enumerate()
        {
            object_features.extend_from_slice(row);
            let extra = match node_to_raw.get(node) {
                // The padding row of a decision without objects.
                None => [0.0; FLAT_V9_OBJECT_EXTRA_DIM_V1],
                Some(&raw) => {
                    let object = objects
                        .get(raw)
                        .ok_or(NativeFlatTensorErrorV2::ObjectShape)?;
                    decision
                        .columns
                        .object_row_v1(object)
                        .map_err(|_| NativeFlatTensorErrorV2::ObjectShape)?
                }
            };
            object_features.extend_from_slice(&extra);
        }
        let mut state = Vec::new();
        state
            .try_reserve_exact(V9_STATE_FEATURE_DIM_V1)
            .map_err(|_| NativeFlatTensorErrorV2::AllocationFailed)?;
        state.extend_from_slice(&self.v8_tensor.state);
        state.extend_from_slice(&decision.columns.state);
        common.clone_from(&self.v8_tensor);
        common.object_features = object_features;
        common.state = state;
        pairing.clone_from(&self.pairs);
        Ok(())
    }
}

pub(crate) fn schema_v9() -> NativeEncodedDecisionSchemaV1 {
    NativeEncodedDecisionSchemaV1 {
        version: FEATURE_SCHEMA_VERSION_V9,
        registry_version: FEATURE_REGISTRY_VERSION_V9,
        contract_digest: FEATURE_CONTRACT_DIGEST_V9,
        encoding_digest: FEATURE_ENCODING_DIGEST_V9,
        state_dim: V9_STATE_FEATURE_DIM_V1,
        object_feature_dim: V9_OBJECT_FEATURE_DIM_V1,
        action_feature_dim: V8_ACTION_FEATURE_DIM_V1,
        ..NativeEncodedDecisionSchemaV1::contract_v1()
    }
}

/// A V9-labeled view of FlatV9 tensors and their pairing relation.
pub(crate) fn encoded_decision_view_v9<'a>(
    t: &'a NativeFlatDecisionTensorV2,
    pairing_node_indices: &'a [i64],
) -> NativeEncodedDecisionViewV1<'a> {
    NativeEncodedDecisionViewV1::from_slices_unvalidated(
        schema_v9(),
        &t.state,
        &t.object_features,
        &t.object_card_ids,
        &t.object_groups,
        &t.object_node_ids,
        &t.edge_features,
        &t.edge_source_indices,
        &t.edge_target_indices,
        &t.action_features,
        &t.action_ref_features,
        &t.action_ref_card_ids,
        &t.action_ref_action_indices,
        &t.action_ref_node_indices,
    )
    .with_pairing_node_indices_v1(pairing_node_indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn flat_v9_source_and_descriptor_match_their_independent_pins() {
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(include_bytes!("../../python/mtg_kernel_rl/features_v12.py"))
            ),
            FEATURES_SOURCE_SHA256_V9
        );
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(include_bytes!(
                    "../../data/flat_policy_v9/feature_contract_v9.json"
                ))
            ),
            FEATURE_DESCRIPTOR_SHA256_V9
        );
        // Never the identity of a generation it extends.
        for (v9, other) in [
            (
                FEATURE_CONTRACT_DIGEST_V9,
                crate::native_flat_tensorizer_v8::FEATURE_CONTRACT_DIGEST_V8,
            ),
            (
                FEATURE_CONTRACT_DIGEST_V9,
                crate::native_flat_tensorizer_v7::FEATURE_CONTRACT_DIGEST_V7,
            ),
            (
                FEATURE_CONTRACT_DIGEST_V9,
                crate::native_flat_tensorizer_v4::FEATURE_CONTRACT_DIGEST_V4,
            ),
            (
                FEATURE_ENCODING_DIGEST_V9,
                crate::native_flat_tensorizer_v8::FEATURE_ENCODING_DIGEST_V8,
            ),
            (
                FEATURES_SOURCE_SHA256_V9,
                crate::native_flat_tensorizer_v8::FEATURES_SOURCE_SHA256_V8,
            ),
        ] {
            assert_ne!(v9, other);
        }
        let schema = schema_v9();
        let v8 = crate::native_flat_tensorizer_v8::schema_v8();
        assert_eq!(
            (schema.state_dim, schema.object_feature_dim),
            (
                v8.state_dim + FLAT_V9_STATE_EXTRA_DIM_V1,
                v8.object_feature_dim + FLAT_V9_OBJECT_EXTRA_DIM_V1
            )
        );
        assert_eq!(
            (
                schema.action_feature_dim,
                schema.edge_feature_dim,
                schema.action_ref_feature_dim,
                schema.object_group_count
            ),
            (
                v8.action_feature_dim,
                v8.edge_feature_dim,
                v8.action_ref_feature_dim,
                v8.object_group_count
            )
        );
    }
}
