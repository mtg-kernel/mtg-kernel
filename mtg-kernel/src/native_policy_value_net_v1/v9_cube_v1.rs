//! The FlatV9 (Cube, M2 task T6) net: the V8 inclusion net
//! (`v8_inclusion_v1`) with its object and state first layers widened for
//! the FlatV9 columns (`native_flat_tensorizer_v9.rs`).
//!
//! - `object_encoder.0.weight` is `[64, 147]`: the 98 FlatV8 object columns,
//!   the 33 FlatV9 object columns, then the 16 card-embedding values.
//! - `state_encoder.0.weight` is `[64, 1550]`: the 219 FlatV8 state columns,
//!   the 51 FlatV9 state columns, then the 1,280 pooled object values.
//!
//! Every other tensor, the include column and the pairing path are the V8
//! net's. The FlatV9 net has its own architecture identity, config and
//! parameter count; it refuses FlatV8 inputs and every other net refuses
//! FlatV9's (their object and state widths differ).

use super::v8_inclusion_v1::V8_PARAMETER_COUNT_V1;
use super::*;
use crate::flat_policy_v6::{FLAT_V9_OBJECT_EXTRA_DIM_V1, FLAT_V9_STATE_EXTRA_DIM_V1};

/// Architecture identity of the FlatV9 net.
pub(crate) const V9_MODEL_ARCHITECTURE_VERSION_V1: &str = "kernel-policy-value-net-8-v9-cube";
/// FlatV9 object width.
pub(crate) const V9_OBJECT_FEATURE_DIM_V1: usize =
    OBJECT_FEATURE_DIM_V1 + FLAT_V9_OBJECT_EXTRA_DIM_V1;
/// FlatV9 state width.
pub(crate) const V9_STATE_DIM_V1: usize = STATE_DIM_V1 + FLAT_V9_STATE_EXTRA_DIM_V1;
/// `object_encoder.0`'s input width in the FlatV9 net.
pub(crate) const V9_OBJECT_ENCODER_INPUT_V1: usize =
    V9_OBJECT_FEATURE_DIM_V1 + CARD_EMBEDDING_DIM_V1;
/// `state_encoder.0`'s input width in the FlatV9 net.
pub(crate) const V9_STATE_ENCODER_INPUT_V1: usize = V9_STATE_DIM_V1 + POOLED_OBJECT_DIM_V1;
/// The V8 net's count plus the new first-layer columns.
pub(crate) const V9_PARAMETER_COUNT_V1: usize = V8_PARAMETER_COUNT_V1
    + HIDDEN_DIM_V1 * (FLAT_V9_OBJECT_EXTRA_DIM_V1 + FLAT_V9_STATE_EXTRA_DIM_V1);

impl NativePolicyValueModelConfigV1 {
    /// The FlatV9 net's config: the V8 net's with the FlatV9 widths, under
    /// the FlatV9 feature identity.
    pub(crate) const fn contract_v9_cube_v1() -> Self {
        use crate::native_flat_tensorizer_v9::*;
        Self {
            model_architecture_version: V9_MODEL_ARCHITECTURE_VERSION_V1,
            feature_schema_version: FEATURE_SCHEMA_VERSION_V9,
            feature_registry_version: FEATURE_REGISTRY_VERSION_V9,
            feature_contract_digest: FEATURE_CONTRACT_DIGEST_V9,
            feature_encoding_digest: FEATURE_ENCODING_DIGEST_V9,
            state_dim: V9_STATE_DIM_V1,
            object_feature_dim: V9_OBJECT_FEATURE_DIM_V1,
            ..Self::contract_v8_inclusion_v1()
        }
    }
}

/// `first` with `new` (row-major `[output, extra]`) inserted as columns
/// `at..at + extra`; the later columns move up by `extra`.
fn widened_v1(first: &LinearV1, at: usize, extra: usize, new: &[f32]) -> LinearV1 {
    debug_assert_eq!(new.len(), first.output_dim * extra);
    let mut weight = Vec::with_capacity(first.output_dim * (first.input_dim + extra));
    for (row, added) in first
        .weight
        .chunks_exact(first.input_dim)
        .zip(new.chunks_exact(extra))
    {
        weight.extend_from_slice(&row[..at]);
        weight.extend_from_slice(added);
        weight.extend_from_slice(&row[at..]);
    }
    LinearV1 {
        input_dim: first.input_dim + extra,
        output_dim: first.output_dim,
        weight,
        bias: first.bias.clone(),
        weight_t: Vec::new(),
        weight_finite: false,
    }
    .with_transposed_weight_v1()
}

impl NativePolicyValueNetV1 {
    /// Whether this is the FlatV9 net.
    pub(crate) fn is_v9_cube_v1(&self) -> bool {
        self.config.model_architecture_version == V9_MODEL_ARCHITECTURE_VERSION_V1
    }

    /// The FlatV9 net of `self` (a V8 inclusion net) with `object_columns`
    /// (row-major `[64, 33]`) and `state_columns` (row-major `[64, 51]`) as
    /// the new first-layer columns. Every V8 parameter keeps its bits.
    pub(crate) fn into_v9_cube_v1(
        self,
        object_columns: &[f32],
        state_columns: &[f32],
    ) -> Result<Self, NativePolicyValueErrorV1> {
        if !self.is_v8_inclusion_v1() || self.is_v9_cube_v1() {
            return Err(NativePolicyValueErrorV1::ParameterInvariant(
                "v9_requires_the_v8_net",
            ));
        }
        exact_len(
            "object_encoder.0.v9_columns",
            object_columns.len(),
            HIDDEN_DIM_V1 * FLAT_V9_OBJECT_EXTRA_DIM_V1,
        )?;
        exact_len(
            "state_encoder.0.v9_columns",
            state_columns.len(),
            HIDDEN_DIM_V1 * FLAT_V9_STATE_EXTRA_DIM_V1,
        )?;
        let mut model = self;
        model.object_encoder.first = widened_v1(
            &model.object_encoder.first,
            OBJECT_FEATURE_DIM_V1,
            FLAT_V9_OBJECT_EXTRA_DIM_V1,
            object_columns,
        );
        model.state_encoder.first = widened_v1(
            &model.state_encoder.first,
            STATE_DIM_V1,
            FLAT_V9_STATE_EXTRA_DIM_V1,
            state_columns,
        );
        model.config = NativePolicyValueModelConfigV1::contract_v9_cube_v1();
        model.validate_parameters_v1()?;
        if model.parameter_count_v1() != V9_PARAMETER_COUNT_V1 {
            return Err(NativePolicyValueErrorV1::ParameterInvariant(
                "parameter_count",
            ));
        }
        Ok(model)
    }

    /// A FlatV9 net to load a FlatV9 snapshot into
    /// (`replace_parameter_snapshot_v1`).
    pub(crate) fn v9_cube_template_v1() -> Result<Self, NativePolicyValueErrorV1> {
        Self::v8_inclusion_template_v1()?.into_v9_cube_v1(
            &[0.0; HIDDEN_DIM_V1 * FLAT_V9_OBJECT_EXTRA_DIM_V1],
            &[0.0; HIDDEN_DIM_V1 * FLAT_V9_STATE_EXTRA_DIM_V1],
        )
    }

    /// The reference forward over FlatV9 inputs (pairing relation included).
    pub(crate) fn forward_feature_transfer_v9_mode(
        &self,
        encoded: NativeEncodedDecisionViewV1<'_>,
        activation_mode: ForwardActivationModeV1,
    ) -> Result<NativePolicyValueOutputV1, NativePolicyValueErrorV1> {
        if !self.is_v9_cube_v1() {
            return Err(NativePolicyValueErrorV1::SchemaMismatch(
                "model_architecture_version",
            ));
        }
        let counts = encoded.validate(self.config)?;
        self.forward_validated_rows_v1(encoded, counts, None, activation_mode)
    }
}

/// The first-layer widths of a net's architecture: (object encoder input,
/// state encoder input).
pub(super) const fn first_layer_inputs_v1(architecture: &str) -> (usize, usize) {
    if const_str_eq_v1(architecture, V9_MODEL_ARCHITECTURE_VERSION_V1) {
        (V9_OBJECT_ENCODER_INPUT_V1, V9_STATE_ENCODER_INPUT_V1)
    } else {
        (OBJECT_ENCODER_INPUT_V1, STATE_ENCODER_INPUT_V1)
    }
}

const fn const_str_eq_v1(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v9_net_widens_only_the_object_and_state_first_layers() {
        let v8 = NativePolicyValueNetV1::v8_inclusion_template_v1().unwrap();
        let object: Vec<f32> = (0..HIDDEN_DIM_V1 * FLAT_V9_OBJECT_EXTRA_DIM_V1)
            .map(|i| i as f32 * 1e-3)
            .collect();
        let state: Vec<f32> = (0..HIDDEN_DIM_V1 * FLAT_V9_STATE_EXTRA_DIM_V1)
            .map(|i| -(i as f32) * 1e-3)
            .collect();
        let v9 = v8.clone().into_v9_cube_v1(&object, &state).unwrap();
        assert!(v9.is_v9_cube_v1() && v9.is_v8_inclusion_v1());
        assert_eq!(v9.parameter_count_v1(), V9_PARAMETER_COUNT_V1);
        assert_eq!(v9.config.object_feature_dim, 131);
        assert_eq!(v9.config.state_dim, 270);
        let first = &v9.object_encoder.first;
        let old = &v8.object_encoder.first;
        assert_eq!(first.input_dim, V9_OBJECT_ENCODER_INPUT_V1);
        for (new_row, (old_row, added)) in first.weight.chunks_exact(first.input_dim).zip(
            old.weight
                .chunks_exact(old.input_dim)
                .zip(object.chunks_exact(FLAT_V9_OBJECT_EXTRA_DIM_V1)),
        ) {
            assert_eq!(new_row[..98], old_row[..98]);
            assert_eq!(new_row[98..131], *added);
            assert_eq!(new_row[131..], old_row[98..]);
        }
        let first = &v9.state_encoder.first;
        let old = &v8.state_encoder.first;
        for (new_row, (old_row, added)) in first.weight.chunks_exact(first.input_dim).zip(
            old.weight
                .chunks_exact(old.input_dim)
                .zip(state.chunks_exact(FLAT_V9_STATE_EXTRA_DIM_V1)),
        ) {
            assert_eq!(new_row[..219], old_row[..219]);
            assert_eq!(new_row[219..270], *added);
            assert_eq!(new_row[270..], old_row[219..]);
        }
        // Only a V8 net converts, once.
        assert!(v9.clone().into_v9_cube_v1(&object, &state).is_err());
        let net8 =
            NativePolicyValueNetV1::runner_fixed_v1(NativePolicyValueModelConfigV1::contract_v1())
                .unwrap();
        assert!(net8.into_v9_cube_v1(&object, &state).is_err());
        // A snapshot round trip keeps every tensor.
        let mut template = NativePolicyValueNetV1::v9_cube_template_v1().unwrap();
        template
            .replace_parameter_snapshot_v1(&v9.parameter_snapshot_v1())
            .unwrap();
        assert_eq!(template.parameter_snapshot_v1(), v9.parameter_snapshot_v1());
    }
}
