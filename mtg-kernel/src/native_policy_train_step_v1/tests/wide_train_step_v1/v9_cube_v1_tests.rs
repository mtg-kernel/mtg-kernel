//! The FlatV9 net (`native_policy_value_net_v1::v9_cube_v1`) in the CPU
//! train step: the tape forward reproduces the reference forward's bits on
//! synthetic FlatV9 rows (every update checks them), the new object and
//! state first-layer columns get the gradients finite differences give,
//! sequential and fixed-partition backward agree, and the FlatV8 and FlatV9
//! nets refuse each other's inputs.

use super::*;
use crate::flat_policy_v6::{FLAT_V9_OBJECT_EXTRA_DIM_V1, FLAT_V9_STATE_EXTRA_DIM_V1};
use crate::native_flat_tensorizer_v8::schema_v8;
use crate::native_flat_tensorizer_v9::schema_v9;
use crate::native_policy_value_net_v1::v9_cube_v1::{
    V9_OBJECT_ENCODER_INPUT_V1, V9_STATE_ENCODER_INPUT_V1,
};
use crate::native_policy_value_net_v1::ForwardActivationModeV1;

/// The include rows and pairs of the V8 tests' synthetic decisions.
const INCLUDE_ROWS_V1: [&[usize]; 2] = [&[1, 3], &[4]];
const PAIRS_V1: [&[i64]; 2] = [&[1, 4, 7, 2], &[3, 6, 8, 6, 0, 9]];

/// A deterministic value in [-1, 1) for synthetic FlatV9 columns.
fn column_v1(seed: usize) -> f32 {
    let mixed = (seed as u32).wrapping_mul(2_654_435_761).rotate_left(7);
    (mixed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
}

/// A FlatV9 version of a gradient decision: every action row widened by its
/// include bit, every object row by 33 columns and the state by 51, with a
/// pairing relation.
struct V9DecisionV1 {
    base: GradientDecisionV1,
    state: Vec<f32>,
    object_features: Vec<f32>,
    action_features: Vec<f32>,
    pairs: Vec<i64>,
}

impl V9DecisionV1 {
    fn new_v1(
        base: GradientDecisionV1,
        include_rows: &[usize],
        pairs: &[i64],
        offset: usize,
    ) -> Self {
        let mut action_features = Vec::new();
        for (row, features) in base
            .action_features
            .chunks_exact(ACTION_FEATURE_DIM_V1)
            .enumerate()
        {
            action_features.extend_from_slice(features);
            action_features.push(f32::from(u8::from(include_rows.contains(&row))));
        }
        let mut object_features = Vec::new();
        for (row, features) in base
            .object_features
            .chunks_exact(OBJECT_FEATURE_DIM_V1)
            .enumerate()
        {
            object_features.extend_from_slice(features);
            object_features.extend(
                (0..FLAT_V9_OBJECT_EXTRA_DIM_V1)
                    // Some rows keep zero columns, as non-public rows do.
                    .map(|column| {
                        if row % 3 == 2 {
                            0.0
                        } else {
                            column_v1(offset + row * 64 + column)
                        }
                    }),
            );
        }
        let mut state = base.state.clone();
        state.extend(
            (0..FLAT_V9_STATE_EXTRA_DIM_V1).map(|column| column_v1(offset + 7_000 + column)),
        );
        Self {
            base,
            state,
            object_features,
            action_features,
            pairs: pairs.to_vec(),
        }
    }

    fn view_v1(&self) -> NativeEncodedDecisionViewV1<'_> {
        let base = self.base.view_v1(false);
        NativeEncodedDecisionViewV1 {
            schema: schema_v9(),
            state: &self.state,
            object_features: &self.object_features,
            action_features: &self.action_features,
            ..base
        }
        .with_pairing_node_indices_v1(&self.pairs)
    }
}

fn v9_decisions_v1() -> [V9DecisionV1; 2] {
    let [first, second] = gradient_decisions_v1();
    [
        V9DecisionV1::new_v1(first, INCLUDE_ROWS_V1[0], PAIRS_V1[0], 11),
        V9DecisionV1::new_v1(second, INCLUDE_ROWS_V1[1], PAIRS_V1[1], 29),
    ]
}

/// A perturbed FlatV9 net, every coordinate live.
fn v9_model_v1(seed: u64) -> NativePolicyValueNetV1 {
    perturbed_dims_v1::<Net8DimsV1>(NativePolicyValueNetV1::v9_cube_template_v1().unwrap(), seed)
}

fn v9_recorded_v1(
    model: &NativePolicyValueNetV1,
    decisions: &[V9DecisionV1; 2],
) -> [RecordedV1; 2] {
    decisions.each_ref().map(|decision| {
        recorded_v1(
            model
                .forward_feature_transfer_v9_mode(
                    decision.view_v1(),
                    ForwardActivationModeV1::LibmTanh,
                )
                .unwrap(),
        )
    })
}

fn v9_substeps_v1<'a>(
    decisions: &'a [V9DecisionV1; 2],
    recorded: &'a [RecordedV1; 2],
) -> Vec<Vec<NativePolicySubstepV1<'a>>> {
    GRADIENT_BATCH_V1
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|&(decision, selected)| NativePolicySubstepV1 {
                    forward: NativePolicyForwardInputV1::Encoded(Box::new(
                        decisions[decision].view_v1(),
                    )),
                    selected_action_index: selected,
                    expected_raw_action_logit_bits: &recorded[decision].logits,
                    expected_value_bits: recorded[decision].value,
                })
                .collect()
        })
        .collect()
}

/// One sequential GAE update of the FlatV9 gradient batch. The update
/// verifies every recorded logit and value bit against its tape forward.
fn v9_update_v1(state: &mut NativePolicyValueTrainStateV1) -> NativePolicyTrainStepResultV1 {
    let decisions = v9_decisions_v1();
    let recorded = v9_recorded_v1(state.model_v1(), &decisions);
    let steps = v9_substeps_v1(&decisions, &recorded);
    let groups = gradient_groups_v1(&steps);
    state
        .train_step_gae_feature_transfer_v4(
            &groups,
            &GRADIENT_TARGETS_V1,
            &GRADIENT_ADVANTAGES_V1,
            VC,
            LR,
        )
        .unwrap()
}

/// [`v9_update_v1`] over 100 groups through the profiled CPU route, with the
/// sequential or a fixed-partition backward.
fn v9_many_update_v1(
    state: &mut NativePolicyValueTrainStateV1,
    backward: Option<usize>,
) -> NativePolicyTrainStepResultV1 {
    let decisions = v9_decisions_v1();
    let recorded = v9_recorded_v1(state.model_v1(), &decisions);
    let one = v9_substeps_v1(&decisions, &recorded);
    let steps: Vec<Vec<NativePolicySubstepV1<'_>>> =
        (0..50).flat_map(|_| one.iter().cloned()).collect();
    let groups: Vec<NativePolicyPhysicalDecisionV1<'_>> = steps
        .iter()
        .zip(GRADIENT_TERMINAL_RETURNS_V1.iter().cycle())
        .map(
            |(substeps, &terminal_return)| NativePolicyPhysicalDecisionV1 {
                substeps,
                terminal_return,
                baseline_bits: 0,
            },
        )
        .collect();
    let cycled = |values: &[f32]| -> Vec<f32> {
        values.iter().cycle().take(groups.len()).copied().collect()
    };
    state
        .train_step_gae_feature_transfer_profiled_v1(
            FreshLineageGenerationV1::V4,
            &groups,
            &cycled(&GRADIENT_TARGETS_V1),
            &cycled(&GRADIENT_ADVANTAGES_V1),
            VC,
            LR,
            backward,
            1,
            &mut NativeTrainingPhaseRecorderV1::disabled_v1(),
        )
        .unwrap()
}

#[test]
fn v9_update_trains_every_tensor_and_keeps_the_v9_manifest() {
    let mut state = NativePolicyValueTrainStateV1::new_v1(v9_model_v1(0x09_0001)).unwrap();
    let before = state.model_v1().parameter_snapshot_v1();
    let result = v9_update_v1(&mut state);
    assert!(result.loss.is_finite());
    let after = state.model_v1().parameter_snapshot_v1();
    assert_eq!(after.len(), before.len());
    assert_eq!(
        after[OBJECT_FIRST_WEIGHT].shape,
        [HIDDEN_DIM_V1, V9_OBJECT_ENCODER_INPUT_V1]
    );
    assert_eq!(
        after[STATE_FIRST_WEIGHT].shape,
        [HIDDEN_DIM_V1, V9_STATE_ENCODER_INPUT_V1]
    );
    // The new columns move.
    let moved = |tensor: usize, columns: std::ops::Range<usize>, width: usize| {
        before[tensor]
            .values
            .iter()
            .zip(&after[tensor].values)
            .enumerate()
            .any(|(index, (a, b))| columns.contains(&(index % width)) && a != b)
    };
    assert!(moved(
        OBJECT_FIRST_WEIGHT,
        98..131,
        V9_OBJECT_ENCODER_INPUT_V1
    ));
    assert!(moved(
        STATE_FIRST_WEIGHT,
        219..270,
        V9_STATE_ENCODER_INPUT_V1
    ));
    // A second update starts from the first's state.
    assert!(v9_update_v1(&mut state).loss.is_finite());
}

#[test]
fn v9_new_column_gradients_match_finite_differences() {
    let model = v9_model_v1(0x09FD_0001);
    let loss_and_gradients = |model: &NativePolicyValueNetV1| {
        let mut state = NativePolicyValueTrainStateV1::new_v1(model.clone()).unwrap();
        let result = v9_update_v1(&mut state);
        (f64::from(result.loss), result.gradients)
    };
    let (_, gradients) = loss_and_gradients(&model);
    let parameters = model.parameter_snapshot_v1();
    let step = 1.0e-2f32;
    let mut checked = 0;
    for (tensor, columns, width) in [
        (OBJECT_FIRST_WEIGHT, 98..131, V9_OBJECT_ENCODER_INPUT_V1),
        (STATE_FIRST_WEIGHT, 219..270, V9_STATE_ENCODER_INPUT_V1),
        // An old column beside the new ones: the embedding moved up.
        (OBJECT_FIRST_WEIGHT, 131..147, V9_OBJECT_ENCODER_INPUT_V1),
        (STATE_FIRST_WEIGHT, 270..1550, V9_STATE_ENCODER_INPUT_V1),
    ] {
        let mut order: Vec<usize> = (0..gradients[tensor].values.len())
            .filter(|&index| columns.contains(&(index % width)))
            .collect();
        order.sort_by(|&a, &b| {
            gradients[tensor].values[b]
                .abs()
                .total_cmp(&gradients[tensor].values[a].abs())
        });
        for &index in order.iter().take(3) {
            let analytic = f64::from(gradients[tensor].values[index]);
            assert!(
                analytic != 0.0,
                "tensor {tensor} index {index} has no gradient"
            );
            let loss_at = |delta: f32| {
                let mut shifted = parameters.clone();
                shifted[tensor].values[index] += delta;
                let mut model = model.clone();
                model.replace_parameter_snapshot_v1(&shifted).unwrap();
                loss_and_gradients(&model).0
            };
            let numeric = (loss_at(step) - loss_at(-step)) / (2.0 * f64::from(step));
            let tolerance = 0.05 * analytic.abs().max(numeric.abs()) + 2.0e-4;
            assert!(
                (analytic - numeric).abs() <= tolerance,
                "tensor {tensor} index {index}: analytic {analytic} numeric {numeric}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 12);
}

#[test]
fn v9_sequential_and_fixed_partition_backward_agree() {
    let model = v9_model_v1(0x09_0002);
    let mut sequential = NativePolicyValueTrainStateV1::new_v1(model.clone()).unwrap();
    let mut partitioned = NativePolicyValueTrainStateV1::new_v1(model).unwrap();
    let a = v9_many_update_v1(&mut sequential, None);
    let b = v9_many_update_v1(&mut partitioned, Some(4));
    assert_eq!(a.loss.to_bits(), b.loss.to_bits());
    for (left, right) in a.gradients.iter().zip(&b.gradients) {
        let close = left
            .values
            .iter()
            .zip(&right.values)
            .all(|(x, y)| (x - y).abs() <= 1.0e-5 * x.abs().max(y.abs()).max(1.0));
        assert!(close, "{}", left.name);
    }
}

#[test]
fn v9_and_v8_nets_refuse_each_others_inputs() {
    let v9 = v9_model_v1(0x09_0003);
    let v8 = NativePolicyValueNetV1::v8_inclusion_template_v1().unwrap();
    let decisions = v9_decisions_v1();
    let v9_view = decisions[0].view_v1();
    assert!(v9
        .forward_feature_transfer_v9_mode(v9_view, ForwardActivationModeV1::LibmTanh)
        .is_ok());
    assert!(v8
        .forward_feature_transfer_v8_mode(v9_view, ForwardActivationModeV1::LibmTanh)
        .is_err());
    // A FlatV8 view (V8 schema, V8 widths) to the FlatV9 net.
    let base = decisions[0].base.view_v1(false);
    let v8_view = NativeEncodedDecisionViewV1 {
        schema: schema_v8(),
        action_features: &decisions[0].action_features,
        ..base
    };
    assert!(v9
        .forward_feature_transfer_v9_mode(v8_view, ForwardActivationModeV1::LibmTanh)
        .is_err());
    assert!(v8
        .forward_feature_transfer_v9_mode(v8_view, ForwardActivationModeV1::LibmTanh)
        .is_err());
}
