// RUN: mlir-opt -resolve-shaped-type-result-dims -split-input-file %s | FileCheck %s

func @insert_slice(
    %arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>,
    %arg2 : index, %arg3 : index, %arg4 : index) -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %0 = tensor.insert_slice %arg0 into %arg1[%arg2, %arg3, %arg4] [%d0, %d1, %d2] [1, 1, 1] : tensor<?x?x?xf32> into tensor<?x?x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?x?xf32>
  %2 = tensor.dim %0, %c1 : tensor<?x?x?xf32>
  %3 = tensor.dim %0, %c2 : tensor<?x?x?xf32>
  return %1, %2, %3 : index, index, index
}
// CHECK-LABEL: func @insert_slice(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//       CHECK:   return %[[D0]], %[[D1]], %[[D2]]

// -----

func @extract_slice(%arg0 : tensor<?x?x?xf32>, %arg1 : index, %arg2 : index,
    %arg3 : index) -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.extract_slice %arg0[0, 0, 0] [%arg1, %arg2, %arg3] [1, 1, 1] :
      tensor<?x?x?xf32> to tensor<?x?x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?x?xf32>
  %2 = tensor.dim %0, %c1 : tensor<?x?x?xf32>
  %3 = tensor.dim %0, %c2 : tensor<?x?x?xf32>
  return %1, %2, %3 : index, index, index
}
// CHECK-LABEL: func @extract_slice(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]]: index
//       CHECK:   return %[[ARG1]], %[[ARG2]], %[[ARG3]]

// -----

func @extract_slice_rank_reduced_1(%arg0 : tensor<?x?x?xf32>,
    %arg1 : index) -> index {
  %c0 = arith.constant 0 : index
  %0 = tensor.extract_slice %arg0[0, 0, 0] [1, %arg1, 1] [1, 1, 1] :
     tensor<?x?x?xf32> to tensor<?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?xf32>
  return %1 : index
}
// CHECK-LABEL: func @extract_slice_rank_reduced_1(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//       CHECK:   return %[[ARG1]]

// -----

func @extract_slice_rank_reduced_2(%arg0 : tensor<?x?x?xf32>,
    %arg1 : index) -> index {
  %c0 = arith.constant 0 : index
  %0 = tensor.extract_slice %arg0[0, 0, 0] [1, %arg1, 1] [1, 1, 1] :
     tensor<?x?x?xf32> to tensor<?x1xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x1xf32>
  return %1 : index
}
// CHECK-LABEL: func @extract_slice_rank_reduced_2(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//       CHECK:   return %[[ARG1]]

// -----

func @extract_slice_rank_reduced_3(%arg0 : tensor<?x?x?xf32>,
    %arg1 : index) -> index {
  %c1 = arith.constant 1 : index
  %0 = tensor.extract_slice %arg0[0, 0, 0] [1, %arg1, 1] [1, 1, 1] :
     tensor<?x?x?xf32> to tensor<1x?xf32>
  %1 = tensor.dim %0, %c1 : tensor<1x?xf32>
  return %1 : index
}
// CHECK-LABEL: func @extract_slice_rank_reduced_3(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//       CHECK:   return %[[ARG1]]

// -----

func @extract_slice_rank_reduced_4(%arg0 : tensor<?x?x?xf32>,
    %arg1 : index) -> index {
  %c1 = arith.constant 1 : index
  %0 = tensor.extract_slice %arg0[0, 0, 0] [1, %arg1, 1] [1, 1, 1] :
     tensor<?x?x?xf32> to tensor<1x?x1xf32>
  %1 = tensor.dim %0, %c1 : tensor<1x?x1xf32>
  return %1 : index
}
// CHECK-LABEL: func @extract_slice_rank_reduced_4(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//       CHECK:   return %[[ARG1]]

// -----

func @extract_slice_rank_reduced_5(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
    %arg2 : index) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.extract_slice %arg0[0, 0, 0] [%arg1, 1, %arg2] [1, 1, 1] :
     tensor<?x?x?xf32> to tensor<?x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %2 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %1, %2 : index, index
}
// CHECK-LABEL: func @extract_slice_rank_reduced_5(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: index
//       CHECK:   return %[[ARG1]], %[[ARG2]]

// -----

func @extract_slice_rank_reduced_6(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
    %arg2 : index) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.extract_slice %arg0[0, 0, 0] [%arg1, 1, %arg2] [1, 1, 1] :
     tensor<?x?x?xf32> to tensor<?x1x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x1x?xf32>
  %2 = tensor.dim %0, %c2 : tensor<?x1x?xf32>
  return %1, %2 : index, index
}
// CHECK-LABEL: func @extract_slice_rank_reduced_6(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9_]+]]: index
//       CHECK:   return %[[ARG1]], %[[ARG2]]

// -----

func @pad_only_high_pad(%tensor: tensor<1x224x224x3xf32>, %arg0: index, %arg1: index) -> (index, index) {
  %f0 = arith.constant 0.0 : f32
  %0 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
  %1 = affine.min affine_map<(d0) -> (d0 * 2 + 3, 224)>(%arg0)
  %2 = affine.apply affine_map<(d0, d1) -> (d0 - d1 * 2)>(%1, %arg0)
  %3 = affine.apply affine_map<(d0, d1) -> (-d0 + d1 * 2 + 3)>(%1, %arg0)
  %4 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
  %5 = affine.min affine_map<(d0) -> (d0 * 2 + 9, 224)>(%arg1)
  %6 = affine.apply affine_map<(d0, d1) -> (d0 - d1 * 2)>(%5, %arg1)
  %7 = affine.apply affine_map<(d0, d1) -> (-d0 + d1 * 2 + 9)>(%5, %arg1)
  %8 = tensor.extract_slice %tensor[0, %0, %4, 0][1, %2, %6, 3][1, 1, 1, 1] : tensor<1x224x224x3xf32> to tensor<1x?x?x3xf32>

  // Dim#1: %2 (source) + %3 (high pad) = (%1 - %arg0 * 2) + (-%1 + %arg0 * 2 + 3) = 3
  // Dim#2: %6 (source) + %7 (high pad) = (%5 - %arg1 * 2) + (-%5 + %arg1 * 2 + 9) = 9
  %pad = tensor.pad %8 low[0, 0, 0, 0] high[0, %3, %7, 0]  {
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %f0 : f32
  } : tensor<1x?x?x3xf32> to tensor<1x?x?x3xf32>

  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim1 = tensor.dim %pad, %c1 : tensor<1x?x?x3xf32>
  %dim2 = tensor.dim %pad, %c2 : tensor<1x?x?x3xf32>
  return %dim1, %dim2 : index, index
}

// CHECK-LABEL: func @pad_only_high_pad
//       CHECK:   %[[C3:.+]] = arith.constant 3 : index
//       CHECK:   %[[C9:.+]] = arith.constant 9 : index
//       CHECK:   return %[[C3]], %[[C9]]

// -----

func @pad_both_low_and_high_pad(%tensor: tensor<1x56x56x144xf32>, %arg0: index, %arg1: index, %arg2: index) -> (index, index) {
  %f0 = arith.constant 0.0 : f32
  %0 = affine.max affine_map<(d0) -> (0, -d0 + 1)>(%arg0)
  %1 = affine.max affine_map<(d0) -> (d0 - 1, 0)>(%arg0)
  %2 = affine.min affine_map<(d0) -> (d0, 56)>(%1)
  %3 = affine.max affine_map<(d0) -> (d0 + 3, 0)>(%arg0)
  %4 = affine.min affine_map<(d0) -> (d0, 56)>(%3)
  %5 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%4, %2)
  %6 = affine.apply affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 4)>(%0, %4, %2)
  %7 = affine.max affine_map<(d0) -> (0, -d0 + 1)>(%arg1)
  %8 = affine.max affine_map<(d0) -> (d0 - 1, 0)>(%arg1)
  %9 = affine.min affine_map<(d0) -> (d0, 56)>(%8)
  %10 = affine.max affine_map<(d0) -> (d0 + 3, 0)>(%arg1)
  %11 = affine.min affine_map<(d0) -> (d0, 56)>(%10)
  %12 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%11, %9)
  %13 = affine.apply affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 4)>(%7, %11, %9)
  %14 = tensor.extract_slice %tensor[0, %2, %9, %arg2][1, %5, %12, 16][1, 1, 1, 1] : tensor<1x56x56x144xf32> to tensor<1x?x?x16xf32>

  // Dim#1: %0 (low pad) + %5  (source) + %6  (high pad) = %0 + (%4 - %2) +  (-%0 - %4 + %2 + 4)  = 4
  // Dim#1: %7 (low pad) + %12 (source) + %13 (high pad) = %7 + (%11 - %9) + (-%7 - %11 + %9 + 4) = 4
  %pad = tensor.pad %14 low[0, %0, %7, 0] high[0, %6, %13, 0]  {
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):  // no predecessors
    tensor.yield %f0 : f32
  } : tensor<1x?x?x16xf32> to tensor<1x?x?x16xf32>

  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim1 = tensor.dim %pad, %c1 : tensor<1x?x?x16xf32>
  %dim2 = tensor.dim %pad, %c2 : tensor<1x?x?x16xf32>
  return %dim1, %dim2 : index, index
}

// CHECK-LABEL: func @pad_both_low_and_high_pad
//       CHECK:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK:   return %[[C4]], %[[C4]]

// -----

func @extract_slice_same_rank(%arg0 : tensor<?x?xf32>,
    %arg1 : index, %arg2: index) -> (index, index) {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.extract_slice %arg0[0, 0] [%c1, %arg1] [1, 1] :
     tensor<?x?xf32> to tensor<?x?xf32>
  %pad = tensor.pad %0 low[0, 0] high[0, %arg2] {
  ^bb0(%arg3: index, %arg4: index):
    tensor.yield %f0 : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = tensor.dim %pad, %c0 : tensor<?x?xf32>
  %2 = tensor.dim %pad, %c1 : tensor<?x?xf32>
  return %1, %2 : index, index
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s1 + s0)>
//      CHECK: func @extract_slice_same_rank
// CHECK-SAME: %{{.+}}: tensor<?x?xf32>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index
//      CHECK:   %[[DIM0:.+]] = arith.constant 1 : index
//      CHECK:   %[[DIM1:.+]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG1]]]
//      CHECK:   return %[[DIM0]], %[[DIM1]]
