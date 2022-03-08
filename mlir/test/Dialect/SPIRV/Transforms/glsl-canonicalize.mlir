// RUN: mlir-opt -split-input-file -spirv-canonicalize-glsl %s | FileCheck %s

// CHECK-LABEL: func @clamp_fordlessthan
//  CHECK-SAME: (%[[INPUT:.*]]: f32, %[[MIN:.*]]: f32, %[[MAX:.*]]: f32)
func @clamp_fordlessthan(%input: f32, %min: f32, %max: f32) -> f32 {
  // CHECK: [[RES:%.*]] = spv.GLSL.FClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.FOrdLessThan %min, %input : f32
  %mid = spv.Select %0, %input, %min : i1, f32
  %1 = spv.FOrdLessThan %mid, %max : f32
  %2 = spv.Select %1, %mid, %max : i1, f32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : f32
}

// -----

// CHECK-LABEL: func @clamp_fordlessthan
//  CHECK-SAME: (%[[INPUT:.*]]: f32, %[[MIN:.*]]: f32, %[[MAX:.*]]: f32)
func @clamp_fordlessthan(%input: f32, %min: f32, %max: f32) -> f32 {
  // CHECK: [[RES:%.*]] = spv.GLSL.FClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.FOrdLessThan %input, %min : f32
  %mid = spv.Select %0, %min, %input : i1, f32
  %1 = spv.FOrdLessThan %max, %input : f32
  %2 = spv.Select %1, %max, %mid : i1, f32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : f32
}

// -----

// CHECK-LABEL: func @clamp_fordlessthanequal
//  CHECK-SAME: (%[[INPUT:.*]]: f32, %[[MIN:.*]]: f32, %[[MAX:.*]]: f32)
func @clamp_fordlessthanequal(%input: f32, %min: f32, %max: f32) -> f32 {
  // CHECK: [[RES:%.*]] = spv.GLSL.FClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.FOrdLessThanEqual %min, %input : f32
  %mid = spv.Select %0, %input, %min : i1, f32
  %1 = spv.FOrdLessThanEqual %mid, %max : f32
  %2 = spv.Select %1, %mid, %max : i1, f32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : f32
}

// -----

// CHECK-LABEL: func @clamp_fordlessthanequal
//  CHECK-SAME: (%[[INPUT:.*]]: f32, %[[MIN:.*]]: f32, %[[MAX:.*]]: f32)
func @clamp_fordlessthanequal(%input: f32, %min: f32, %max: f32) -> f32 {
  // CHECK: [[RES:%.*]] = spv.GLSL.FClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.FOrdLessThanEqual %input, %min : f32
  %mid = spv.Select %0, %min, %input : i1, f32
  %1 = spv.FOrdLessThanEqual %max, %input : f32
  %2 = spv.Select %1, %max, %mid : i1, f32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : f32
}

// -----

// CHECK-LABEL: func @clamp_slessthan
//  CHECK-SAME: (%[[INPUT:.*]]: si32, %[[MIN:.*]]: si32, %[[MAX:.*]]: si32)
func @clamp_slessthan(%input: si32, %min: si32, %max: si32) -> si32 {
  // CHECK: [[RES:%.*]] = spv.GLSL.SClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.SLessThan %min, %input : si32
  %mid = spv.Select %0, %input, %min : i1, si32
  %1 = spv.SLessThan %mid, %max : si32
  %2 = spv.Select %1, %mid, %max : i1, si32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : si32
}

// -----

// CHECK-LABEL: func @clamp_slessthan
//  CHECK-SAME: (%[[INPUT:.*]]: si32, %[[MIN:.*]]: si32, %[[MAX:.*]]: si32)
func @clamp_slessthan(%input: si32, %min: si32, %max: si32) -> si32 {
  // CHECK: [[RES:%.*]] = spv.GLSL.SClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.SLessThan %input, %min : si32
  %mid = spv.Select %0, %min, %input : i1, si32
  %1 = spv.SLessThan %max, %input : si32
  %2 = spv.Select %1, %max, %mid : i1, si32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : si32
}

// -----

// CHECK-LABEL: func @clamp_slessthanequal
//  CHECK-SAME: (%[[INPUT:.*]]: si32, %[[MIN:.*]]: si32, %[[MAX:.*]]: si32)
func @clamp_slessthanequal(%input: si32, %min: si32, %max: si32) -> si32 {
  // CHECK: [[RES:%.*]] = spv.GLSL.SClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.SLessThanEqual %min, %input : si32
  %mid = spv.Select %0, %input, %min : i1, si32
  %1 = spv.SLessThanEqual %mid, %max : si32
  %2 = spv.Select %1, %mid, %max : i1, si32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : si32
}

// -----

// CHECK-LABEL: func @clamp_slessthanequal
//  CHECK-SAME: (%[[INPUT:.*]]: si32, %[[MIN:.*]]: si32, %[[MAX:.*]]: si32)
func @clamp_slessthanequal(%input: si32, %min: si32, %max: si32) -> si32 {
  // CHECK: [[RES:%.*]] = spv.GLSL.SClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.SLessThanEqual %input, %min : si32
  %mid = spv.Select %0, %min, %input : i1, si32
  %1 = spv.SLessThanEqual %max, %input : si32
  %2 = spv.Select %1, %max, %mid : i1, si32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : si32
}

// -----

// CHECK-LABEL: func @clamp_ulessthan
//  CHECK-SAME: (%[[INPUT:.*]]: i32, %[[MIN:.*]]: i32, %[[MAX:.*]]: i32)
func @clamp_ulessthan(%input: i32, %min: i32, %max: i32) -> i32 {
  // CHECK: [[RES:%.*]] = spv.GLSL.UClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.ULessThan %min, %input : i32
  %mid = spv.Select %0, %input, %min : i1, i32
  %1 = spv.ULessThan %mid, %max : i32
  %2 = spv.Select %1, %mid, %max : i1, i32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : i32
}

// -----

// CHECK-LABEL: func @clamp_ulessthan
//  CHECK-SAME: (%[[INPUT:.*]]: i32, %[[MIN:.*]]: i32, %[[MAX:.*]]: i32)
func @clamp_ulessthan(%input: i32, %min: i32, %max: i32) -> i32 {
  // CHECK: [[RES:%.*]] = spv.GLSL.UClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.ULessThan %input, %min : i32
  %mid = spv.Select %0, %min, %input : i1, i32
  %1 = spv.ULessThan %max, %input : i32
  %2 = spv.Select %1, %max, %mid : i1, i32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : i32
}

// -----

// CHECK-LABEL: func @clamp_ulessthanequal
//  CHECK-SAME: (%[[INPUT:.*]]: i32, %[[MIN:.*]]: i32, %[[MAX:.*]]: i32)
func @clamp_ulessthanequal(%input: i32, %min: i32, %max: i32) -> i32 {
  // CHECK: [[RES:%.*]] = spv.GLSL.UClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.ULessThanEqual %min, %input : i32
  %mid = spv.Select %0, %input, %min : i1, i32
  %1 = spv.ULessThanEqual %mid, %max : i32
  %2 = spv.Select %1, %mid, %max : i1, i32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : i32
}

// -----

// CHECK-LABEL: func @clamp_ulessthanequal
//  CHECK-SAME: (%[[INPUT:.*]]: i32, %[[MIN:.*]]: i32, %[[MAX:.*]]: i32)
func @clamp_ulessthanequal(%input: i32, %min: i32, %max: i32) -> i32 {
  // CHECK: [[RES:%.*]] = spv.GLSL.UClamp %[[INPUT]], %[[MIN]], %[[MAX]]
  %0 = spv.ULessThanEqual %input, %min : i32
  %mid = spv.Select %0, %min, %input : i1, i32
  %1 = spv.ULessThanEqual %max, %input : i32
  %2 = spv.Select %1, %max, %mid : i1, i32

  // CHECK-NEXT: spv.ReturnValue [[RES]]
  spv.ReturnValue %2 : i32
}

// -----

// CHECK-LABEL: func @splat_fma
//  CHECK-SAME: (%[[A:.+]]: vector<3xf32>, %[[SCALAR:.+]]: f32, %[[C:.+]]: vector<3xf32>) -> vector<3xf32> {
func @splat_fma(%a : vector<3xf32>, %scalar: f32, %c: vector<3xf32>) -> vector<3xf32> {
  %b = spv.CompositeConstruct %scalar, %scalar, %scalar : vector<3xf32>
  // CHEKCK: %[[MUL:.+]] = spv.VectorTimesScalar %[[A]], %[[SCALAR]] : (vector<3xf32>, f32) -> vector<3xf32>
  // CHEKCK: %[[ADD:.+]] = spv.FAdd %[[MUL]], %[[C]] : vector<3xf32>
  %0 = spv.GLSL.Fma %a, %b, %c : vector<3xf32>
  // CHEKCK: return %[[ADD]]
  return %0 : vector<3xf32>
}

// -----

// CHECK-LABEL: func @splat_fma
func @splat_fma(%b : vector<3xf32>, %scalar: f32, %c: vector<3xf32>) -> vector<3xf32> {
  %a = spv.CompositeConstruct %scalar, %scalar, %scalar : vector<3xf32>
  // CHECK: spv.VectorTimesScalar
  // CHECK: spv.FAdd
  %0 = spv.GLSL.Fma %a, %b, %c : vector<3xf32>
  return %0 : vector<3xf32>
}

// -----

// CHECK-LABEL: func @splat_fma_no_contraction
func @splat_fma_no_contraction(%a : vector<3xf32>, %scalar: f32, %c: vector<3xf32>) -> vector<3xf32> {
  %b = spv.CompositeConstruct %scalar, %scalar, %scalar : vector<3xf32>
  // CHECK: spv.GLSL.Fma
  %0 = spv.GLSL.Fma %a, %b, %c {NoContraction} : vector<3xf32>
  return %0 : vector<3xf32>
}

// -----

// CHECK-LABEL: func @not_splat_fma
func @not_splat_fma(%a : vector<3xf32>, %s0: f32, %s1: f32, %c: vector<3xf32>) -> vector<3xf32> {
  %b = spv.CompositeConstruct %s1, %s0, %s1 : vector<3xf32>
  // CHECK: spv.GLSL.Fma
  %0 = spv.GLSL.Fma %a, %b, %c : vector<3xf32>
  return %0 : vector<3xf32>
}
