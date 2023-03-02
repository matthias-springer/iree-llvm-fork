// RUN: mlir-opt \
// RUN:     -test-strict-pattern-driver="strictness=AnyOp" \
// RUN:     --split-input-file %s | FileCheck %s --check-prefix=CHECK-AN

// RUN: mlir-opt \
// RUN:     -test-strict-pattern-driver="strictness=ExistingAndNewOps" \
// RUN:     --split-input-file %s | FileCheck %s --check-prefix=CHECK-EN

// RUN: mlir-opt \
// RUN:     -test-strict-pattern-driver="strictness=ExistingOps" \
// RUN:     --split-input-file %s | FileCheck %s --check-prefix=CHECK-EX

// CHECK-EN-LABEL: func @test_erase
//  CHECK-EN-SAME:     pattern_driver_all_erased = true, pattern_driver_changed = true}
//       CHECK-EN:   test.arg0
//       CHECK-EN:   test.arg1
//   CHECK-EN-NOT:   test.erase_op
func.func @test_erase() {
  %0 = "test.arg0"() : () -> (i32)
  %1 = "test.arg1"() : () -> (i32)
  %erase = "test.erase_op"(%0, %1) {worklist} : (i32, i32) -> (i32)
  return
}

// -----

// CHECK-EN-LABEL: func @test_insert_same_op
//  CHECK-EN-SAME:     {pattern_driver_all_erased = false, pattern_driver_changed = true}
//       CHECK-EN:   "test.insert_same_op"() {skip = true}
//       CHECK-EN:   "test.insert_same_op"() {skip = true}
func.func @test_insert_same_op() {
  %0 = "test.insert_same_op"() {worklist} : () -> (i32)
  return
}

// -----

// CHECK-EN-LABEL: func @test_replace_with_new_op
//  CHECK-EN-SAME:     {pattern_driver_all_erased = true, pattern_driver_changed = true}
//       CHECK-EN:   %[[n:.*]] = "test.new_op"
//       CHECK-EN:   "test.dummy_user"(%[[n]])
//       CHECK-EN:   "test.dummy_user"(%[[n]])
func.func @test_replace_with_new_op() {
  %0 = "test.replace_with_new_op"() {worklist} : () -> (i32)
  %1 = "test.dummy_user"(%0) : (i32) -> (i32)
  %2 = "test.dummy_user"(%0) : (i32) -> (i32)
  return
}

// -----

// CHECK-EN-LABEL: func @test_replace_with_erase_op
//  CHECK-EN-SAME:     {pattern_driver_all_erased = true, pattern_driver_changed = true}
//   CHECK-EN-NOT:   test.replace_with_new_op
//   CHECK-EN-NOT:   test.erase_op

// CHECK-EX-LABEL: func @test_replace_with_erase_op
//  CHECK-EX-SAME:     {pattern_driver_all_erased = true, pattern_driver_changed = true}
//   CHECK-EX-NOT:   test.replace_with_new_op
//       CHECK-EX:   test.erase_op
func.func @test_replace_with_erase_op() {
  "test.replace_with_new_op"() {create_erase_op, worklist} : () -> ()
  return
}

// -----

// CHECK-AN-LABEL: func @test_trigger_rewrite_through_block
//       CHECK-AN: "test.change_block_op"()[^[[BB0:.*]], ^[[BB0]]]
//       CHECK-AN: return
//       CHECK-AN: ^[[BB1:[^:]*]]:
//       CHECK-AN: "test.implicit_change_op"()[^[[BB1]]]
func.func @test_trigger_rewrite_through_block() {
  return
^bb1:
  // Uses bb1. ChangeBlockOp replaces that and all other usages of bb1 with bb2.
  "test.change_block_op"() [^bb1, ^bb2] {worklist} : () -> ()
^bb2:
  return
^bb3:
  // Also uses bb1. ChangeBlockOp replaces that usage with bb2. This triggers
  // this op being put on the worklist, which triggers ImplicitChangeOp, which,
  // in turn, replaces the successor with bb3.
  "test.implicit_change_op"() [^bb1] : () -> ()
}

// -----

// Make sure that "test.erase_op" is put on the worklist during mergeBlocks and
// subsequently deleted.

// CHECK-EN-LABEL: func @test_merge_blocks(
// CHECK-EX-LABEL: func @test_merge_blocks(
// CHECK-AN-LABEL: func @test_merge_blocks(
//       CHECK-AN:   "test.merge_blocks"() ({
//  CHECK-AN-NEXT:     "test.return"
//  CHECK-AN-NEXT:   }) : () -> i32
//  CHECK-AN-NEXT:   "test.return"
func.func @test_merge_blocks(%arg0: i32) -> () {
  %0 = "test.merge_blocks"() ({
  ^bb0:
    cf.br ^bb1 (%arg0: i32)
  ^bb1(%arg3 : i32):
    "test.erase_op"(%arg3) : (i32) -> ()
    "test.return"(%arg3) : (i32) -> ()
  }) {worklist} : () -> (i32)
  "test.return"(%0) : (i32) -> ()
}
