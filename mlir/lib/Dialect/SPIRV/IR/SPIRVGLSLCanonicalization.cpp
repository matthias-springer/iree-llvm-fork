//===- SPIRVGLSLCanonicalization.cpp - SPIR-V GLSL canonicalization patterns =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the canonicalization patterns for SPIR-V GLSL-specific ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVGLSLCanonicalization.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

/// A pattern to convert spv.GLSL.Fma with splat vectors into
/// spv.VectorTimesScalar and spv.FAdd to save vector registers.
struct ConvertSplatFma final : public OpRewritePattern<spirv::GLSLFmaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::GLSLFmaOp fmaOp,
                                PatternRewriter &rewriter) const override {
    // If there is `NoContraction` decoration, we cannot break the FMA up.
    auto attr = spirv::stringifyDecoration(spirv::Decoration::NoContraction);
    if (fmaOp->getAttrOfType<UnitAttr>(attr))
      return failure();

    auto splatOp = fmaOp.x().getDefiningOp<spirv::CompositeConstructOp>();
    auto vectorOp = fmaOp.y();
    if (!splatOp || !llvm::is_splat(splatOp.getOperands())) {
      splatOp = fmaOp.y().getDefiningOp<spirv::CompositeConstructOp>();
      vectorOp = fmaOp.x();
    }
    if (!splatOp || !llvm::is_splat(splatOp.getOperands()))
      return failure();

    auto mulOp = rewriter.create<spirv::VectorTimesScalarOp>(
        fmaOp.getLoc(), splatOp.getType(), vectorOp, splatOp.getOperand(0));
    rewriter.replaceOpWithNewOp<spirv::FAddOp>(fmaOp, mulOp, fmaOp.z());
    return success();
  }
};

#include "SPIRVCanonicalization.inc"

} // namespace

namespace mlir {
namespace spirv {
void populateSPIRVGLSLCanonicalizationPatterns(RewritePatternSet &results) {
  results
      .add<ConvertComparisonIntoClamp1_SPV_FOrdLessThanOp,
           ConvertComparisonIntoClamp1_SPV_FOrdLessThanEqualOp,
           ConvertComparisonIntoClamp1_SPV_SLessThanOp,
           ConvertComparisonIntoClamp1_SPV_SLessThanEqualOp,
           ConvertComparisonIntoClamp1_SPV_ULessThanOp,
           ConvertComparisonIntoClamp1_SPV_ULessThanEqualOp,
           ConvertComparisonIntoClamp2_SPV_FOrdLessThanOp,
           ConvertComparisonIntoClamp2_SPV_FOrdLessThanEqualOp,
           ConvertComparisonIntoClamp2_SPV_SLessThanOp,
           ConvertComparisonIntoClamp2_SPV_SLessThanEqualOp,
           ConvertComparisonIntoClamp2_SPV_ULessThanOp,
           ConvertComparisonIntoClamp2_SPV_ULessThanEqualOp, ConvertSplatFma>(
          results.getContext());
}
} // namespace spirv
} // namespace mlir
