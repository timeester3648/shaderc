// Copyright 2015 The Shaderc Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libshaderc_util/compiler.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>
#include <vector>

#include "SPIRV/GlslangToSpv.h"
#include "libshaderc_util/format.h"
#include "libshaderc_util/io_shaderc.h"
#include "libshaderc_util/message.h"
#include "libshaderc_util/resources.h"
#include "libshaderc_util/shader_stage.h"
#include "libshaderc_util/spirv_tools_wrapper.h"
#include "libshaderc_util/string_piece.h"
#include "libshaderc_util/version_profile.h"
#include "spirv-tools/libspirv.hpp"

#include "glslang/Include/Common.h"
#include "glslang/Include/InfoSink.h"
#include "glslang/Include/Types.h"
#include "glslang/Include/intermediate.h" 

#include "glslang/MachineIndependent/iomapper.h"
#include "glslang/MachineIndependent/LiveTraverser.h"
#include "glslang/MachineIndependent/SymbolTable.h"

#include "../../libshaderc/include/shaderc/shaderc_ext.h"

namespace glslang {
 // Note: until they make it public, so also need to keep definition in sync
struct TVarEntryInfo {
  long long id;
  TIntermSymbol* symbol;
  bool live;
  TLayoutPacking
      upgradedToPushConstantPacking;  // ElpNone means it hasn't been upgraded
  int newBinding;
  int newSet;
  int newLocation;
  int newComponent;
  int newIndex;
  EShLanguage stage;

  void clearNewAssignments() {
    upgradedToPushConstantPacking = ElpNone;
    newBinding = -1;
    newSet = -1;
    newLocation = -1;
    newComponent = -1;
    newIndex = -1;
  }

  struct TOrderById {
    inline bool operator()(const TVarEntryInfo& l, const TVarEntryInfo& r) {
      return l.id < r.id;
    }
  };

  struct TOrderByPriority {
    // ordering:
    // 1) has both binding and set
    // 2) has binding but no set
    // 3) has no binding but set
    // 4) has no binding and no set
    inline bool operator()(const TVarEntryInfo& l, const TVarEntryInfo& r) {
      const TQualifier& lq = l.symbol->getQualifier();
      const TQualifier& rq = r.symbol->getQualifier();

      // simple rules:
      // has binding gives 2 points
      // has set gives 1 point
      // who has the most points is more important.
      int lPoints = (lq.hasBinding() ? 2 : 0) + (lq.hasSet() ? 1 : 0);
      int rPoints = (rq.hasBinding() ? 2 : 0) + (rq.hasSet() ? 1 : 0);

      if (lPoints == rPoints) return l.id < r.id;
      return lPoints > rPoints;
    }
  };

  struct TOrderByPriorityAndLive {
    // ordering:
    // 1) do live variables first
    // 2) has both binding and set
    // 3) has binding but no set
    // 4) has no binding but set
    // 5) has no binding and no set
    inline bool operator()(const TVarEntryInfo& l, const TVarEntryInfo& r) {
      const TQualifier& lq = l.symbol->getQualifier();
      const TQualifier& rq = r.symbol->getQualifier();

      // simple rules:
      // has binding gives 2 points
      // has set gives 1 point
      // who has the most points is more important.
      int lPoints = (lq.hasBinding() ? 2 : 0) + (lq.hasSet() ? 1 : 0);
      int rPoints = (rq.hasBinding() ? 2 : 0) + (rq.hasSet() ? 1 : 0);

      if (l.live != r.live) return l.live > r.live;

      if (lPoints != rPoints) return lPoints > rPoints;

      return l.id < r.id;
    }
  };
};
}

namespace {
using shaderc_util::string_piece;

constexpr const char* kLineDirectivePrefixCstr = "#line ";
constexpr string_piece kLineDirectivePrefix(kLineDirectivePrefixCstr,
                                            kLineDirectivePrefixCstr + 6);

// For use with glslang parsing calls.
const bool kNotForwardCompatible = false;

// Returns true if #line directive sets the line number for the next line in the
// given version and profile.
inline bool LineDirectiveIsForNextLine(int version, EProfile profile) {
  return profile == EEsProfile || version >= 330;
}

// Returns a #line directive whose arguments are line and filename.
inline std::string GetLineDirective(int line, const string_piece& filename) {
  return kLineDirectivePrefix.str() + std::to_string(line) + " \"" +
         filename.str() + "\"\n";
}

// Given a canonicalized #line directive (starting exactly with "#line", using
// single spaces to separate different components, and having an optional
// newline at the end), returns the line number and string name/number. If no
// string name/number is provided, the second element in the returned pair is an
// empty string_piece. Behavior is undefined if the directive parameter is not a
// canonicalized #line directive.
std::pair<int, string_piece> DecodeLineDirective(string_piece directive) {
  assert(directive.starts_with(kLineDirectivePrefix));
  directive = directive.substr(kLineDirectivePrefix.size());

  const int line = std::atoi(directive.data());
  const size_t space_loc = directive.find_first_of(' ');
  if (space_loc == string_piece::npos) return std::make_pair(line, "");

  directive = directive.substr(space_loc);
  directive = directive.strip("\" \n");
  return std::make_pair(line, directive);
}

// Returns the Glslang message rules for the given target environment,
// source language, and whether we want HLSL offset rules.  We assume
// only valid combinations are used.
EShMessages GetMessageRules(shaderc_util::Compiler::TargetEnv env,
                            shaderc_util::Compiler::SourceLanguage lang,
                            bool hlsl_offsets, bool hlsl_16bit_types,
                            bool debug_info) {
  using shaderc_util::Compiler;
  EShMessages result = EShMsgCascadingErrors;
  if (lang == Compiler::SourceLanguage::HLSL) {
    result = static_cast<EShMessages>(result | EShMsgReadHlsl);
  }
  switch (env) {
    case Compiler::TargetEnv::OpenGLCompat:
      // The compiler will have already errored out before now.
      // But we need to handle this enum.
      break;
    case Compiler::TargetEnv::OpenGL:
      result = static_cast<EShMessages>(result | EShMsgSpvRules);
      break;
    case Compiler::TargetEnv::Vulkan:
      result =
          static_cast<EShMessages>(result | EShMsgSpvRules | EShMsgVulkanRules);
      break;
  }
  if (hlsl_offsets) {
    result = static_cast<EShMessages>(result | EShMsgHlslOffsets);
  }
  if (hlsl_16bit_types) {
    result = static_cast<EShMessages>(result | EShMsgHlslEnable16BitTypes);
  }
  if (debug_info) {
    result = static_cast<EShMessages>(result | EShMsgDebugInfo);
  }
  return result;
}

// Forces the shaderc-generated default (global) uniform block to std430
// packing and reorders its members to minimize padding.
//
// Algorithm (greedy best-fit bin packing):
//  1. Compute each field's (alignment, size) under std430 rules via
//     TIntermediate::getBaseAlignment.
//  2. Process fields from largest alignment to smallest (ties: larger size
//     first, then name, for determinism). Largest-alignment fields go first
//     since they have zero packing flexibility anyway -- they must sit at a
//     multiple of their own alignment and nothing can be done to shrink that.
//  3. For each field, look for the smallest already-open padding gap it fits
//     in (best-fit, so bigger gaps stay available for bigger later fields).
//     If none fits, append it at the current end of the block, which may
//     itself open a new gap due to alignment rounding.
//  4. The final member order is the order fields ended up placed at
//     (by offset), which is what actually determines the padding once
//     glslang assigns real offsets sequentially over this reordered list.
class GlobalUniformBlockOptimizer : public glslang::TIntermTraverser {
 public:
  explicit GlobalUniformBlockOptimizer(const glslang::TIntermediate& intermediate)
      : intermediate_(intermediate) {}

  void visitSymbol(glslang::TIntermSymbol* symbol) override {
    if (symbol->getAccessName() != SHADERC_EXT_DEFAULT_UNIFORM_BLOCK_NAME) {
      return;
    }

    glslang::TType& blockType = symbol->getWritableType();
    blockType.getQualifier().layoutPacking = glslang::ElpStd430;

    auto fields = blockType.getWritableStruct();
    if (!fields || fields->size() < 2) return;

    if (!already_reordered_) {
      Repack(*fields);
      already_reordered_ = true;
    }

    fields->clear();
    for (const auto& loc : reordered_) fields->push_back(loc);
  }

  // old-declaration-index -> new-index, valid once the full tree has been
  // traversed at least once (i.e. after root->traverse(&optimizer) returns).
  const std::vector<int>& indexMap() const { return old_to_new_; }

 private:
  struct Candidate {
    const glslang::TTypeLoc* loc;
    int alignment;
    int size;
    int original_index;
  };

  struct Gap {
    int offset;
    int size;
  };

  void Repack(const glslang::TTypeList& original) {
    std::vector<Candidate> candidates;
    candidates.reserve(original.size());
    for (size_t i = 0; i < original.size(); ++i) {
      int size = 0;
      int stride = 0;
      const int alignment = intermediate_.getBaseAlignment(
          *original[i].type, size, stride, glslang::ElpStd430,
          /*rowMajor=*/false);
      candidates.push_back({&original[i], alignment, size, static_cast<int>(i)});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                if (a.alignment != b.alignment) return a.alignment > b.alignment;
                if (a.size != b.size) return a.size > b.size;
                return a.loc->type->getFieldName() < b.loc->type->getFieldName();
              });

    std::vector<Gap> gaps;
    struct Placed {
      int offset;
      const glslang::TTypeLoc* loc;
      int original_index;
    };
    std::vector<Placed> placed;
    placed.reserve(candidates.size());
    int cursor = 0;

    for (const Candidate& field : candidates) {
      int best_gap = -1;
      int best_gap_aligned_start = 0;
      for (size_t i = 0; i < gaps.size(); ++i) {
        const int aligned_start =
            (gaps[i].offset + field.alignment - 1) / field.alignment *
            field.alignment;
        const int gap_end = gaps[i].offset + gaps[i].size;
        if (aligned_start + field.size > gap_end) continue;
        if (best_gap == -1 || gaps[i].size < gaps[best_gap].size) {
          best_gap = static_cast<int>(i);
          best_gap_aligned_start = aligned_start;
        }
      }

      int placement_offset;
      if (best_gap != -1) {
        const Gap gap = gaps[best_gap];
        gaps.erase(gaps.begin() + best_gap);
        placement_offset = best_gap_aligned_start;

        const int leading = best_gap_aligned_start - gap.offset;
        const int trailing =
            (gap.offset + gap.size) - (best_gap_aligned_start + field.size);
        if (leading > 0) gaps.push_back({gap.offset, leading});
        if (trailing > 0)
          gaps.push_back({best_gap_aligned_start + field.size, trailing});
      } else {
        const int aligned_start =
            (cursor + field.alignment - 1) / field.alignment * field.alignment;
        if (aligned_start > cursor)
          gaps.push_back({cursor, aligned_start - cursor});
        placement_offset = aligned_start;
        cursor = aligned_start + field.size;
      }

      placed.push_back({placement_offset, field.loc, field.original_index});
    }

    std::sort(placed.begin(), placed.end(),
              [](const Placed& a, const Placed& b) { return a.offset < b.offset; });

    reordered_.clear();
    reordered_.reserve(placed.size());
    old_to_new_.assign(placed.size(), -1);
    for (size_t new_index = 0; new_index < placed.size(); ++new_index) {
      reordered_.push_back(*placed[new_index].loc);
      old_to_new_[placed[new_index].original_index] = static_cast<int>(new_index);
    }
  }

  bool already_reordered_ = false;
  std::vector<glslang::TTypeLoc> reordered_;
  std::vector<int> old_to_new_;
  const glslang::TIntermediate& intermediate_;
};

// Second pass: rewrites every existing access into the default uniform
// block so its baked-in member index matches the NEW post-reorder layout.
// Required because EOpIndexDirectStruct indices are set at parse time,
// long before GlobalUniformBlockOptimizer runs.
class GlobalUniformBlockIndexRemapper : public glslang::TIntermTraverser {
 public:
  explicit GlobalUniformBlockIndexRemapper(const std::vector<int>& old_to_new)
      : old_to_new_(old_to_new) {}

  bool visitBinary(glslang::TVisit, glslang::TIntermBinary* node) override {
    if (node->getOp() != glslang::EOpIndexDirectStruct) return true;

    glslang::TIntermSymbol* leftSym = node->getLeft()->getAsSymbolNode();
    if (!leftSym ||
        leftSym->getAccessName() != SHADERC_EXT_DEFAULT_UNIFORM_BLOCK_NAME)
      return true;

    glslang::TIntermConstantUnion* idxNode = node->getRight()->getAsConstantUnion();
    if (!idxNode) return true;

    const int old_index = idxNode->getConstArray()[0].getIConst();
    if (old_index < 0 || old_index >= static_cast<int>(old_to_new_.size()))
      return true;
    const int new_index = old_to_new_[old_index];
    if (new_index == old_index) return true;

    // Rebuild rather than mutate in place: getConstArray() commonly returns
    // a const reference, so this is the portable way to change the value.
    glslang::TConstUnionArray newArray(1);
    newArray[0].setIConst(new_index);
    glslang::TIntermConstantUnion* newIdx =
        new glslang::TIntermConstantUnion(newArray, idxNode->getType());
    newIdx->setLoc(idxNode->getLoc());
    node->setRight(newIdx);

    return true;
  }

 private:
  const std::vector<int>& old_to_new_;
};

// Walks the whole tree (live + dead code) collecting every uniform/buffer
// resource declaration exactly once, regardless of traversal order.
class UniformResourceCollector : public glslang::TIntermTraverser {
 public:
  struct Entry {
    glslang::TString name;
    const glslang::TType* type;
    bool hasSet;
    int set;
    bool hasBinding;
    int binding;
  };

  void visitSymbol(glslang::TIntermSymbol* symbol) override {
    const glslang::TQualifier& q = symbol->getQualifier();
    if (!q.isUniformOrBuffer() || q.isPushConstant() || q.isShaderRecord())
      return;
    const glslang::TString& name = symbol->getAccessName();
    if (!seen_.insert(name).second) return;  // same resource seen before

    // The shaderc-generated default (global) uniform block is only given a
    // set/binding via Compiler::Compile's setGlobalUniformSet/Binding calls
    // so parsing succeeds. Treat it as if neither were user-explicit so it
    // gets folded into the normal "first unused set / first free binding"
    // allocation below, exactly like any other resource with no explicit
    // layout qualifiers.
    const bool is_default_uniform_block =
        name == SHADERC_EXT_DEFAULT_UNIFORM_BLOCK_NAME;

    Entry e;
    e.name = name;
    e.type = &symbol->getType();
    e.hasSet = is_default_uniform_block ? false : q.hasSet();
    e.set = e.hasSet ? q.layoutSet : 0;
    e.hasBinding = is_default_uniform_block ? false : q.hasBinding();
    e.binding = e.hasBinding ? q.layoutBinding : 0;
    entries.push_back(e);
  }

  std::vector<Entry> entries;

 private:
  std::set<glslang::TString> seen_;
};

// Custom binding/set resolver used when auto-map-locations is enabled.
//
// Rules:
//  * set + binding both explicit -> kept as-is
//  * set only                    -> first free binding *within that set*
//  * binding only                -> lands in the first set index with no
//                                    explicit `set` anywhere in the shader
//  * neither                     -> same "first undeclared set", first
//                                    free binding in it
//
// Everything is precomputed once in the constructor from a name-sorted
// (then resource-type-sorted) list of declarations, so the result is
// completely independent of declaration/include order.
class ShadercAutoBindResolver : public glslang::TDefaultIoResolverBase {
 public:
  explicit ShadercAutoBindResolver(const glslang::TIntermediate& intermediate)
      : glslang::TDefaultIoResolverBase(intermediate) {
    UniformResourceCollector collector;
    TIntermNode* root = intermediate.getTreeRoot();
    if (root) root->traverse(&collector);

    // Deterministic order: name first, resource type as tiebreak.
    std::sort(collector.entries.begin(), collector.entries.end(),
              [this](const UniformResourceCollector::Entry& a,
                     const UniformResourceCollector::Entry& b) {
                if (a.name != b.name) return a.name < b.name;
                return getResourceType(*a.type) < getResourceType(*b.type);
              });

    // Pass 1: find every explicitly-declared set -> pick the catch-all.
    for (const auto& e : collector.entries) {
      if (e.hasSet) explicit_sets_.insert(e.set);
    }
    int candidate = 0;
    while (explicit_sets_.count(candidate)) ++candidate;
    catch_all_set_ = candidate;

    const EShLanguage stage = intermediate.getStage();

    // Pass 2: reserve everything with an explicit binding first (both the
    // "set+binding" and "binding only" cases), exactly as written.
    for (const auto& e : collector.entries) {
      if (!e.hasBinding) continue;
      const int set = e.hasSet ? e.set : catch_all_set_;
      const glslang::TResourceType resource = getResourceType(*e.type);
      const int resourceKey =
          referenceIntermediate.getBindingsPerResourceType() ? resource : 0;
      const int numBindings =
          e.type->isSizedArray() ? e.type->getCumulativeArraySize() : 1;
      const int binding = getBaseBinding(stage, resource, set) + e.binding;
      resolved_[e.name] = {set, binding};
      reserveSlot(resourceKey, set, binding, numBindings);
    }

    // Pass 3: fill "set only" / "neither" into the first free hole, in the
    // deterministic name-sorted order above.
    for (const auto& e : collector.entries) {
      if (e.hasBinding) continue;
      const int set = e.hasSet ? e.set : catch_all_set_;
      const glslang::TResourceType resource = getResourceType(*e.type);
      const int resourceKey =
          referenceIntermediate.getBindingsPerResourceType() ? resource : 0;
      const int numBindings =
          e.type->isSizedArray() ? e.type->getCumulativeArraySize() : 1;
      const int base = getBaseBinding(stage, resource, set);
      const int binding = getFreeSlot(resourceKey, set, base, numBindings);
      resolved_[e.name] = {set, binding};
    }
  }

  bool validateBinding(EShLanguage, glslang::TVarEntryInfo&) override {
    return true;
  }

  glslang::TResourceType getResourceType(const glslang::TType& type) override {
    if (isImageType(type)) return glslang::EResImage;
    if (isTextureType(type)) return glslang::EResTexture;
    if (isSsboType(type)) return glslang::EResSsbo;
    if (isSamplerType(type)) return glslang::EResSampler;
    if (isUboType(type)) return glslang::EResUbo;
    if (isCombinedSamplerType(type)) return glslang::EResCombinedSampler;
    if (isAsType(type)) return glslang::EResAs;
    if (isTensorType(type)) return glslang::EResTensor;
    return glslang::EResCount;
  }

  int resolveSet(EShLanguage, glslang::TVarEntryInfo& ent) override {
    auto it = resolved_.find(ent.symbol->getAccessName());
    return ent.newSet = (it == resolved_.end() ? 0 : it->second.first);
  }

  int resolveBinding(EShLanguage, glslang::TVarEntryInfo& ent) override {
    auto it = resolved_.find(ent.symbol->getAccessName());
    if (it == resolved_.end()) return ent.newBinding = -1;

    const glslang::TType& type = ent.symbol->getType();
    if (type.getQualifier().hasBinding()) {
      return ent.newBinding = it->second.second;  // explicit: always applied
    }
    if (!ent.live || !doAutoBindingMapping()) {
      return ent.newBinding = -1;
    }
    return ent.newBinding = it->second.second;
  }

 private:
  std::set<int> explicit_sets_;
  int catch_all_set_ = 0;
  std::map<glslang::TString, std::pair<int, int>> resolved_;
};


}  // anonymous namespace

namespace shaderc_util {

unsigned int GlslangInitializer::initialize_count_ = 0;
std::mutex* GlslangInitializer::glslang_mutex_ = nullptr;

GlslangInitializer::GlslangInitializer() {
  static std::mutex first_call_mutex;

  // If this is the first call, glslang_mutex_ needs to be created, but in
  // thread safe manner.
  {
    const std::lock_guard<std::mutex> first_call_lock(first_call_mutex);
    if (glslang_mutex_ == nullptr) {
      glslang_mutex_ = new std::mutex();
    }
  }

  const std::lock_guard<std::mutex> glslang_lock(*glslang_mutex_);

  if (initialize_count_ == 0) {
    glslang::InitializeProcess();
  }

  initialize_count_++;
}

GlslangInitializer::~GlslangInitializer() {
  const std::lock_guard<std::mutex> glslang_lock(*glslang_mutex_);

  initialize_count_--;

  if (initialize_count_ == 0) {
    glslang::FinalizeProcess();
    // There is no delete for glslang_mutex_ here, because we cannot guarantee
    // there isn't a caller waiting for glslang_mutex_ in GlslangInitializer().
    //
    // This means that this class does leak one std::mutex worth of memory after
    // the final instance is destroyed, but this allows us to defer allocating
    // and constructing until we are sure we need to.
  }
}

void Compiler::SetLimit(Compiler::Limit limit, int value) {
  switch (limit) {
#define RESOURCE(NAME, FIELD, CNAME) \
  case Limit::NAME:                  \
    limits_.FIELD = value;           \
    break;
#include "libshaderc_util/resources.inc"
#undef RESOURCE
  }
}

int Compiler::GetLimit(Compiler::Limit limit) const {
  switch (limit) {
#define RESOURCE(NAME, FIELD, CNAME) \
  case Limit::NAME:                  \
    return limits_.FIELD;
#include "libshaderc_util/resources.inc"
#undef RESOURCE
  }
  return 0;  // Unreachable
}

std::tuple<bool, std::vector<uint32_t>, size_t> Compiler::Compile(
    const string_piece& input_source_string, EShLanguage forced_shader_stage,
    const std::string& error_tag, const char* entry_point_name,
    const std::function<EShLanguage(std::ostream* error_stream,
                                    const string_piece& error_tag)>&
        stage_callback,
    CountingIncluder& includer, OutputType output_type,
    std::ostream* error_stream, size_t* total_warnings,
    size_t* total_errors) const {
  // Compilation results to be returned:
  // Initialize the result tuple as a failed compilation. In error cases, we
  // should return result_tuple directly without setting its members.
  auto result_tuple =
      std::make_tuple(false, std::vector<uint32_t>(), (size_t)0u);
  // Get the reference of the members of the result tuple. We should set their
  // values for succeeded compilation before returning the result tuple.
  bool& succeeded = std::get<0>(result_tuple);
  std::vector<uint32_t>& compilation_output_data = std::get<1>(result_tuple);
  size_t& compilation_output_data_size_in_bytes = std::get<2>(result_tuple);

  // Check target environment.
  const auto target_client_info = GetGlslangClientInfo(
      error_tag, target_env_, target_env_version_, target_spirv_version_,
      target_spirv_version_is_forced_);
  if (!target_client_info.error.empty()) {
    *error_stream << target_client_info.error;
    *total_warnings = 0;
    *total_errors = 1;
    return result_tuple;
  }

#if !SHADERC_ENABLE_HLSL
  if (source_language_ == SourceLanguage::HLSL) {
    std::string err =
        "Shaderc was built without HLSL support. See "
        "https://github.com/KhronosGroup/glslang/issues/4210";
    *error_stream << err << '\n';
    succeeded = false;
    compilation_output_data = ConvertStringToVector(err);
    compilation_output_data_size_in_bytes = err.size();
    return result_tuple;
  }
#endif

  EShLanguage used_shader_stage = forced_shader_stage;
  const std::string macro_definitions =
      shaderc_util::format(predefined_macros_, "#define ", " ", "\n");
  const std::string pound_extension =
      "#extension GL_GOOGLE_include_directive : enable\n";
  const std::string preamble = macro_definitions + pound_extension;

  std::string preprocessed_shader;

  // If only preprocessing, we definitely need to preprocess. Otherwise, if
  // we don't know the stage until now, we need the preprocessed shader to
  // deduce the shader stage.
  if (output_type == OutputType::PreprocessedText ||
      used_shader_stage == EShLangCount) {
    bool success;
    std::string glslang_errors;
    std::tie(success, preprocessed_shader, glslang_errors) =
        PreprocessShader(error_tag, input_source_string, preamble, includer);

    success &= PrintFilteredErrors(error_tag, error_stream, warnings_as_errors_,
                                   /* suppress_warnings = */ true,
                                   glslang_errors.c_str(), total_warnings,
                                   total_errors);
    if (!success) return result_tuple;
    // Because of the behavior change of the #line directive, the #line
    // directive introducing each file's content must use the syntax for the
    // specified version. So we need to probe this shader's version and
    // profile.
    int version;
    EProfile profile;
    std::tie(version, profile) = DeduceVersionProfile(preprocessed_shader);
    const bool is_for_next_line = LineDirectiveIsForNextLine(version, profile);

    preprocessed_shader =
        CleanupPreamble(preprocessed_shader, error_tag, pound_extension,
                        includer.num_include_directives(), is_for_next_line);

    if (output_type == OutputType::PreprocessedText) {
      // Set the values of the result tuple.
      succeeded = true;
      compilation_output_data = ConvertStringToVector(preprocessed_shader);
      compilation_output_data_size_in_bytes = preprocessed_shader.size();
      return result_tuple;
    } else if (used_shader_stage == EShLangCount) {
      std::string errors;
      std::tie(used_shader_stage, errors) =
          GetShaderStageFromSourceCode(error_tag, preprocessed_shader);
      if (!errors.empty()) {
        *error_stream << errors;
        return result_tuple;
      }
      if (used_shader_stage == EShLangCount) {
        if ((used_shader_stage = stage_callback(error_stream, error_tag)) ==
            EShLangCount) {
          return result_tuple;
        }
      }
    }
  }

  // Parsing requires its own Glslang symbol tables.
  glslang::TShader shader(used_shader_stage);
  const char* shader_strings = input_source_string.data();
  const int shader_lengths = static_cast<int>(input_source_string.size());
  const char* string_names = error_tag.c_str();
  shader.setStringsWithLengthsAndNames(&shader_strings, &shader_lengths,
                                       &string_names, 1);
  shader.setPreamble(preamble.c_str());
  shader.setSourceEntryPoint("main");
  shader.setEntryPoint(entry_point_name);
  shader.setAutoMapBindings(auto_bind_uniforms_);

  shader.setGlobalUniformBlockName(SHADERC_EXT_DEFAULT_UNIFORM_BLOCK_NAME);
  shader.setGlobalUniformSet(SHADERC_EXT_DEFAULT_UNIFORM_BLOCK_INITIAL_SET);
  shader.setGlobalUniformBinding(SHADERC_EXT_DEFAULT_UNIFORM_BLOCK_INITIAL_BINDING);
  
  if (auto_combined_image_sampler_) {
    shader.setTextureSamplerTransformMode(
        EShTexSampTransUpgradeTextureRemoveSampler);
  }
  shader.setAutoMapLocations(auto_map_locations_);
  const auto& bases = auto_binding_base_[static_cast<int>(used_shader_stage)];
  shader.setShiftImageBinding(bases[static_cast<int>(UniformKind::Image)]);
  shader.setShiftSamplerBinding(bases[static_cast<int>(UniformKind::Sampler)]);
  shader.setShiftTextureBinding(bases[static_cast<int>(UniformKind::Texture)]);
  shader.setShiftUboBinding(bases[static_cast<int>(UniformKind::Buffer)]);
  shader.setShiftSsboBinding(
      bases[static_cast<int>(UniformKind::StorageBuffer)]);
  shader.setShiftUavBinding(
      bases[static_cast<int>(UniformKind::UnorderedAccessView)]);
#if SHADERC_ENABLE_HLSL
  shader.setHlslIoMapping(hlsl_iomap_);
#endif
  shader.setResourceSetBinding(
      hlsl_explicit_bindings_[static_cast<int>(used_shader_stage)]);
  shader.setEnvClient(target_client_info.client,
                      target_client_info.client_version);
  shader.setEnvTarget(target_client_info.target_language,
                      target_client_info.target_language_version);
#if SHADERC_ENABLE_HLSL
  if (hlsl_functionality1_enabled_) {
    shader.setEnvTargetHlslFunctionality1();
  }
#endif
  if (vulkan_rules_relaxed_) {
    glslang::EShSource language = glslang::EShSourceNone;
    switch (source_language_) {
      case SourceLanguage::GLSL:
        language = glslang::EShSourceGlsl;
        break;
      case SourceLanguage::HLSL:
        language = glslang::EShSourceHlsl;
        break;
    }
    // This option will only be used if the Vulkan client is used.
    // If new versions of GL_KHR_vulkan_glsl come out, it would make sense to
    // let callers specify which version to use. For now, just use 100.
    shader.setEnvInput(language, used_shader_stage, glslang::EShClientVulkan,
                       100);
    shader.setEnvInputVulkanRulesRelaxed();
  }
  shader.setInvertY(invert_y_enabled_);
  shader.setNanMinMaxClamp(nan_clamp_);

  const EShMessages rules =
      GetMessageRules(target_env_, source_language_, hlsl_offsets_,
                      hlsl_16bit_types_enabled_, generate_debug_info_);

  bool success = shader.parse(&limits_, default_version_, default_profile_,
                              force_version_profile_, kNotForwardCompatible,
                              rules, includer);

  success &= PrintFilteredErrors(error_tag, error_stream, warnings_as_errors_,
                                 suppress_warnings_, shader.getInfoLog(),
                                 total_warnings, total_errors);
  if (!success) return result_tuple;

  GlobalUniformBlockOptimizer block_optimizer(*shader.getIntermediate());
  if (auto root = shader.getIntermediate()->getTreeRoot()) {
    root->traverse(&block_optimizer);

    GlobalUniformBlockIndexRemapper index_remapper(block_optimizer.indexMap());
    root->traverse(&index_remapper);
  }

  glslang::TProgram program;
  program.addShader(&shader);
  success = program.link(EShMsgDefault);
  if (success) {
    if (auto_bind_uniforms_) {
      ShadercAutoBindResolver io_resolver(*shader.getIntermediate());
      success = program.mapIO(&io_resolver);
    } else {
      success = program.mapIO();
    }
  }

  success &= PrintFilteredErrors(error_tag, error_stream, warnings_as_errors_,
                                 suppress_warnings_, program.getInfoLog(),
                                 total_warnings, total_errors);
  if (!success) return result_tuple;

  // 'spirv' is an alias for the compilation_output_data. This alias is added
  // to serve as an input for the call to DissassemblyBinary.
  std::vector<uint32_t>& spirv = compilation_output_data;
  glslang::SpvOptions options;
  options.generateDebugInfo = generate_debug_info_;
  options.disableOptimizer = true;
  options.optimizeSize = false;
  // Note the call to GlslangToSpv also populates compilation_output_data.
  glslang::GlslangToSpv(*program.getIntermediate(used_shader_stage), spirv,
                        &options);

  // Set the tool field (the top 16-bits) in the generator word to
  // 'Shaderc over Glslang'.
  const uint32_t shaderc_generator_word = 13;  // From SPIR-V XML Registry
  const uint32_t generator_word_index = 2;     // SPIR-V 2.3: Physical layout
  assert(spirv.size() > generator_word_index);
  spirv[generator_word_index] =
      (spirv[generator_word_index] & 0xffff) | (shaderc_generator_word << 16);

  std::vector<PassId> opt_passes;

  if (hlsl_legalization_enabled_ && source_language_ == SourceLanguage::HLSL) {
    // If from HLSL, run this passes to "legalize" the SPIR-V for Vulkan
    // eg. forward and remove memory writes of opaque types.
    opt_passes.push_back(PassId::kLegalizationPasses);
  }

  opt_passes.insert(opt_passes.end(), enabled_opt_passes_.begin(),
                    enabled_opt_passes_.end());

  if (!opt_passes.empty()) {
    spvtools::OptimizerOptions opt_options;
    opt_options.set_preserve_bindings(preserve_bindings_);
    opt_options.set_max_id_bound(max_id_bound_);

    std::string opt_errors;
    if (!SpirvToolsOptimize(target_env_, target_env_version_, opt_passes,
                            opt_options, &spirv, &opt_errors)) {
      *error_stream << "shaderc: internal error: compilation succeeded but "
                       "failed to optimize: "
                    << opt_errors << "\n";
      return result_tuple;
    }
  }

  if (output_type == OutputType::SpirvAssemblyText) {
    std::string text_or_error;
    if (!SpirvToolsDisassemble(target_env_, target_env_version_, spirv,
                               &text_or_error)) {
      *error_stream << "shaderc: internal error: compilation succeeded but "
                       "failed to disassemble: "
                    << text_or_error << "\n";
      return result_tuple;
    }
    succeeded = true;
    compilation_output_data = ConvertStringToVector(text_or_error);
    compilation_output_data_size_in_bytes = text_or_error.size();
    return result_tuple;
  } else {
    succeeded = true;
    // Note compilation_output_data is already populated in GlslangToSpv().
    compilation_output_data_size_in_bytes = spirv.size() * sizeof(spirv[0]);
    return result_tuple;
  }
}

void Compiler::AddMacroDefinition(const char* macro, size_t macro_length,
                                  const char* definition,
                                  size_t definition_length) {
  predefined_macros_[std::string(macro, macro_length)] =
      definition ? std::string(definition, definition_length) : "";
}

void Compiler::SetTargetEnv(Compiler::TargetEnv env,
                            Compiler::TargetEnvVersion version) {
  target_env_ = env;
  target_env_version_ = version;
}

void Compiler::SetTargetSpirv(Compiler::SpirvVersion version) {
  target_spirv_version_ = version;
  target_spirv_version_is_forced_ = true;
}

void Compiler::SetSourceLanguage(Compiler::SourceLanguage lang) {
  source_language_ = lang;
}

void Compiler::SetForcedVersionProfile(int version, EProfile profile) {
  default_version_ = version;
  default_profile_ = profile;
  force_version_profile_ = true;
}

void Compiler::SetWarningsAsErrors() { warnings_as_errors_ = true; }

void Compiler::SetGenerateDebugInfo() {
  generate_debug_info_ = true;
  for (size_t i = 0; i < enabled_opt_passes_.size(); ++i) {
    if (enabled_opt_passes_[i] == PassId::kStripDebugInfo) {
      enabled_opt_passes_[i] = PassId::kNullPass;
    }
  }
}

void Compiler::SetOptimizationLevel(Compiler::OptimizationLevel level) {
  // Clear previous settings first.
  enabled_opt_passes_.clear();

  switch (level) {
    case OptimizationLevel::Size:
      if (!generate_debug_info_) {
        enabled_opt_passes_.push_back(PassId::kStripDebugInfo);
      }
      enabled_opt_passes_.push_back(PassId::kSizePasses);
      break;
    case OptimizationLevel::Performance:
      if (!generate_debug_info_) {
        enabled_opt_passes_.push_back(PassId::kStripDebugInfo);
      }
      enabled_opt_passes_.push_back(PassId::kPerformancePasses);
      break;
    default:
      break;
  }
}

void Compiler::EnableHlslLegalization(bool hlsl_legalization_enabled) {
  hlsl_legalization_enabled_ = hlsl_legalization_enabled;
}

void Compiler::EnableHlslFunctionality1(bool enable) {
  hlsl_functionality1_enabled_ = enable;
}

void Compiler::SetVulkanRulesRelaxed(bool enable) {
  vulkan_rules_relaxed_ = enable;
}

void Compiler::EnableHlsl16BitTypes(bool enable) {
  hlsl_16bit_types_enabled_ = enable;
}

void Compiler::EnableInvertY(bool enable) { invert_y_enabled_ = enable; }

void Compiler::SetNanClamp(bool enable) { nan_clamp_ = enable; }

void Compiler::SetSuppressWarnings() { suppress_warnings_ = true; }

std::tuple<bool, std::string, std::string> Compiler::PreprocessShader(
    const std::string& error_tag, const string_piece& shader_source,
    const string_piece& shader_preamble, CountingIncluder& includer) const {
  // The stage does not matter for preprocessing.
  glslang::TShader shader(EShLangVertex);
  const char* shader_strings = shader_source.data();
  const int shader_lengths = static_cast<int>(shader_source.size());
  const char* string_names = error_tag.c_str();
  shader.setStringsWithLengthsAndNames(&shader_strings, &shader_lengths,
                                       &string_names, 1);
  shader.setPreamble(shader_preamble.data());
  auto target_client_info = GetGlslangClientInfo(
      error_tag, target_env_, target_env_version_, target_spirv_version_,
      target_spirv_version_is_forced_);
  if (!target_client_info.error.empty()) {
    return std::make_tuple(false, "", target_client_info.error);
  }
  shader.setEnvClient(target_client_info.client,
                      target_client_info.client_version);
#if SHADERC_ENABLE_HLSL
  if (hlsl_functionality1_enabled_) {
    shader.setEnvTargetHlslFunctionality1();
  }
#endif
  shader.setInvertY(invert_y_enabled_);
  shader.setNanMinMaxClamp(nan_clamp_);

  // The preprocessor might be sensitive to the target environment.
  // So combine the existing rules with the just-give-me-preprocessor-output
  // flag.
  const auto rules = static_cast<EShMessages>(
      EShMsgOnlyPreprocessor |
      GetMessageRules(target_env_, source_language_, hlsl_offsets_,
                      hlsl_16bit_types_enabled_, false));

  std::string preprocessed_shader;
  const bool success = shader.preprocess(
      &limits_, default_version_, default_profile_, force_version_profile_,
      kNotForwardCompatible, rules, &preprocessed_shader, includer);

  if (success) {
    return std::make_tuple(true, preprocessed_shader, shader.getInfoLog());
  }
  return std::make_tuple(false, "", shader.getInfoLog());
}

std::string Compiler::CleanupPreamble(const string_piece& preprocessed_shader,
                                      const string_piece& error_tag,
                                      const string_piece& pound_extension,
                                      int num_include_directives,
                                      bool is_for_next_line) const {
  // Those #define directives in preamble will become empty lines after
  // preprocessing. We also injected an #extension directive to turn on #include
  // directive support. In the original preprocessing output from glslang, it
  // appears before the user source string. We need to do proper adjustment:
  // * Remove empty lines generated from #define directives in preamble.
  // * If there is no #include directive in the source code, we do not need to
  //   output the injected #extension directive. Otherwise,
  // * If there exists a #version directive in the source code, it should be
  //   placed at the first line. Its original line will be filled with an empty
  //   line as placeholder to maintain the code structure.

  const std::vector<string_piece> lines =
      preprocessed_shader.get_fields('\n', /* keep_delimiter = */ true);

  std::ostringstream output_stream;

  size_t pound_extension_index = lines.size();
  size_t pound_version_index = lines.size();
  for (size_t i = 0; i < lines.size(); ++i) {
    if (lines[i] == pound_extension) {
      pound_extension_index = std::min(i, pound_extension_index);
    } else if (lines[i].starts_with("#version")) {
      // In a preprocessed shader, directives are in a canonical format, so we
      // can confidently compare to '#version' verbatim, without worrying about
      // whitespace.
      pound_version_index = i;
      if (num_include_directives > 0) output_stream << lines[i];
      break;
    }
  }
  // We know that #extension directive exists and appears before #version
  // directive (if any).
  assert(pound_extension_index < lines.size());

  for (size_t i = 0; i < pound_extension_index; ++i) {
    // All empty lines before the #line directive we injected are generated by
    // preprocessing preamble. Do not output them.
    if (lines[i].strip_whitespace().empty()) continue;
    output_stream << lines[i];
  }

  if (num_include_directives > 0) {
    output_stream << pound_extension;
    // Also output a #line directive for the main file.
    output_stream << GetLineDirective(is_for_next_line, error_tag);
  }

  for (size_t i = pound_extension_index + 1; i < lines.size(); ++i) {
    if (i == pound_version_index) {
      if (num_include_directives > 0) {
        output_stream << "\n";
      } else {
        output_stream << lines[i];
      }
    } else {
      output_stream << lines[i];
    }
  }

  return output_stream.str();
}

std::pair<EShLanguage, std::string> Compiler::GetShaderStageFromSourceCode(
    string_piece filename, const std::string& preprocessed_shader) const {
  const string_piece kPragmaShaderStageDirective = "#pragma shader_stage";

  int version;
  EProfile profile;
  std::tie(version, profile) = DeduceVersionProfile(preprocessed_shader);
  const bool is_for_next_line = LineDirectiveIsForNextLine(version, profile);

  std::vector<string_piece> lines =
      string_piece(preprocessed_shader).get_fields('\n');
  // The filename, logical line number (which starts from 1 and is sensitive to
  // #line directives), and stage value for #pragma shader_stage() directives.
  std::vector<std::tuple<string_piece, size_t, string_piece>> stages;
  // The physical line numbers of the first #pragma shader_stage() line and
  // first non-preprocessing line in the preprocessed shader text.
  size_t first_pragma_physical_line = lines.size() + 1;
  size_t first_non_pp_line = lines.size() + 1;

  for (size_t i = 0, logical_line_no = 1; i < lines.size(); ++i) {
    const string_piece current_line = lines[i].strip_whitespace();
    if (current_line.starts_with(kPragmaShaderStageDirective)) {
      const string_piece stage_value =
          current_line.substr(kPragmaShaderStageDirective.size()).strip("()");
      stages.emplace_back(filename, logical_line_no, stage_value);
      first_pragma_physical_line = std::min(first_pragma_physical_line, i + 1);
    } else if (!current_line.empty() && !current_line.starts_with("#")) {
      first_non_pp_line = std::min(first_non_pp_line, i + 1);
    }

    // Update logical line number for the next line.
    if (current_line.starts_with(kLineDirectivePrefix)) {
      string_piece name;
      std::tie(logical_line_no, name) = DecodeLineDirective(current_line);
      if (!name.empty()) filename = name;
      // Note that for core profile, the meaning of #line changed since version
      // 330. The line number given by #line used to mean the logical line
      // number of the #line line. Now it means the logical line number of the
      // next line after the #line line.
      if (!is_for_next_line) ++logical_line_no;
    } else {
      ++logical_line_no;
    }
  }
  if (stages.empty()) return std::make_pair(EShLangCount, "");

  std::string error_message;

  const string_piece& first_pragma_filename = std::get<0>(stages[0]);
  const std::string first_pragma_line = std::to_string(std::get<1>(stages[0]));
  const string_piece& first_pragma_stage = std::get<2>(stages[0]);

  if (first_pragma_physical_line > first_non_pp_line) {
    error_message += first_pragma_filename.str() + ":" + first_pragma_line +
                     ": error: '#pragma': the first 'shader_stage' #pragma "
                     "must appear before any non-preprocessing code\n";
  }

  EShLanguage stage = MapStageNameToLanguage(first_pragma_stage);
  if (stage == EShLangCount) {
    error_message +=
        first_pragma_filename.str() + ":" + first_pragma_line +
        ": error: '#pragma': invalid stage for 'shader_stage' #pragma: '" +
        first_pragma_stage.str() + "'\n";
  }

  for (size_t i = 1; i < stages.size(); ++i) {
    const string_piece& current_stage = std::get<2>(stages[i]);
    if (current_stage != first_pragma_stage) {
      const string_piece& current_filename = std::get<0>(stages[i]);
      const std::string current_line = std::to_string(std::get<1>(stages[i]));
      error_message += current_filename.str() + ":" + current_line +
                       ": error: '#pragma': conflicting stages for "
                       "'shader_stage' #pragma: '" +
                       current_stage.str() + "' (was '" +
                       first_pragma_stage.str() + "' at " +
                       first_pragma_filename.str() + ":" + first_pragma_line +
                       ")\n";
    }
  }

  return std::make_pair(error_message.empty() ? stage : EShLangCount,
                        error_message);
}

std::pair<int, EProfile> Compiler::DeduceVersionProfile(
    const std::string& preprocessed_shader) const {
  int version = default_version_;
  EProfile profile = default_profile_;
  if (!force_version_profile_) {
    std::tie(version, profile) =
        GetVersionProfileFromSourceCode(preprocessed_shader);
    if (version == 0 && profile == ENoProfile) {
      version = default_version_;
      profile = default_profile_;
    }
  }
  return std::make_pair(version, profile);
}

std::pair<int, EProfile> Compiler::GetVersionProfileFromSourceCode(
    const std::string& preprocessed_shader) const {
  string_piece pound_version = preprocessed_shader;
  const size_t pound_version_loc = pound_version.find("#version");
  if (pound_version_loc == string_piece::npos) {
    return std::make_pair(0, ENoProfile);
  }
  pound_version =
      pound_version.substr(pound_version_loc + std::strlen("#version"));
  pound_version = pound_version.substr(0, pound_version.find_first_of("\n"));

  std::string version_profile;
  for (const auto character : pound_version) {
    if (character != ' ') version_profile += character;
  }

  int version;
  EProfile profile;
  if (!ParseVersionProfile(version_profile, &version, &profile)) {
    return std::make_pair(0, ENoProfile);
  }
  return std::make_pair(version, profile);
}

// Converts a string to a vector of uint32_t by copying the content of a given
// string to a vector<uint32_t> and returns it. Appends '\0' at the end if extra
// bytes are required to complete the last element.
std::vector<uint32_t> ConvertStringToVector(const std::string& str) {
  size_t num_bytes_str = str.size() + 1u;
  size_t vector_length =
      (num_bytes_str + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  std::vector<uint32_t> result_vec(vector_length, 0);
  std::strncpy(reinterpret_cast<char*>(result_vec.data()), str.c_str(),
               str.size());
  return result_vec;
}

GlslangClientInfo GetGlslangClientInfo(
    const std::string& error_tag, shaderc_util::Compiler::TargetEnv env,
    shaderc_util::Compiler::TargetEnvVersion env_version,
    shaderc_util::Compiler::SpirvVersion spv_version,
    bool spv_version_is_forced) {
  GlslangClientInfo result;
  std::ostringstream errs;

  using shaderc_util::Compiler;
  switch (env) {
    case Compiler::TargetEnv::Vulkan:
      result.client = glslang::EShClientVulkan;
      if (env_version == Compiler::TargetEnvVersion::Default ||
          env_version == Compiler::TargetEnvVersion::Vulkan_1_0) {
        result.client_version = glslang::EShTargetVulkan_1_0;
      } else if (env_version == Compiler::TargetEnvVersion::Vulkan_1_1) {
        result.client_version = glslang::EShTargetVulkan_1_1;
        result.target_language_version = glslang::EShTargetSpv_1_3;
      } else if (env_version == Compiler::TargetEnvVersion::Vulkan_1_2) {
        result.client_version = glslang::EShTargetVulkan_1_2;
        result.target_language_version = glslang::EShTargetSpv_1_5;
      } else if (env_version == Compiler::TargetEnvVersion::Vulkan_1_3) {
        result.client_version = glslang::EShTargetVulkan_1_3;
        result.target_language_version = glslang::EShTargetSpv_1_6;
      } else if (env_version == Compiler::TargetEnvVersion::Vulkan_1_4) {
        result.client_version = glslang::EShTargetVulkan_1_4;
        result.target_language_version = glslang::EShTargetSpv_1_6;
      } else {
        errs << "error:" << error_tag << ": Invalid target client version "
             << static_cast<uint32_t>(env_version) << " for Vulkan environment "
             << int(env);
      }
      break;
    case Compiler::TargetEnv::OpenGLCompat:
      errs << "error: OpenGL compatibility profile is not supported";
      break;
    case Compiler::TargetEnv::OpenGL:
      result.client = glslang::EShClientOpenGL;
      if (env_version == Compiler::TargetEnvVersion::Default ||
          env_version == Compiler::TargetEnvVersion::OpenGL_4_5) {
        result.client_version = glslang::EShTargetOpenGL_450;
      } else {
        errs << "error:" << error_tag << ": Invalid target client version "
             << static_cast<uint32_t>(env_version) << " for OpenGL environment "
             << int(env);
      }
      break;
    default:
      errs << "error:" << error_tag << ": Invalid target client environment "
           << int(env);
      break;
  }

  if (spv_version_is_forced && errs.str().empty()) {
    switch (spv_version) {
      case Compiler::SpirvVersion::v1_0:
        result.target_language_version = glslang::EShTargetSpv_1_0;
        break;
      case Compiler::SpirvVersion::v1_1:
        result.target_language_version = glslang::EShTargetSpv_1_1;
        break;
      case Compiler::SpirvVersion::v1_2:
        result.target_language_version = glslang::EShTargetSpv_1_2;
        break;
      case Compiler::SpirvVersion::v1_3:
        result.target_language_version = glslang::EShTargetSpv_1_3;
        break;
      case Compiler::SpirvVersion::v1_4:
        result.target_language_version = glslang::EShTargetSpv_1_4;
        break;
      case Compiler::SpirvVersion::v1_5:
        result.target_language_version = glslang::EShTargetSpv_1_5;
        break;
      case Compiler::SpirvVersion::v1_6:
        result.target_language_version = glslang::EShTargetSpv_1_6;
        break;
      default:
        errs << "error:" << error_tag << ": Unknown SPIR-V version " << std::hex
             << uint32_t(spv_version);
        break;
    }
  }
  result.error = errs.str();
  return result;
}

}  // namespace shaderc_util
