#version 450

// ============================================================================
// Test shader for GlobalUniformBlockOptimizer (std430 repacking of the
// shaderc-generated default/global uniform block)
//
// Test with:
//  glslc --target-env=vulkan1.4 -O -fauto-bind-uniforms -g globalblock_test.frag
//
// All uniforms below are declared LOOSE (no named block), which is exactly
// what glslang folds into the shaderc-generated default uniform block
// (SHADERC_EXT_DEFAULT_UNIFORM_BLOCK_NAME) that GlobalUniformBlockOptimizer
// targets.
//
// std430 per-type (base alignment, size) used below:
//   float / int / uint / bool  -> align 4,  size 4
//   vec2                       -> align 8,  size 8
//   vec3                       -> align 16, size 12   (align rounds to 4N)
//   vec4                       -> align 16, size 16
//   mat3 (3x vec3 columns)     -> align 16, size 48   (16 per column)
//   mat4 (4x vec4 columns)     -> align 16, size 64
//   float[3]                  -> align 4,  size 12   (std430: no vec4 stride
//                                                       rounding on arrays)
//   vec3[2]                   -> align 16, size 32   (stride 16/element)
//   struct { vec2; float; }    -> align 8,  size 16   (max-member align,
//                                                       size rounded to that
//                                                       align; NOT rounded to
//                                                       16 like std140 would)
//
// Declaration order below is DELIBERATELY scrambled/adversarial. A naive
// "lay out fields in declaration order" packer produces:
//     total size (incl. trailing round-up) = 256 bytes, 28 bytes wasted
//
// GlobalUniformBlockOptimizer's greedy best-fit (sort by alignment desc,
// then size desc, then name; place each field in the smallest existing gap
// it fits, else append) instead produces:
//     total size (incl. trailing round-up) = 240 bytes, 12 bytes wasted
//     (all 12 wasted bytes are the unavoidable final block-alignment
//     round-up -- there are ZERO internal holes)
//
// Expected FINAL member order after optimization, by byte offset:
//   [ 0] mat4Field       ( 0..64 )   align16 size64
//   [64] mat3Field       ( 64..112)  align16 size48
//   [112] vec3ArrField   (112..144)  align16 size32
//   [144] vec4Field      (144..160)  align16 size16
//   [160] vec3Field      (160..172)  align16 size12
//   [172] aScalar        (172..176)  align4  size4   <-- see note below
//   [176] structField    (176..192)  align8  size16
//   [192] vec2Field      (192..200)  align8  size8
//   [200] floatArrField  (200..212)  align4  size12
//   [212] mScalar        (212..216)  align4  size4
//   [216] zScalar        (216..220)  align4  size4
//   [220] zzUintScalar   (220..224)  align4  size4
//   [224] zzzBoolScalar  (224..228)  align4  size4
//   -- end of useful data at 228, block rounds up to 240 (align 16) --
//
// NOTE on aScalar @172: placing vec3Field (align16,size12) at 160 leaves the
// block's cursor at 172, which is NOT 16-aligned. structField (align8) then
// has to jump ahead to offset 176, opening a 4-byte hole at [172,176). Later,
// when the four leftover 4-byte scalars (aScalar/mScalar/zScalar/
// zzUintScalar/zzzBoolScalar) are processed in alphabetical order, aScalar
// is the FIRST one considered and it fits that 4-byte hole EXACTLY -- so it
// gets placed at offset 172, ahead of structField/vec2Field/floatArrField in
// the final layout, even though mScalar/zScalar/etc. (same type, same
// alignment tier, later alphabetically) end up much further down at the
// tail. This is the one detail that only a true best-fit gap search
// produces; a packer that just concatenates alignment-tier buckets in order
// would instead group all five scalars together at the very end and leave
// that 4-byte hole (and the corresponding 4 extra bytes of total size)
// unfilled.
// ============================================================================

struct BlockishStruct {
  vec2 s0;
  float s1;
};

// --- declared in scrambled/adversarial order on purpose ---
uniform float aScalar;
uniform bool zzzBoolScalar;
uniform BlockishStruct structField;
uniform mat4 mat4Field;
uniform vec2 vec2Field;
uniform float mScalar;
uniform uint zzUintScalar;
uniform vec3 vec3ArrField[2];
uniform float floatArrField[3];
uniform vec4 vec4Field;
uniform mat3 mat3Field;
uniform int zScalar;
uniform vec3 vec3Field;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 c = vec4(0.0);

    c += vec4(aScalar);
    c += vec4(zzzBoolScalar ? 1.0 : 0.0);
    c += vec4(structField.s0, structField.s1, 0.0);
    c += mat4Field[0];
    c += vec4(vec2Field, 0.0, 0.0);
    c += vec4(mScalar);
    c += vec4(float(zzUintScalar));
    c += vec4(vec3ArrField[0] + vec3ArrField[1], 0.0);
    c += vec4(floatArrField[0] + floatArrField[1] + floatArrField[2]);
    c += vec4Field;
    c += vec4(mat3Field[0], 0.0);
    c += vec4(float(zScalar));
    c += vec4(vec3Field, 0.0);

    outColor = c;
}