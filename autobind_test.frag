#version 450

// ============================================================================
// Test shader for ShadercAutoBindResolver (-fauto-bind-uniforms / auto_map_locations)
//
// Test with:
//  glslc --target-env=vulkan1.4 -O -fauto-bind-uniforms -g autobind_test.frag
// (add whatever flag your build exposes for auto_map_locations_ / auto_bind_uniforms_)
//
// Explicit sets used anywhere in this file: 0, 2, 5
// => the resolver's "first undeclared set" (catch-all) is SET 1, because
//    0 is taken, 1 is free.
//
// Every variable below has a comment stating the EXPECTED resolved
// (set, binding). Only sampler2D / UBO are used so all resources share
// the same binding namespace within a set (assuming
// getBindingsPerResourceType() == false, the common default).
// ============================================================================


// ---------------------------------------------------------------------------
// CASE 1: set + binding both explicit -> stays exactly as written
// ---------------------------------------------------------------------------
layout(set = 0, binding = 0) uniform sampler2D texExplicit_0_0;   // expect (0,0)
layout(set = 0, binding = 1) uniform Ubo0 { vec4 a; } ubo_0_1;    // expect (0,1)
layout(set = 0, binding = 5) uniform sampler2D texExplicit_0_5;   // expect (0,5)  <- leaves holes at 2,3,4
layout(set = 5, binding = 2) uniform sampler2D texExplicit_5_2;   // expect (5,2)
layout(set = 2, binding = 3) uniform Ubo2 { vec4 b; } ubo_2_3;    // expect (2,3)


// ---------------------------------------------------------------------------
// CASE 2: set only -> first free binding WITHIN that set (hole-filling)
// ---------------------------------------------------------------------------
// Set 0 already has bindings {0,1,5} taken -> holes at 2,3,4.
layout(set = 0) uniform sampler2D texSetOnly_0_a;   // expect (0,2)
layout(set = 0) uniform sampler2D texSetOnly_0_b;   // expect (0,3)
layout(set = 0) uniform sampler2D texSetOnly_0_c;   // expect (0,4)
layout(set = 0) uniform sampler2D texSetOnly_0_d;   // expect (0,6)  (0-5 now full, next free is 6)

// Set 2 only has binding {3} taken so far -> first free is 0.
layout(set = 2) uniform sampler2D texSetOnly_2_a;   // expect (2,0)
layout(set = 2) uniform sampler2D texSetOnly_2_b;   // expect (2,1)


// ---------------------------------------------------------------------------
// CASE 3: binding only (no set) -> lands in the first undeclared set (= 1),
// using the exact binding value given
// ---------------------------------------------------------------------------
layout(binding = 0) uniform sampler2D texBindOnly_a;  // expect (1,0)
layout(binding = 3) uniform sampler2D texBindOnly_b;  // expect (1,3)


// ---------------------------------------------------------------------------
// CASE 4: neither set nor binding -> also lands in set 1 (same catch-all),
// using the first free binding there
// ---------------------------------------------------------------------------
// Set 1 currently has bindings {0,3} taken (from case 3 above) -> holes at 1,2,4...
uniform sampler2D texNeither_a;   // expect (1,1)
uniform sampler2D texNeither_b;   // expect (1,2)
uniform sampler2D texNeither_c;   // expect (1,4)


// ---------------------------------------------------------------------------
// Trivial main so this compiles as a valid fragment shader on its own.
// ---------------------------------------------------------------------------
layout(location = 0) out vec4 outColor;

void main() {
    vec4 c = texture(texExplicit_0_0, vec2(0.0));
    c += ubo_0_1.a;
    c += texture(texExplicit_0_5, vec2(0.0));
    c += texture(texExplicit_5_2, vec2(0.0));
    c += ubo_2_3.b;

    c += texture(texSetOnly_0_a, vec2(0.0));
    c += texture(texSetOnly_0_b, vec2(0.0));
    c += texture(texSetOnly_0_c, vec2(0.0));
    c += texture(texSetOnly_0_d, vec2(0.0));
    c += texture(texSetOnly_2_a, vec2(0.0));
    c += texture(texSetOnly_2_b, vec2(0.0));

    c += texture(texBindOnly_a, vec2(0.0));
    c += texture(texBindOnly_b, vec2(0.0));

    c += texture(texNeither_a, vec2(0.0));
    c += texture(texNeither_b, vec2(0.0));
    c += texture(texNeither_c, vec2(0.0));

    outColor = c;
}
