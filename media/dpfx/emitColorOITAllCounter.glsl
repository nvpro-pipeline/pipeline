// emitColor is the function called at the end of the fragment shader to emit the calculated color
// There are a couple of emitColor functions (like in emitColorDepth.glsl, emitColorOITAllCounter.glsl, etc.) which reflect
// the various needs of the various algorithms.

// Version of emitColor() used in OITAll in the depth pass for transparent fragments.
// This function counts the number of samples per fragment. That number of color values
// needs to be gathered in the non-depth pass.

// enable early depth test to prevent the expensive fragment shader on samples that
// are already covered by opaque samples
layout(early_fragment_tests) in;

layout(size1x32) uniform restrict uimage2D  perFragmentCount;   // 2D image sized as the view to count the samples per fragment

layout(location = 0, index = 0) out vec4 Color;

void emitColor( in vec4 color )
{
  // atomically add one to the number of samples on this fragment (this is used in the transparent pass only!)
  imageAtomicAdd( perFragmentCount, ivec2( gl_FragCoord.xy ), uint(1) );

  Color = vec4( 0 );    // the actual color doesn't matter, as this is just the depth pass
}
