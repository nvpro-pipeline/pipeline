// emitColor is the function called at the end of the fragment shader to emit the calculated color
// There are a couple of emitColor functions (like in emitColorDepth.glsl, emitColorOITAllCounter.glsl, etc.) which reflect
// the various needs of the various algorithms.

// Version of emitColor() used in OITAll in the non-depth pass for transparent fragments.
// This function stores the incoming color and its depth (gl_FragCoord.z) in an image buffer.
// The size of that buffer has been determined by the depth pass, such that every sample
// for every fragment fits into that buffer.

// enable early depth test to prevent the expensive fragment shader on samples that
// are already covered by opaque samples
layout(early_fragment_tests) in;

layout(size1x32) uniform restrict uimage2D                perFragmentCount;   // 2D image sized as the view to count the samples per fragment
layout(size1x32) uniform restrict readonly uimage2D       perFragmentOffset;  // 2D image sized as the view holding the base offset per fragment
layout(size2x32) uniform restrict writeonly uimageBuffer  samplesBuffer;      // image buffer sized to hold color and depth on all samples on all fragments

layout(location = 0, index = 0) out vec4 Color;

void emitColor( in vec4 color )
{
  color = clamp( color, 0.0f, 1.0f );
  if ( sys_TransparentPass )
  {
    color.rgb *= color.a;

    // get the base offset for this fragment out of perFragmentOffset
    // plus atomically add one to the number of samples on this fragment
    // gives offset into the image buffer to store color and depth per sample and fragment
    ivec2 screenPos = ivec2( gl_FragCoord.xy );
    unsigned int offset = imageLoad( perFragmentOffset, screenPos ).x + imageAtomicAdd( perFragmentCount, screenPos, uint(1) );
    imageStore( samplesBuffer, int(offset), uvec4( packUnorm4x8( color ), floatBitsToUint( gl_FragCoord.z ), 0, 0 ) );

    Color = vec4( 0 );    // the actual color doesn't matter, as it is calculated in the resolve pass
  }
  else
  {
    Color = color;
  }
}
