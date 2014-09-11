// emitColor is the function called at the end of the fragment shader to emit the calculated color
// There are a couple of emitColor functions (like in emitColorDepth.glsl, emitColorOITAllCounter.glsl, etc.) which reflect
// the various needs of the various algorithms.

// Version of emitColor() used in OITClosestList in the non-depth pass for transparent fragments.
// This function stores the all samples up to the samples buffer size. Any sample beyond that size
// is simply ignored!
// Later on, in oitClosestListResolve_fs, the closest up to sys_OITDepth samples per fragment are
// sorted and blended, while any additional samples on that fragment are blended unsorted
// Note, that the buffer for the samples might be too small, and severe visual artifacts might be visible!

// enable early depth test to prevent the expensive fragment shader on samples that
// are already covered by opaque samples
layout(early_fragment_tests) in;

layout(size1x32) uniform restrict uimage1D     counterAccu;       // 1-element image to count all the samples
layout(size4x32) uniform restrict uimageBuffer samplesBuffer;     // image buffer sized to hold samples up to its size
layout(size1x32) uniform restrict uimage2D     perFragmentOffset; // 2D image sized as the view holding the base offset per fragment

layout(location = 0, index = 0) out vec4 Color;

void emitColor( in vec4 color )
{
  color = clamp( color, 0.0f, 1.0f );
  if ( sys_TransparentPass )
  {
    if ( 1.0f / 255.0f <= color.a )   // just do something if the transparency of the current color can make a difference
    {
      int bufferSize = imageSize( samplesBuffer );          // make sure, we don't store more than bufferSize samples

      uint newOffset = imageAtomicAdd( counterAccu, 0, 1 ); // add one to the counterAccu, and get the previous value as offset
      if ( newOffset < bufferSize )
      {
        // store the offset of this sample into the per-fragment offset buffer
        uint oldOffset = imageAtomicExchange( perFragmentOffset, ivec2( gl_FragCoord.xy ), newOffset );

        // and store the alpha-adjusted sample into the samplesBuffer
        color.rgb *= color.a;
        imageStore( samplesBuffer, int(newOffset), uvec4( packUnorm4x8( color ), floatBitsToUint( gl_FragCoord.z ), oldOffset, 0 ) );
      }
    }
    Color = vec4( 0 );    // the actual color doesn't matter, as it is calculated in the resolve pass
  }
  else
  {
    Color = color;
  }
}
