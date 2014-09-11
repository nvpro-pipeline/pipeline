// main() of the resolve counters pass of OITAll (right after the depth pass)

#version 420

layout(size1x32) uniform restrict uimage1D            counterAccu;        // 1-element image to count all the samples
layout(size1x32) uniform restrict uimage2D            perFragmentCount;   // 2D image sized as the view to count the samples per fragment
layout(size1x32) uniform restrict writeonly uimage2D  perFragmentOffset;  // 2D image sized as the view holding the base offset per fragment

layout(location = 0, index = 0) out vec4 Color;

void main()
{
  ivec2 screenPos = ivec2( gl_FragCoord.xy );

  unsigned int count = imageLoad( perFragmentCount, screenPos ).x;

  // if there is at least one sample on this fragment
  if ( count != 0 )
  {
    unsigned int offset = imageAtomicAdd( counterAccu, 0, count );    // add that count to the counterAcc
    imageStore( perFragmentOffset, screenPos, uvec4( offset ) );      // store the resulting offset for this fragment

    imageStore( perFragmentCount, screenPos, uvec4( 0 ) );            // and clear the per-fragment count
  }

  Color = vec4( 0 );   // color doesn't matter
}
