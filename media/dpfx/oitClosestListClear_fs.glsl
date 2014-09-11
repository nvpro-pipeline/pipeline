// main() of the clear pass of OITClosestList (right before the transparent pass)

#version 420

layout(size1x32) uniform restrict writeonly uimage1D  counterAccu;        // 1-element image to count all the samples
layout(size1x32) uniform restrict writeonly uimage2D  perFragmentOffset;  // 2D image sized as the view holding the base offset per fragment

layout(location = 0, index = 0) out vec4 Color;

void main()
{
  ivec2 screenPos = ivec2( gl_FragCoord.xy );

  // clear the counterAccu just once
  if ( screenPos == ivec2( 0, 0 ) )
  {
    imageStore( counterAccu, 0, uvec4( 0 ) );
  }

  imageStore( perFragmentOffset, screenPos, uvec4( 0xFFFFFFFF ) );

  Color = vec4( 0 );   // color doesn't matter
}
