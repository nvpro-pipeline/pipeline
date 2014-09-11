// main() of the clear pass of OITClosestArray (right before the transparent pass)

#version 420

layout(size1x32) uniform restrict writeonly uimage2D perFragmentCount;      // 2D image sized as the view to count the samples stored per fragment
layout(size1x32) uniform restrict writeonly uimage2D perFragmentIndex;      // 2D image sized as the view to count the samples encounterd per fragment
layout(size1x32) uniform restrict writeonly uimage2D perFragmentSpinLock;   // 2D image sized as the view to hold a spin lock per fragment, guarding the storing of the sample
layout(size4x32) uniform restrict writeonly image2D perFragmentSamplesAccu; // 2D image sized as the view to hold the farthest transparent samples behind the nearest sys_OITDepth samples

layout(location = 0, index = 0) out vec4 Color;

void main()
{
  ivec2 screenPos = ivec2( gl_FragCoord.xy );

  // clear all the pre-fragment images with unsigned values
  uvec4 uBlack = uvec4( 0 );
  imageStore( perFragmentCount,    screenPos, uBlack );
  imageStore( perFragmentIndex,    screenPos, uBlack );
  imageStore( perFragmentSpinLock, screenPos, uBlack );

  // hybrid transparency handling: clear the perFragmentSamplesAccu
  vec4 black = vec4( 0 );
  imageStore( perFragmentSamplesAccu, screenPos, black );

  Color = vec4( 0 );   // color doesn't matter
}
