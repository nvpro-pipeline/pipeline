// emitColor is the function called at the end of the fragment shader to emit the calculated color
// There are a couple of emitColor functions (like in emitColorDepth.glsl, emitColorOITAllCounter.glsl, etc.) which reflect
// the various needs of the various algorithms.

// Version of emitColor() used in OITClosestArray in the non-depth pass for transparent fragments.
// This function stores first sys_OITDepth samples per fragment in the samples buffer.
// If there comes an additional sample on this fragment which is closer than the farthest sample
// encountered before, it replaces that sample in the samples buffer.
// That farthest sample (or the new one that was farther than the farthest in the samples buffer)
// is accumulated into the perFragmentSamplesAccu, without correct depth sorting.
// As such samples are behind sys_OITDepth correctly sorted transparent samples, it is assumed
// that that error can hardly be noticed.

// enable early depth test to prevent the expensive fragment shader on samples that
// are already covered by opaque samples
layout(early_fragment_tests) in;

layout(size1x32) uniform coherent uimage2D perFragmentCount;      // 2D image sized as the view to count the samples stored per fragment; needs to be coherent !!
layout(size1x32) uniform restrict uimage2D perFragmentIndex;      // 2D image sized as the view to count the samples encounterd per fragment
layout(size1x32) uniform restrict uimage2D perFragmentSpinLock;   // 2D image sized as the view to hold a spin lock per fragment, guarding the storing of the sample
layout(size2x32) uniform restrict uimageBuffer samplesBuffer;     // image buffer sized to hold up to sys_OITDepth samples per fragment
layout(size4x32) uniform restrict image2D perFragmentSamplesAccu; // 2D image sized as the view to hold the farthest transparent samples behind the nearest sys_OITDepth samples

layout(location = 0, index = 0) out vec4 Color;

void emitColor( in vec4 color )
{
  color = clamp( color, 0.0f, 1.0f );
  if ( sys_TransparentPass )
  {
    if ( 1.0f / 255.0f <= color.a )    // just do something if the transparency of the current color can make a difference
    {
      color.rgb *= color.a;

      // get the base offset into the samples buffer by using the screen pos and the image size of view-sized buffer
      // Note: it's substantially faster to store the samples "in slices" (viewport-sized slices with one sample per fragment)
      // than "in chunks" (per fragment one chunk of size sys_OITDepth)!
      ivec2 screenPos = ivec2( gl_FragCoord.xy );
      ivec2 viewport = imageSize( perFragmentCount );
      int viewSize = viewport.x * viewport.y;
      int listPos = screenPos.y * viewport.x + screenPos.x;

      // atomically add one to the perFragmentIndex to set and get the number of samples handled on this fragment
      uint index = imageAtomicAdd( perFragmentIndex, screenPos, uint(1) );
      if ( index < sys_OITDepth )
      {
        // if there are less than sys_OITDepth samples, just store the color and depth into the samples buffer
        imageStore( samplesBuffer, int(index) * viewSize + listPos, uvec4( packUnorm4x8( color ), floatBitsToUint( gl_FragCoord.z ), 0, 0 ) );

        // and atomically increment the number of fragments stored
        imageAtomicAdd( perFragmentCount, screenPos, uint(1) );
      }
      if ( index >= sys_OITDepth )    // you need to explicitly check here again!! If you just use 'else', the following while-loop never ends
      {
        // if there are more than sys_OITDepth samples, wait 'til the fragment count has been settled
        // Note: to be precise, that isn't really sufficient to be sure, that the corresponding samples are stored into the samples
        //       buffer... but we pretend, it is.
        while ( imageLoad( perFragmentCount, screenPos ).x < sys_OITDepth )
          ;

        // loop 'til we're done with storing the current sample
        bool done = false;
        while ( ! done )
        {
          // determine (non-atomically) the index of the current furthest sample for this fragment
          uint furthestIndex = uint(0);
          float furthestDepth = -2.0f;    // just some depth that's really close
          for ( uint i=uint(0) ; i<sys_OITDepth ; i++ )
          {
            float storedDepth = uintBitsToFloat( imageLoad( samplesBuffer, int(i) * viewSize + listPos ).y );
            if ( furthestDepth < storedDepth )
            {
              furthestIndex = i;
              furthestDepth = storedDepth;
            }
          }
          int listIdx = int(furthestIndex) * viewSize + listPos;

          // lock this fragment
          if ( imageAtomicExchange( perFragmentSpinLock, screenPos, uint(1) ) == uint(0) )
          {
            // Check if the sample at the furthestIndex still is the furthest. This is needed, as the
            // lock is as tight to the storage operation as possible and thus the samples might have
            // changed between determining the furthest sample above and the lock
            uvec2 currentFrag = imageLoad( samplesBuffer, listIdx ).xy;
            if ( uintBitsToFloat( currentFrag.y ) == furthestDepth )
            {
              // if the current sample is closer than the furthest sample store the current color and depth
              // replacing the furthest sample, and get the color of the furthest sample for accumulation
              if ( gl_FragCoord.z < furthestDepth )
              {
                imageStore( samplesBuffer, listIdx, uvec4( packUnorm4x8( color ), floatBitsToUint( gl_FragCoord.z ), 0, 0 ) );
                color = unpackUnorm4x8( currentFrag.x );   // use the color dropped off the samples buffer for the perFragmentSamplesAccu buffer
              }

              // hybrid transparency handling: accumulate any fragments that dropped off the samples buffer,
              // or are behind the furthest entry in the samples buffer, into the perFragmentSamplesAccu buffer
              vec4 htColor = imageLoad( perFragmentSamplesAccu, screenPos );
              imageStore( perFragmentSamplesAccu, screenPos, htColor + color * color.a );

              done = true;
            }

            // unlock the fragment
            imageStore( perFragmentSpinLock, screenPos, uvec4( 0 ) );
          }
        }
      }
    }
    Color = vec4( 0 );    // the actual color doesn't matter, as it is calculated in the resolve pass
  }
  else
  {
    Color = color;
  }
}
