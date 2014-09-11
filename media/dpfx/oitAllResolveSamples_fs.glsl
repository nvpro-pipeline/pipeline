// main() of the resolve pass of OITAll (right after the transparent pass)

#version 420

layout(size1x32) uniform restrict readonly uimage2D perFragmentCount;   // 2D image sized as the view to count the samples per fragment
layout(size1x32) uniform restrict readonly uimage2D perFragmentOffset;  // 2D image sized as the view holding the base offset per fragment
layout(size2x32) uniform restrict uimageBuffer      samplesBuffer;      // image buffer sized to hold color and depth on all samples on all fragments

layout(location = 0, index = 0) out vec4 Color;

// bubbleSort working directly on the uimageBuffer samples
// might be slower than sorting in a local buffer, but can be done with dynamic size
void bubbleSort( int offset, unsigned int n )
{
  bool bubbled = true;
  int step = 1;
  for ( int i=int(n)-1 ; 0<i && bubbled ; i-=step )
  {
    bubbled = false;
    uvec4 nextSample = imageLoad( samplesBuffer, offset );
    for ( int j=0 ; j<i ; ++j )
    {
      // sort back-to-front (far-to-near)
      uvec4 thisSample = nextSample;
      nextSample = imageLoad( samplesBuffer, offset + j + 1 );
      if ( uintBitsToFloat( thisSample.y ) < uintBitsToFloat( nextSample.y ) )
      {
        imageStore( samplesBuffer, offset + j, nextSample );
        imageStore( samplesBuffer, offset + j + 1, thisSample );
        nextSample = thisSample;
        bubbled = true;
        step = i - j;
      }
    }
  }
}

void main()
{
  vec4 color = vec4( 0.0f, 0.0f, 0.0f, 0.0f );    // initialize color with transparent black

  ivec2 screenPos = ivec2( gl_FragCoord.xy );

  unsigned int count  = imageLoad( perFragmentCount, screenPos ).x;
  if ( 0 < count )                                // if there's a tranparent sample on this fragment
  {
    int offset = int(imageLoad( perFragmentOffset, screenPos ).x);

    bubbleSort( offset, count );                  // sort the samples on this fragment

    for ( int i=0 ; i<count ; i++ )               // blend them from back to front
    {
      vec4 colorLayer = unpackUnorm4x8( imageLoad( samplesBuffer, offset + i ).x );
      color = color * ( 1 - colorLayer.a ) + colorLayer * colorLayer.a;
    }
  }

  Color = color;                                  // the resulting color is blended on top of any opaque sample
}
