// main() of the resolve pass of OITClosestList (right after the transparent pass)

#version 420

// format specifier in the source, which is replaced by some integer value in TransparencyManagerOITClosestArray.cpp
#define MAX_SAMPLE_COUNT %s

layout(size4x32) uniform restrict uimageBuffer  samplesBuffer;      // image buffer sized to hold samples up to its size
layout(size1x32) uniform restrict uimage2D      perFragmentOffset;  // 2D image sized as the view holding the base offset per fragment

layout(location = 0, index = 0) out vec4 Color;

// bubbleSort working on a local buffer of samples
void bubbleSort( inout uvec2 samplesList[MAX_SAMPLE_COUNT], int n )
{
  bool bubbled = true;
  int step = 1;
  for ( int i=n-1 ; 0<i && bubbled ; i-=step )
  {
    bubbled = false;
    for ( int j=0 ; j<i ; ++j )
    {
      // sort back-to-front (far-to-near)
      if ( uintBitsToFloat( samplesList[j].y ) < uintBitsToFloat( samplesList[j+1].y ) )
      {
        uvec2 temp = samplesList[j+1];
        samplesList[j+1] = samplesList[j];
        samplesList[j] = temp;
        bubbled = true;
        step = i - j;
      }
    }
  }
}

void main()
{
  vec4 color = vec4( 0.0f, 0.0f, 0.0f, 0.0f );    // initialize color with transparent black

  // the the first up to MAX_SAMPLE_COUNT samples of this fragment from the samplesBuffer
  // the offset latest sample is in perFragmentOffset, while the offset of the previous
  // sample is in the z-value of in the samplesBuffer
  int samplesCount = 0;
  uvec2 samplesList[MAX_SAMPLE_COUNT];
  uint offset = imageLoad( perFragmentOffset, ivec2( gl_FragCoord.xy ) ).x;
  while ( ( offset != 0xFFFFFFFF ) && ( samplesCount < MAX_SAMPLE_COUNT ) )
  {
    uvec4 stored = imageLoad( samplesBuffer, int(offset) );
    samplesList[samplesCount] = stored.xy;
    offset = stored.z;
    ++samplesCount;
  }

  // if there's at lest on sample on this fragment
  if ( 1 < samplesCount )
  {
    bubbleSort( samplesList, samplesCount );

    while ( offset != 0xFFFFFFFF )
    {
      // if we get here, the samplesList is filled, but there are still samples to handle!
      // -> accumulate the farthest samples without ordering into color
      uvec4 stored = imageLoad( samplesBuffer, int(offset) );
      vec4 fragColor;
      if ( uintBitsToFloat( stored.y ) < uintBitsToFloat( samplesList[0].y ) )
      {
        // the fragment is closer than the farthest one in the list -> accumulate farthest (first) in list
        // and insert fragment, shifting every entry in samplesList which is farther than the fragment
        fragColor = unpackUnorm4x8( samplesList[0].x );
        int i;
        for ( i=1 ; i<MAX_SAMPLE_COUNT && ( uintBitsToFloat( stored.y ) < uintBitsToFloat( samplesList[i].y ) ) ; i++ )
        {
          samplesList[i-1] = samplesList[i];
        }
        samplesList[i-1] = stored.xy;
      }
      else
      {
        // the fragment is farther than the farthest one in the list
        // -> just accumulate it and don't change the samplesList
        fragColor = unpackUnorm4x8( stored.x );
      }
      color += fragColor * fragColor.a;
      offset = stored.z;
    }

    // clamp the color to [0,1], in case it holds the accumulated farthest fragments
    color = clamp( color, 0.0f, 1.0f );
  }

  // blend the closest samples onto the color, from back to front
  for ( int i=0 ; i<samplesCount ; i++ )
  {
    vec4 colorLayer = unpackUnorm4x8( samplesList[i].x );
    color = color * ( 1 - colorLayer.a ) + colorLayer * colorLayer.a;
  }

  Color = color;    // the resulting color is blended on top of any opaque sample
}
