// main() of the resolve pass of OITClosestArray (right after the transparent pass)

// needs version 430 because of imageSize !
#version 430

// format specifier in the source, which is replaced by some integer value in TransparencyManagerOITClosestArray.cpp
#define MAX_SAMPLE_COUNT %s

layout(size1x32) uniform restrict readonly uimage2D     perFragmentCount;        // 2D image sized as the view to count the samples stored per fragment
layout(size4x32) uniform restrict readonly image2D      perFragmentSamplesAccu;  // 2D image sized as the view to hold the farthest transparent samples behind the nearest sys_OITDepth samples
layout(size2x32) uniform restrict readonly uimageBuffer samplesBuffer;           // image buffer sized to hold up to sys_OITDepth samples per fragment

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
  // determine the position in the samples buffer depending on screen position and view size
  ivec2 screenPos = ivec2( gl_FragCoord.xy );
  ivec2 viewport = imageSize( perFragmentCount );
  int viewSize = viewport.x * viewport.y;
  int listPos = screenPos.y * viewport.x + screenPos.x;

  // gather the samples of this fragment into a local buffer
  uvec2 samplesList[MAX_SAMPLE_COUNT];
  int fragmentCount = int( imageLoad( perFragmentCount, screenPos ).x );
  for ( int i=0 ; i<fragmentCount ; i++ )
  {
    samplesList[i] = imageLoad( samplesBuffer, i * viewSize + listPos ).xy;
  }

#if 1 < MAX_SAMPLE_COUNT
  // sort the samples (if there is more than one)
  if ( 1 < fragmentCount )
  {
    bubbleSort( samplesList, fragmentCount );
  }
#endif

  // accumulate back to front, assuming that the samplesList colors are in the [0,1]-range
  // that is we just have to clamp the perFragmentSamplesAccu
  // if accumulating from front to back, we'd have to clamp on each level
  vec4 color = clamp( imageLoad( perFragmentSamplesAccu, screenPos ), 0, 1 );
  for ( int i=0 ; i<fragmentCount ; i++ )
  {
    vec4 colorLayer = unpackUnorm4x8( samplesList[i].x );
    color = color * ( 1 - colorLayer.a ) + colorLayer * colorLayer.a;
  }

  Color = color;    // the resulting color is blended on top of any opaque sample
}
