
int texRemap( int size, int offset, int texIn, int wrap )
{
  int texi = texIn;

  // Wrap or Clamp
  if ( wrap == wrap_clamp )
  {
    texi = clamp( texi, 0, size - 1 );
  }
  else
  {
    int s = texi < 0 ? 1 : 0; // extract sign to handle all < 0 magic below
    int d = texi / size;
    texi = texi % size;  // texi -= d * size;
    int alternate = ( wrap == wrap_mirrored_repeat ) ? 1 : 0;
    int a = alternate & (d ^ s) & 1;
    bool altu = (a != 0);
    if ( altu )   // if alternating, negative tex has to be flipped, could also be: (tex^-a)+a (-m = (m^-1)+1)
    {
      texi = -texi;
    }
    if ( s != a ) // "otherwise" negative tex will be padded back to positive
    {
      texi += size - 1;
    }
  }

  // Crop
  return( texi + offset );
}

ivec2 texRemap( ivec2 size, ivec2 offset, vec2 tex, int wrapU,  int wrapV )
{
  return( ivec2( texRemap( size.x, offset.x, int( tex.x ), wrapU )
               , texRemap( size.y, offset.y, int( tex.y ), wrapV ) ) );
}

float interpolateTexelSpace( sampler2D sampler, vec4 st, ivec4 texi, int bumpSource )
{
  vec4 tex = texelFetch( sampler, ivec2( texi.x, texi.y ), 0 ) * st.z
           + texelFetch( sampler, ivec2( texi.z, texi.y ), 0 ) * st.y
           + texelFetch( sampler, ivec2( texi.x, texi.w ), 0 ) * st.w
           + texelFetch( sampler, ivec2( texi.z, texi.w ), 0 ) * st.x;
  return( monoChannel( tex, bumpSource ) );
}

// compute a normal based on a heightfield style bump texture
vec3 mdl_base_fileBumpTexture( in texture2D tex, in float factor, in int bumpSource, in _base_textureCoordinateInfo uvw
                             , in vec2 cropU, in vec2 cropV, in int wrapU, in int wrapV, in vec3 normal, in bool clip )
{
  vec3 ret;
  if ( clip && ( ( ( wrapU == wrap_clamp ) && ( ( uvw.position.x < 0.0f ) || ( 1.0f < uvw.position.x ) ) )
              || ( ( wrapV == wrap_clamp ) && ( ( uvw.position.y < 0.0f ) || ( 1.0f < uvw.position.y ) ) ) ) )
  {
    ret = normal;
  }
  else
  {
    ivec2 texSize = textureSize( tex.sampler, 0 );
    ivec2 cropOffset = texSize * ivec2( cropU.x, cropV.x );
    ivec2 cropTexSize = texSize * ivec2( cropU.y - cropU.x, cropV.y - cropV.x );
    vec2 tex2 = uvw.position.xy * cropTexSize;
    ivec2 texi0 = texRemap( cropTexSize, cropOffset, tex2 - 1.0f, wrapU, wrapV );
    ivec2 texi1 = texRemap( cropTexSize, cropOffset, tex2, wrapU, wrapV );
    ivec2 texi2 = texRemap( cropTexSize, cropOffset, tex2 + 1.0f, wrapU, wrapV );
    ivec2 texi3 = texRemap( cropTexSize, cropOffset, tex2 + 2.0f, wrapU, wrapV );

    vec2 lerp = tex2 - floor( tex2 );
    lerp *= lerp * lerp * ( lerp * ( lerp * 6.0f - 15.0f ) + 10.0f );   //smootherstep
    vec4 st = vec4( lerp.x * lerp.y, lerp.x * ( 1.0f - lerp.y ), ( 1.0f - lerp.x ) * ( 1.0f - lerp.y ), ( 1.0f - lerp.x ) * lerp.y );

    vec2 bump = factor * vec2( interpolateTexelSpace( tex.sampler, st, ivec4( texi0.x, texi1.y, texi1.x, texi2.y ), bumpSource )
                             - interpolateTexelSpace( tex.sampler, st, ivec4( texi2.x, texi1.y, texi3.x, texi2.y ), bumpSource )
                             , interpolateTexelSpace( tex.sampler, st, ivec4( texi1.x, texi0.y, texi2.x, texi1.y ), bumpSource )
                             - interpolateTexelSpace( tex.sampler, st, ivec4( texi1.x, texi2.y, texi2.x, texi3.y ), bumpSource ) );

    ret = normalize( normal + uvw.tangentU * bump.x + uvw.tangentV * bump.y );
  }
  return( ret );
}
