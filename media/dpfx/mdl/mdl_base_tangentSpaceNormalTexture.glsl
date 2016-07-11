
//interpreting the color values of a bitmap as a vector in tangent space
vec3 mdl_base_tangentSpaceNormalTexture( in sampler2D sampler, in float gamma, in float factor, in bool flipTangentU, in bool flipTangentV, in _base_textureCoordinateInfo uvw
                                       , in vec2 cropU, in vec2 cropV, in int wrapU, in int wrapV, in bool clip, in float scale, in float offset )
{
  vec3 ret;
  if ( clip && ( ( ( wrapU == wrap_clamp ) && ( ( uvw.position.x < 0.0f ) || ( 1.0f < uvw.position.x ) ) )
              || ( ( wrapV == wrap_clamp ) && ( ( uvw.position.y < 0.0f ) || ( 1.0f < uvw.position.y ) ) ) ) )
  {
    ret = stateNormal;
  }
  else
  {
    // if we mirror repeat a tangent space texture, tangent space needs to be flipped for every other tile
    bool flipU = flipTangentU;
    bool flipV = flipTangentV;

    if ( wrapU == wrap_mirrored_repeat )
    {
      if (    ( ( 0.0f < uvw.position.x ) && ( int( uvw.position.x ) % 2 == 1 ) )
          ||  ( ( uvw.position.x < 0.0f ) && ( int( uvw.position.x ) % 2 == 0 ) ) )
      {
        flipU = !flipU;
      }
    }
    if ( wrapV == wrap_mirrored_repeat )
    {
      if (    ( ( 0.0f < uvw.position.y ) && ( int( uvw.position.y ) % 2 == 1 ) )
          ||  ( ( uvw.position.y < 0.0f ) && ( int( uvw.position.y ) % 2 == 0 ) ) )
      {
        flipV = !flipV;
      }
    }

    vec3 tangentU = normalize( flipU ? -uvw.tangentU : uvw.tangentU );
    vec3 tangentV = normalize( flipV ? -uvw.tangentV : uvw.tangentV );
    vec3 tangent = factor * ( 2.0f * texture( sampler, uvw.position.xy ).xyz - 1.0f );
    vec3 normal = normalize( cross( tangentU, tangentV ) );
    ret = normalize( tangent.x * tangentU + tangent.y * tangentV + ( tangent.z + ( 1.0f - factor ) ) * normal );
  }

  return( ret );
}
