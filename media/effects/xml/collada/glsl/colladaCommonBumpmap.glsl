
vec3 bumpmap( vec3 ns )
{
  if ( 0 <= bumpTC )
  {
    // TODO Add parameter support for bumpiness
    float bumpiness = 1.0;

    ivec2 texSize = textureSize( bumpSampler, 0 );

    // Offsets are not transformed on purpose to stay in texel space.
    vec2 offset = 1.0 / vec2( texSize.xy );

    // Central differencing, 
    vec2 texCoord = evaluateTexCoord( bumpTC );
    float left   = texture( bumpSampler, vec2( texCoord.x - offset.x, texCoord.y)).r;
    float right  = texture( bumpSampler, vec2( texCoord.x + offset.x, texCoord.y)).r;
    float bottom = texture( bumpSampler, vec2( texCoord.x, texCoord.y - offset.y)).r;
    float top    = texture( bumpSampler, vec2( texCoord.x, texCoord.y + offset.y)).r;

    // This resulting vector v.xyz was derived from an expanded cross product.
    vec3 v = vec3( left - right, bottom - top, 1.0 );
    v.xy *= bumpiness;  // With this a scale of 0.0 eliminates the bump effect! 
    v = normalize(v);

    return v.x * normalize( varTangent ) + v.y * normalize( varBinormal ) + v.z * ns;
  }
  else
  {
    return ns;
  }
}
