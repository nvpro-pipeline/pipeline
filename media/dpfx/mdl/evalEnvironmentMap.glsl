vec3 evalEnvironmentMap( in vec3 R, in float roughness )
{
  // convert R to 2D
  vec2 tc = vec2( ( atan( R.x, -R.z ) + PI ) / ( 2.0f * PI ), acos( -R.y ) / PI );

  // Due to discontinuity of atan2 at PI and -PI, standard texture sampling
  // produces a seam of pixels sampled from the lowest mip level because
  // dFdx and dFdy of the latitude are artificially high.  To remedy this we
  // approximate the magnitude of the derivative of latitude analytically
  // and use our own gradients.

  // Compute dx and dy
  // The 2*PI factor transforms from a derivative in radians to a derivative in texture space.
  vec2 dx = vec2( length( dFdx( R.xy ) ) / ( 2 * PI ), dFdx( tc.y ) );
  vec2 dy = vec2( length( dFdy( R.xy ) ) / ( 2 * PI ), dFdy( tc.y ) );

  // determine the corresponding lod level
  ivec2 envMapSize = textureSize( sys_EnvironmentSampler, 0 );
  vec2 lodMapSize = envMapSize * max( dx, dy );
  float lodLevel = log2( max( lodMapSize.x, lodMapSize.y ) );

  // determine the maximal lod level
  float maxLevel = log2( max( envMapSize.x, envMapSize.y ) );

  // mix lodLevel and maxLevel with roughness
  float roughLevel = mix( lodLevel, maxLevel, roughness );

  return( textureLod( sys_EnvironmentSampler, tc, roughLevel ).rgb );
}
