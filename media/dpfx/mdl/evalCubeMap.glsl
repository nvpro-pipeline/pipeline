vec3 evalCubeMap( in vec3 R, in float roughness )
{
#if 0

  // simply return the cube map value
  return( texture( sys_EnvironmentSampler, R ).rgb );

#else

  // explicitly determine the lod level into the cube map, using
  // roughness as the weight on the level
  ivec2 cubeMapSize = textureSize( sys_EnvironmentSampler, 0 );
  float cubeMapLevels = log2( max( cubeMapSize.x, cubeMapSize.y ) ) + 1;
  float clearLevel = textureQueryLod( sys_EnvironmentSampler, R ).x;
  float roughLevel = mix( clearLevel, cubeMapLevels, roughness );

  // then return the cube map on the determined LOD
  // -> image is nicely blurred, but with high roughness, the
  //    cube map edges are clearly visible
  return( textureLod( sys_EnvironmentSampler, R, roughLevel ).rgb );

#endif
}

