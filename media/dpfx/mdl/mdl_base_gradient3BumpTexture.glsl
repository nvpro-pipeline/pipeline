vec3 mdl_base_gradient3BumpTexture( in int gradientMode, in float[3] gradientPositions, in vec3[3] gradientColors, in int[3] interpolationModes
                                  , in _base_textureCoordinateInfo uvw, in float distortion, in float scale, in vec3 normal )
{
  const float delta = 0.0025f;    // magic, looks good with this value

  float position = gradientGetPosition( gradientMode, uvw.position.xy );
  vec3 r0 = mdl_base_gradient3Recolor( gradientPositions, gradientColors, interpolationModes, 0, distortion, position ).tint;

  position = gradientGetPosition( gradientMode, uvw.position.xy + delta * uvw.tangentU.xy );
  vec3 r1 = mdl_base_gradient3Recolor( gradientPositions, gradientColors, interpolationModes, 0, distortion, position ).tint;

  position = gradientGetPosition( gradientMode, uvw.position.xy + delta * uvw.tangentV.xy );
  vec3 r2 = mdl_base_gradient3Recolor( gradientPositions, gradientColors, interpolationModes, 0, distortion, position ).tint;

  return( normalize( normal - average( r1 - r0 ) * uvw.tangentU - average( r2 - r0 ) * uvw.tangentV ) );
}

