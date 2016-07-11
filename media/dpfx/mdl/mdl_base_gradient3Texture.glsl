_base_textureReturn mdl_base_gradient3Texture( in int gradientMode, in float[3] gradientPositions, in vec3[3] gradientColors, in int[3] interpolationModes, in _base_textureCoordinateInfo uvw, in float distortion )
{
  float position = gradientGetPosition( gradientMode, uvw.position.xy );
  return( mdl_base_gradient3Recolor( gradientPositions, gradientColors, interpolationModes, 0, distortion, position ) );
}

