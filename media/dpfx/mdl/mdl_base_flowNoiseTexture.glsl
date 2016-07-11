_base_textureReturn mdl_base_flowNoiseTexture( in _base_textureCoordinateInfo uvw, in vec3 color1, in vec3 color2, in float size, in float phase, in int levels, in bool absoluteNoise
                                             , in float levelGain, in float levelScale, in float levelProgressiveUScale, in float levelProgressiveVMotion )
{
  _base_textureReturn tr;
  tr.tint = mix( color1, color2, flowNoise( uvw.position.xy / size, phase, levels, absoluteNoise, levelGain, levelScale, levelProgressiveUScale, levelProgressiveVMotion ) );
  tr.mono = 0.0f;
  return( tr );
}

