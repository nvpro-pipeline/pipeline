vec3 mdl_base_flowNoiseBumpTexture( in _base_textureCoordinateInfo uvw, in float factor, in float size, in float phase, in int levels, in bool absoluteNoise
                                  , in float levelGain, in float levelScale, in float levelProgressiveUScale, in float levelProgressiveVMotion, in vec3 normal )
{
  float delta = 0.1f * size;

  float r0 = flowNoise( uvw.position.xy / size, phase, levels, absoluteNoise, levelGain, levelScale, levelProgressiveUScale, levelProgressiveVMotion );
  float r1 = flowNoise( ( uvw.position + delta * uvw.tangentU ).xy / size, phase, levels, absoluteNoise, levelGain, levelScale, levelProgressiveUScale, levelProgressiveVMotion );
  float r2 = flowNoise( ( uvw.position + delta * uvw.tangentV ).xy / size, phase, levels, absoluteNoise, levelGain, levelScale, levelProgressiveUScale, levelProgressiveVMotion );
  float r3 = flowNoise( ( uvw.position + delta * normal ).xy / size, phase, levels, absoluteNoise, levelGain, levelScale, levelProgressiveUScale, levelProgressiveVMotion );

  return( normalize( normal - factor * ( ( r1 - r0 ) * uvw.tangentU + ( r2 - r0 ) * uvw.tangentV + ( r3 - r0 ) * normal ) ) );
}
