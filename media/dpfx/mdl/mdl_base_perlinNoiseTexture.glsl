_base_textureReturn mdl_base_perlinNoiseTexture( in _base_textureCoordinateInfo uvw, in vec3 color1, in vec3 color2, in float size, in bool applyMarble, in bool applyDent
                                               , in float noisePhase, in int noiseLevels, in bool absoluteNoise, bool ridgedNoise, in vec3 noiseDistortion
                                               , in float noiseThresholdHigh, in float noiseThresholdLow, in float noiseBands )
{
  vec3 scaledPos = uvw.position / size;
  float noise = applyNoiseModifications( summedPerlinNoise( scaledPos, noisePhase, noiseLevels, noiseDistortion, absoluteNoise, ridgedNoise )
                                       , scaledPos.x, applyMarble, applyDent, noiseThresholdHigh, noiseThresholdLow, noiseBands );

  _base_textureReturn tr;
  tr.tint = mix( color1, color2, noise );
  tr.mono = average( tr.tint );
  return( tr );
}

