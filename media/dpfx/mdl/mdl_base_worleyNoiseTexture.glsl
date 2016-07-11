_base_textureReturn mdl_base_worleyNoiseTexture( in _base_textureCoordinateInfo uvw, in vec3 color1, in vec3 color2, in float size, in int mode, in int metric, in bool applyMarble, in bool applyDent
                                               , in vec3 noiseDistortion, in float noiseThresholdHigh, in float noiseThresholdLow, in float noiseBands, in float stepThreshold, in float edge )
{
  vec3 scaledPos = uvw.position / size;
  float noise = applyNoiseModifications( worleyNoise( scaledPos, noiseDistortion, stepThreshold, mode, metric, 1.0f )
                                       , scaledPos.x, applyMarble, applyDent, noiseThresholdHigh, noiseThresholdLow, noiseBands );

  _base_textureReturn tr;
  tr.tint = mix( color1, color2, noise );
  tr.mono = average( tr.tint );
  return( tr );
}

