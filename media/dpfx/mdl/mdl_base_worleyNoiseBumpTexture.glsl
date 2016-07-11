
vec3 mdl_base_worleyNoiseBumpTexture( in _base_textureCoordinateInfo uvw, in float factor, in float size, in int mode, in int metric, in bool applyMarble, in bool applyDent
                                    , in vec3 noiseDistortion, in float noiseThresholdHigh, in float noiseThresholdLow, in float noiseBands, in float stepThreshold, in float edge, in vec3 normal )
{
  float delta = 0.1f * size / noiseBands; //!! magic

  vec3[4] offsets;
  offsets[0] = vec3(  0.0f,  0.0f,  0.0f );
  offsets[1] = vec3( delta,  0.0f,  0.0f );
  offsets[2] = vec3(  0.0f, delta,  0.0f);
  offsets[3] = vec3(  0.0f,  0.0f, delta );
  float[4] results;
  for ( int i=0 ; i<4 ; i++ )
  {
    vec3 scaledPosition = ( uvw.position + offsets[i] ) / size;
    results[i] = applyNoiseModifications( worleyNoise( scaledPosition, noiseDistortion, stepThreshold, mode, metric, 1.0f )
                                        , scaledPosition.x, applyMarble, applyDent, noiseThresholdHigh, noiseThresholdLow, noiseBands );
  }

  float bump_factor = -factor;

  return( normalize( normal    * ( abs( ( results[3] - results[0] ) * bump_factor ) + 1.0f )
                       + uvw.tangentU * ( results[1] - results[0] ) * bump_factor
                       + uvw.tangentV * ( results[2] - results[0] ) * bump_factor ) );
}
