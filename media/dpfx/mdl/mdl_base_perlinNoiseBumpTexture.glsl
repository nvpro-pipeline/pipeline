
vec3 mdl_base_perlinNoiseBumpTexture( in _base_textureCoordinateInfo uvw, in float factor, in float size, in bool applyMarble, in bool applyDent
                                    , in float noisePhase, in int noiseLevels, in bool absoluteNoise, bool ridgedNoise, in vec3 noiseDistortion
                                    , in float noiseThresholdHigh, in float noiseThresholdLow, in float noiseBands, in vec3 normal )
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
    results[i] = applyNoiseModifications( summedPerlinNoise( scaledPosition, noisePhase, noiseLevels, noiseDistortion, absoluteNoise, ridgedNoise )
                                        , scaledPosition.x, applyMarble, applyDent, noiseThresholdHigh, noiseThresholdLow, noiseBands );
  }

  float bump_factor = -factor;

  vec3 tangentV = normalize(cross(normal, uvw.tangentU)); 
  vec3 tangentU = normalize(cross(uvw.tangentV, normal));
  return( normalize( normal * ( abs( ( results[3] - results[0] ) * bump_factor ) + 1.0f )
                      + tangentU * ( results[1] - results[0] ) * bump_factor
                      + tangentV * ( results[2] - results[0] ) * bump_factor ) );
}

