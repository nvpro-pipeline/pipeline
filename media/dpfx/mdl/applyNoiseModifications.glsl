
float applyNoiseModifications( in float value, in float position, in bool applyMarble, in bool applyDent, in float noiseThresholdHigh, in float noiseThresholdLow, in float noiseBands )
{
  float result = value;

  if ( applyMarble )
  {
    // Classic Perlin marble function
    result = cos( position + result * 5.0f );  //!! 5.0f = magic
  }

  if ( applyDent )
  {
    result = cube( result );
  }

  if ( noiseBands != 1.0f )
  {
    // Create banding/stripes by using the fraction component only
    result *= noiseBands;
    result -= floor( result );
    result += pow( 1.0f - result, 20.0f );
  }

  if ( noiseThresholdLow < noiseThresholdHigh )
  {
    // clamp the noise
    result = clamp( ( result - noiseThresholdLow ) / ( noiseThresholdHigh - noiseThresholdLow ), 0.0f, 1.0f );
  }

  return( result );
}
