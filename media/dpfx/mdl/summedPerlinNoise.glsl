
float summedPerlinNoise( in vec3 pos, in float time, in int terms, in vec3 turbulenceWeight, in bool absNoise, in bool ridged )
{
  float sum = 0.0f;
  float weight = ridged ? 0.625f : 1.0f;
  float prev = 1.0f;
  vec4 p = vec4( pos, time );
  while ( terms-- != 0 )
  {
    float noise = perlinNoise( p );
    noise = ridged ? square( 1.0f - abs( noise ) ) : ( absNoise ? abs( noise ) : noise );   // ridged offset = 1.0f, could be configurable
    sum += weight * prev * noise;
    p += p;         // frequency doubled, could be configurable
    weight *= 0.5f; // gain halfed, could be configurable
    if ( ridged )
    {
      prev = noise;
    }
  }

  if ( turbulenceWeight != vec3( 0.0f, 0.0f, 0.0f ) )
  {
    sum = sin( dot( pos, turbulenceWeight ) + sum );
  }

  if ( ! absNoise && !ridged )
  {
    sum = 0.5f * sum + 0.5f;    // scale [-1,1] to [0,1]
  }

  return( sum );
}
