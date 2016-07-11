vec4 mdl_df_thinFilm( in float thickness, in vec3 ior, in vec4 base )
{
  // wavelength of rgb found at http://en.wikipedia.org/wiki/Visible_spectrum
  const vec3 oneOverLambda = vec3( 1.0f / 685.0f, 1.0f / 533.0f, 1.0f / 473.0f );

  // get the refraction vector T, and check for total internal reflection
  vec3 T0 = refract( -viewDir, stateNormal, ior.r );
  vec3 T1 = refract( -viewDir, stateNormal, ior.g );
  vec3 T2 = refract( -viewDir, stateNormal, ior.b );

  vec3 cosTheta2 = vec3( dot( -stateNormal, T0 ), dot( -stateNormal, T1 ), dot( -stateNormal, T2 ) );   // angle between negative stateNormal and refraction vector T
  vec3 opd = 2.0f * ior * thickness * cosTheta2;                                                        // optical path difference, as found at https://en.wikipedia.org/wiki/Thin-film_interference
  vec3 m = opd * oneOverLambda;                                                                         // opd in multiples of wavelenghts

  // adjust for total reflection
  if ( m.r == 0.0f )
  {
    m.r = 0.5f;
  }
  if ( m.g == 0.0f )
  {
    m.g = 0.5f;
  }
  if ( m.b == 0.0f )
  {
    m.b = 0.5f;
  }

  // with 1 < ior, we have a face shift of 180 degree at the upper boundary of the film, then,
  // with fract(m) == 0.0, we have destructive interference
  // with fract(m) == 0.5, we have constructive interference
  vec3 pd = fract(m);               // range [0.0,1.0)
  vec3 modulate = sin( PI * pd );   // range [0.0,1.0), with maximum at pd == 0.5
  return( vec4( modulate * base.rgb, base.a ) );
}

