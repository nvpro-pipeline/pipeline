vec4 specularReflectionBSDF( in vec3 N, in vec3 L, in vec3 lightSpecular, in vec3 materialSpecular )
{
  const float shininess = 256.0f;

  vec4 rgba = vec4( 0.0f, 0.0f, 0.0f, 1.0f );
  float cosTheta = dot( N, L );
  if ( 0.0f < cosTheta )
  {
    vec3 R = reflect( -L, N );
    float cosAlpha = max( 0.0f, dot( R, viewDir ) );
    float shine = pow( cosAlpha, shininess );
    rgba.rgb = shine * lightSpecular * materialSpecular;
  }
  return( rgba );
}

vec4 specularReflectionBSDFEnvironment( in vec3 N, in vec3 materialSpecular )
{
  vec3 rgb = vec3( 0.0f, 0.0f, 0.0f );
  if ( sys_EnvironmentSamplerEnabled )
  {
    vec3 R = reflect( -viewDir, N );
    rgb = evalEnvironmentMap( R, 0.0f );
    rgb = specularBSDF( N, R, rgb, rgb, materialSpecular ).rgb;
  }
  return( vec4( rgb, 1.0f ) );
}
