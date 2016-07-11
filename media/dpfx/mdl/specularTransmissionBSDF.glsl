vec4 specularTransmissionBSDF( in vec3 N, in vec3 L, in vec3 lightSpecular, in vec3 materialSpecular, in float ior )
{
  const float shininess = 256.0f;

  vec4 rgba = vec4( 0.0f, 0.0f, 0.0f, 1.0f );
  float cosTheta = dot( N, L );

  // check against total reflection
  vec3 R = refract( -viewDir, N, ior );
  if ( R == vec3( 0.0f, 0.0f, 0.0f ) )
  {
    rgba.a = 1.0f;
  }
  else if ( mode == scatter_transmit )
  {
    rgba.a = 0.0f;
  }
  else
  {
    rgba.a = 1.0f - luminance( materialSpecular );
  }

  return( rgba );
}

vec4 specularBSDFEnvironment( in vec3 N, in vec3 materialSpecular, in float ior )
{
  vec3 rgb = vec3( 0.0f, 0.0f, 0.0f );
  if ( sys_EnvironmentSamplerEnabled )
  {
    vec3 R = reflect( -viewDir, N );
    rgb = evalEnvironmentMap( R, 0.0f );
    rgb = specularBSDF( N, R, rgb, rgb, 1.0f, materialSpecular, scatter_reflect ).rgb;
  }
  return( vec4( rgb, 1.0f ) );
}
