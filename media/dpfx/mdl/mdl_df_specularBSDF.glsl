vec4 mdl_df_specularBSDF( in vec3 tint, int mode, in vec3 normal )
{
  const float shininess = 256.0f;

  vec4 rgba = vec4( 0.0f, 0.0f, 0.0f, 1.0f );
  float cosTheta = dot( normal, lightDir );
  if ( 0.0f < cosTheta )
  {
    if ( ( mode == scatter_reflect ) || ( mode == scatter_reflect_transmit ) )
    {
      vec3 R = reflect( -lightDir, normal );
      float cosAlpha = max( 0.0f, dot( R, viewDir ) );
      float shine = pow( cosAlpha, shininess );
      rgba.rgb = shine * lightSpecular * tint;
    }
  }
  if ( ( mode == scatter_transmit ) || ( mode == scatter_reflect_transmit ) )
  {
    // check against total reflection
    vec3 R = refract( -viewDir, normal, materialIOR );
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
      rgba.a = 1.0f - mdl_math_luminance( tint );
    }
  }
  return( rgba );
}


vec4 mdl_df_specularBSDFEnvironment( in vec3 tint, int mode, in vec3 normal )
{
  vec3 rgb = vec3( 0.0f, 0.0f, 0.0f );
  if ( sys_EnvironmentSamplerEnabled )
  {
    lightDir = reflect( -viewDir, normal );
    lightDiffuse = evalEnvironmentMap( lightDir, 0.0f );
    lightSpecular = lightDiffuse;
    materialIOR = 1.0f;
    rgb = mdl_df_specularBSDF( tint, scatter_reflect, normal ).rgb;
  }
  return( vec4( rgb, 1.0f ) );
}
