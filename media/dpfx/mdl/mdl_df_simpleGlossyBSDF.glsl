vec4 mdl_df_simpleGlossyBSDF( in float roughnessU, in float roughnessV, in vec3 tint, in vec3 tangentU, int mode, in vec3 normal )
{
  vec4 rgba = vec4( 0.0f, 0.0f, 0.0f, 1.0f );

  float cosTheta = dot( normal, lightDir );
  if ( 0.0f < cosTheta )
  {
    float roughness = calculateRoughness( normal, roughnessU, roughnessV, tangentU );

    if ( ( mode == scatter_reflect ) || ( mode == scatter_reflect_transmit ) )
    {
      vec3 R = reflect( -lightDir, normal );
      float cosine = dot( R, viewDir );
      float shine = ( 0.0f < cosine ) ? ( ( 0.0f < roughness ) ? pow( cosine, 1.0f / roughness ) : ( 0.9999f <= cosine ) ? 1.0f : 0.0f ) : 0.0f;
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

vec4 mdl_df_simpleGlossyBSDFEnvironment( in float roughnessU, in float roughnessV, in vec3 tint, in vec3 tangentU, int mode, in vec3 normal )
{
  vec3 rgb = vec3( 0.0f, 0.0f, 0.0f );
  if ( sys_EnvironmentSamplerEnabled )
  {
    float roughness = calculateRoughness( normal, roughnessU, roughnessV, tangentU );
    lightDir = reflect( -viewDir, normal );
    lightDiffuse = evalEnvironmentMap( lightDir, roughness );
    lightSpecular = lightDiffuse;
    materialIOR = 1.0f;
    rgb = mdl_df_simpleGlossyBSDF( roughnessU, roughnessV, tint, tangentU, scatter_reflect, normal ).rgb;
  }
  return( vec4( rgb, 1.0f ) );
}
