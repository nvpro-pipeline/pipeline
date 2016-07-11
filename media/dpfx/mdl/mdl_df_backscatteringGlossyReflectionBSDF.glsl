vec4 mdl_df_backscatteringGlossyReflectionBSDF( in float roughnessU, in float roughnessV, in vec3 tint, in vec3 tangentU, in vec3 normal )
{
  const float edginess = 10.0f;

  vec3 rgb = vec3( 0.0f, 0.0f, 0.0f );
  float cosTheta = dot( normal, lightDir );
  if ( 0.0f < cosTheta )
  {
    float roughness = calculateRoughness( normal, roughnessU, roughnessV, tangentU );

    // retro-reflection
    vec3 R = reflect( -lightDir, normal );
    float cosine = dot( R, viewDir ); // cos(alpha): angle between reflection and eye-direction
    float shine = ( 0.0f < cosine ) ? ( ( 0.0f < roughness ) ? pow( cosine, 1.0f / roughness ) : ( 0.9999f <= cosine ) ? 1.0f : 0.0f ) : 0.0f;

    // horizon scattering
    cosine = dot( normal, viewDir );       // cos(theta+alpha): angle between normal and eye-direction
    if ( 0.0f < cosine )
    {
      shine += pow( sqrt( 1.0f - cosine * cosine ), edginess ) * cosTheta;
    }

    rgb = shine * lightSpecular * tint;
  }

  return( vec4( rgb, 1.0f ) );
}

vec4 mdl_df_backscatteringGlossyReflectionBSDFEnvironment( in float roughnessU, in float roughnessV, in vec3 tint, in vec3 tangentU, in vec3 normal )
{
  vec3 rgb = vec3( 0.0f, 0.0f, 0.0f );
  if ( sys_EnvironmentSamplerEnabled )
  {
    float roughness = calculateRoughness( normal, roughnessU, roughnessV, tangentU );
    lightDir = reflect( -viewDir, normal );
    lightDiffuse = evalEnvironmentMap( lightDir, roughness );
    lightSpecular = lightDiffuse;
    rgb = mdl_df_backscatteringGlossyReflectionBSDF( roughnessU, roughnessV, tint, tangentU, normal ).rgb;
  }
  return( vec4( rgb, 1.0f ) );
}
