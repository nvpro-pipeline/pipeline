vec4 mdl_df_diffuseTransmissionBSDF( in vec3 materialDiffuse, in vec3 normal )
{
  // use -normal, to get color from lights behind
  float cosThetaI = max( 0.0f, dot( -normal, lightDir ) );
  return( vec4( cosThetaI * lightDiffuse * materialDiffuse, 1.0f ) );
}

vec4 mdl_df_diffuseTransmissionBSDFEnvironment( in vec3 tint, in vec3 normal )
{
  return( vec4( 0.0f, 0.0f, 0.0f, 1.0f ) );
}
