vec4 mdl_df_directionalFactor( in vec3 normalTint, in vec3 grazingTint, in float exponent, in vec4 base )
{
  return( vec4( mix( normalTint, grazingTint, pow( 1.0f - max( 0.0f, dot( stateNormal, viewDir ) ), exponent ) ), 1.0f ) * base );
}

