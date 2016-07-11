vec4 mdl_df_tint( in vec3 tint, in vec4 base )
{
  return( vec4( tint * base.rgb, base.a ) );
}

