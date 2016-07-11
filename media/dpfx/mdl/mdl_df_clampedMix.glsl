vec4 mdl_df_clampedMix( in _df_bsdfComponent components[2] )
{
  vec4 result = vec4( 0.0f, 0.0f, 0.0f, 0.0f );
  float sum = 0.0f;
  for ( int i=0 ; i<2 && sum < 1.0f ; i++ )
  {
    if ( sum + components[i].weight < 1.0f )
    {
      result += components[i].weight * components[i].component;
    }
    else if ( sum < 1.0f )
    {
      result += ( 1.0f - sum ) * components[i].component;
    }
    sum += components[i].weight;
  }
  return( result );
}

