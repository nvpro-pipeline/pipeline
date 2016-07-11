vec4 mdl_df_normalizedMix( in _df_bsdfComponent components[1] )
{
  return( ( components[0].weight < 1.0f ? components[0].weight : 1.0f ) * components[0].component );
}

vec4 mdl_df_normalizedMix( in _df_bsdfComponent components[2] )
{
  float sum = components[0].weight + components[1].weight;
  float invSum = ( sum <= 1.0f ) ? 1.0f : 1.0f / sum;
  return( invSum * ( components[0].weight * components[0].component + components[1].weight * components[1].component ) );
}

