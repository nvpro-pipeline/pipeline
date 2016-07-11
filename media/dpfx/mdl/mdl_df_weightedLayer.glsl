vec4 mdl_df_weightedLayer( in float weight, in vec4 layer, in vec4 base )
{
  return( mix( base, layer, weight ) );
}

