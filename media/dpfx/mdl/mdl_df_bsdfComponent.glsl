_df_bsdfComponent mdl_df_bsdfComponent( in float weight, in vec4 component )
{
  _df_bsdfComponent bc;
  bc.weight = weight;
  bc.component = component;
  return( bc );
}

