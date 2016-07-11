vec4 mdl_df_customCurveLayer( in float normal_reflectivity, in float grazing_reflectivity, in float exponent, float weight, vec4 layer, vec4 base )
{
  vec3 H = normalize( viewDir + lightDir );
  return( vec4( mix( base.rgb, layer.rgb, weight * ( normal_reflectivity + grazing_reflectivity * pow( abs( dot( viewDir, H ) ), exponent ) ) ), base.a ) );
}

