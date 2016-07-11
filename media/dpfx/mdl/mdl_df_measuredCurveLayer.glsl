vec4 mdl_df_measuredCurveLayer( in vec3 curveValues[45], in float weight, in vec4 layer, in vec4 base, in vec3 normal )
{
  float z = acos( clamp( dot( normal, lightDir ), 0.0f, 1.0f ) ) * TWO_OVER_PI * 45.0f;    // multiply with the number of curveValues, to get an index
  return( vec4( mix( base.rgb, layer.rgb, weight * mix( curveValues[int(floor(z))], curveValues[int(ceil(z))], fract(z) ) ), base.a ) );
}

