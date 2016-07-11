vec4 mdl_df_measuredCurveFactor( in vec3 curveValues[5], in vec4 base )
{
  float z = acos( clamp( dot( stateNormal, lightDir ), 0.0f, 1.0f ) ) * TWO_OVER_PI * 5.0f;      // multiply with the number of curveValues, to get an index
  return( vec4( base.rgb * mix( curveValues[int(floor(z))], curveValues[int(ceil(z))], fract(z) ), base.a ) );
}

vec4 mdl_df_measuredCurveFactor( in vec3 curveValues[45], in vec4 base )
{
  float z = acos( clamp( dot( stateNormal, lightDir ), 0.0f, 1.0f ) ) * TWO_OVER_PI * 45.0f;    // multiply with the number of curveValues, to get an index
  return( vec4( base.rgb * mix( curveValues[int(floor(z))], curveValues[int(ceil(z))], fract(z) ), base.a ) );
}

