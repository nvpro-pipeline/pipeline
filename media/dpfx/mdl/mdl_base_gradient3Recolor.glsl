float gradientInterpolate( in int interpolationMode, in float value )
{
  switch( interpolationMode )
  {
    default :   // fall through
    case gradient_interpolation_linear :
      return( value );
    case gradient_interpolation_off :
      return( 0.0f );
    case gradient_interpolation_ease_in :
      return( 1.0f - pow( 1.0f - value, 2.0f/3.0f ) );
    case gradient_interpolation_ease_out :
      return( pow( value, 2.0f/3.0f ) );
    case gradient_interpolation_ease_in_out :
      return( ( value <= 0.5f )
            ? 0.5f * pow( 2.0f * value, 2.0f/3.0f )
            : 0.5f + 0.5f * ( 1.0f - pow( 2.0f * ( 1.0f - value ), 2.0f/3.0f ) ) );
  }
}

_base_textureReturn mdl_base_gradient3Recolor( in float[3] gradientPositions, in vec3[3] gradientColors, in int[3] interpolationModes, in int monoSource, in float distortion, in float position )
{
  _base_textureReturn tr;
  float pos = position + distortion;
  if ( pos <= gradientPositions[0] )
  {
    tr.tint = gradientColors[0];
  }
  else if ( gradientPositions[2] <= pos )
  {
    tr.tint = gradientColors[2];
  }
  else
  {
    int index = ( position < gradientPositions[1] ) ? 0 : 1;
    float relPos = gradientInterpolate( interpolationModes[index], ( pos - gradientPositions[index] ) / ( gradientPositions[index+1] - gradientPositions[index] ) );
    tr.tint = mix( gradientColors[index], gradientColors[index+1], relPos );
  }
  tr.mono = 0.0f;
  return( tr );
}

