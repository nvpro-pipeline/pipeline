vec3 mdl_math_blackbody( in float temperature )
{
  vec3 color;
  temperature /= 100.0f;
  
  if ( temperature <= 66.0f )
  {
    color.r = 1.0f;
    color.g = clamp( ( 99.4708025861f * log(temperature) - 161.1195681661f ) / 255.0f, 0.0f, 1.0f );
    if ( temperature <= 19.0f )
    {
      color.b = 0.0f;
    }
    else
    {
      color.b = clamp( ( 138.5177312231f * log(temperature) - 305.0447927307f ) / 255.0f, 0.0f, 1.0f );
    }
  }
  else
  {
    color.r = clamp( 329.698727446f * pow(temperature - 60.0f, -0.1332047592f) / 255.0f, 0.0f, 1.0f );
    color.g = clamp( 288.1221695283f * pow(temperature, -0.0755148492f) / 255.0f, 0.0f, 1.0f );
    color.b = 1.0f;
  }
  return( color );
}
