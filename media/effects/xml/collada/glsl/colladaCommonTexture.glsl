
vec2 evaluateTexCoord( in int tc )
{
  switch( tc )
  {
    case 0 :
      return( varTexCoord0 );
    case 1 :
      return( varTexCoord1 );
    case 2 :
      return( varTexCoord2 );
    case 3 :
      return( varTexCoord3 );
    case 4 :
      return( varTexCoord4 );
    case 5 :
      return( varTexCoord5 );
    case 6 :
      return( varTexCoord6 );
    case 7 :
      return( varTexCoord7 );
    default :
      return( vec2( 0.0f, 0.0f ) );
  }
}

vec4 evaluateColor( in vec4 color, in sampler2D sampler, in int tc )
{
  if ( 0 <= tc )
  {
    return( texture2D( sampler, evaluateTexCoord( tc ) ) );
  }
  return( color );
}
