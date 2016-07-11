#ifndef SQRT_TWO
#define SQRT_TWO sqrt(2.0f)
#endif

// calculate the length of the hypotenuse of a right-angle triangle
float hypot( in float x, in float y )
{
  // naive implementation (sqrt( x * x + y * y )) might over- or underflow
  x = abs( x );
  y = abs( y );
  float t = min( x, y );
  x = max( x, y );
  y = t;
  return( x * sqrt( 1.0f + square( x / y ) ) );
}

float gradientGetPosition( in int gradientMode, in vec2 xy )
{
  vec2 tex = xy - floor( xy );

  switch( gradientMode )
  {
    case gradient_linear :
      return( tex.x );
    case gradient_squared :
      return( square( tex.x ) );
    case gradient_box :
      // Linear from center to edge
      return( 2.0f * max( abs( tex.x - 0.5f ), abs( tex.y - 0.5f ) ) );
    case gradient_diagonal :
      // Linear, diagonally, from center to upper-left and lower-right corners
      // Calculate distance from diagonal
      return( SQRT_TWO * abs( tex.x - tex.y ) );
    case gradient_90_degree :
      if ( tex.y < tex.x )
      {
        return( 1.0f - 0.5f * ( tex.y / tex.x ) );
      }
      else if ( tex.x < tex.y )
      {
        return( 0.5f * ( tex.x / tex.y ) );
      }
      else
      {
        return( 0.5f );
      }
    case gradient_symmetric_90_degree :
      if ( tex.y < tex.x )
      {
        return( tex.y / tex.x );
      }
      else if ( tex.x < tex.y )
      {
        return( tex.x / tex.y );
      }
      else
      {
        return( 1.0f );
      }
    case gradient_radial :
      // Distance from center
      return( 2.0f * hypot( tex.x - 0.5f, tex.y - 0.5f ) );
    case gradient_360_degree :
      // Consider two vectors (0,1) and tex, and calculate the angle between them.
      //  Center the coord around (0.5f,0.5f)
      vec2 dist = tex - 0.5f;
      float ret = acos( dist.y / hypot( dist.x, dist.y ) ) * 0.5f / PI;
      return( ( dist.x < 0.0f ) ? ret : 1.0f - ret );
    default :
      return( 0.0f );
  }
}

