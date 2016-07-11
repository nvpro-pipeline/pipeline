float monoChannel( in vec4 t, in int monoSource )
{
  switch( monoSource )
  {
    case mono_alpha :
      return( t.w );
    case mono_average :
      return( ( t.x + t.y + t.z ) / 3.0f );
    case mono_luminance :
      return( mdl_math_luminance( t.xyz ) );
    case mono_maximum :
      return( max( t.x, max( t.y, t.z ) ) );
    default :
      return( 1.0f );
  }
}

