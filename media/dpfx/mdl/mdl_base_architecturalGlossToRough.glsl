float mdl_base_architecturalGlossToRough( in float glossiness )
{
  return( ( 1.0 <= glossiness ) ? 0.0 : sqrt( 2.0 * exp2( -4.0 - 14.0 * glossiness ) ) );
}

