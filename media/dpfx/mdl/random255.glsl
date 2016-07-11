
// random [0..255] calculated by some simple Linear Congruential Generator (LCG)
// Verfied, that each function is just a permutation of {0,...,255}, that is, with 0<=x<=255, each value in [0,255] is returned once

int rnd1( in int x )
{
  return( ( ( 26665 * ( x & 0xFF ) + 15537 ) >> 0 ) & 0xFF );
}

int rnd2( in int x )
{
  return( ( ( 30726 * ( x & 0xFF ) + 19453 ) >> 1 ) & 0xFF );
}

int rnd3( in int x )
{
  return( ( ( 31300 * ( x & 0xFF ) + 24493 ) >> 2 ) & 0xFF );
}

int rnd4( in int x )
{
  return( ( ( 32007 * ( x & 0xFF ) + 642 ) >> 0 ) & 0xFF );
}

int rnd5( in int x )
{
  return( ( ( 31607 * ( x & 0xFF ) + 19070 ) >> 0 ) & 0xFF );
}

int rnd6( in int x )
{
  return( ( ( 30266 * ( x & 0xFF ) + 20594 ) >> 1 ) & 0xFF );
}

int rnd7( in int x )
{
  return( ( ( 27884 * ( x & 0xFF ) + 18519 ) >> 2 ) & 0xFF );
}

int rnd8( in int x )
{
  return( ( ( 12765 * ( x & 0xFF ) + 15702 ) >> 0 ) & 0xFF );
}

