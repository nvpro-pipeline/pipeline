
vec4 miNoise( in vec3 xyz )
{
  const float div = 1.0f / 256f;

  ivec3 itmp = ivec3( xyz );
  ivec3 ixyz[3] = ivec3[]( itmp & 0xFF, itmp+1 & 0xFF, itmp+2 & 0xFF );
  vec3 fxyz = xyz - floor( xyz );

  vec3 dux, ux;
  dux.z = fxyz.x * div;
  dux.y = div - 2.0f * dux.z;
  dux.x = dux.z - div;
  ux.z = 0.5f * fxyz.x * dux.z;
  ux.y = dux.z + 0.5f * ( div - fxyz.x * dux.z );
  ux.x = ux.z - dux.z + 0.5f * div;

  vec3 duy, uy;
  duy.z = fxyz.y;
  duy.y = 1.0f - 2.0f * duy.z;
  duy.x = duy.z - 1.0f;
  uy.z = 0.5f * square( duy.z );
  uy.y = duy.z + 0.5f - square( duy.z );
  uy.x = uy.z - duy.z + 0.5f;

  float duz[3] = float[]( fxyz.z - 1.0f, 1.0f - 2 * fxyz.z, fxyz.z );
  float uz[3]  = float[]( 0.5f * square( fxyz.z ) - fxyz.z + 0.5f, fxyz.z + 0.5f - square( fxyz.z ), 0.5f * square( fxyz.z ) );

  int xx = rnd1( ixyz[0].x );
  int yx = rnd1( ixyz[0].y );
  int zx = rnd1( ixyz[0].z );
  int xy = rnd2( ixyz[1].x );
  int yy = rnd2( ixyz[1].y );
  int zy = rnd2( ixyz[1].z );

  vec4 ret = vec4( 0.0f, 0.0f, 0.0f, 0.0f );

  for ( int i=0 ; i<3 ; i++ )
  {
    int iz = rnd3( ixyz[2][i] );

    mat3x3 nf = mat3x3( vec3( float( rnd5( xx ^ xy ^ iz ^ rnd4( ixyz[0].x ^ ixyz[1].x ^ ixyz[2][i] ) ) )
                            , float( rnd5( yx ^ xy ^ iz ^ rnd4( ixyz[0].y ^ ixyz[1].x ^ ixyz[2][i] ) ) )
                            , float( rnd5( zx ^ xy ^ iz ^ rnd4( ixyz[0].z ^ ixyz[1].x ^ ixyz[2][i] ) ) ) )
                      , vec3( float( rnd5( xx ^ yy ^ iz ^ rnd4( ixyz[0].x ^ ixyz[1].y ^ ixyz[2][i] ) ) )
                            , float( rnd5( yx ^ yy ^ iz ^ rnd4( ixyz[0].y ^ ixyz[1].y ^ ixyz[2][i] ) ) )
                            , float( rnd5( zx ^ yy ^ iz ^ rnd4( ixyz[0].z ^ ixyz[1].y ^ ixyz[2][i] ) ) ) )
                      , vec3( float( rnd5( xx ^ zy ^ iz ^ rnd4( ixyz[0].x ^ ixyz[1].z ^ ixyz[2][i] ) ) )
                            , float( rnd5( yx ^ zy ^ iz ^ rnd4( ixyz[0].y ^ ixyz[1].z ^ ixyz[2][i] ) ) )
                            , float( rnd5( zx ^ zy ^ iz ^ rnd4( ixyz[0].z ^ ixyz[1].z ^ ixyz[2][i] ) ) ) ) );

    float fxdz = dot( uy, dux * nf );
    vec3 dy = ux * nf;
    float dz = dot( uy, dy );

    ret.x += uz[i] * fxdz;
    ret.y += uz[i] * dot( duy, dy );
    ret.z += duz[i] * dz;
    ret.w += uz[i] * dz;
  }
  return( ret );
}

vec4 mi_noise( in ivec3 xyz )
{
  vec3 grad;
  int ix0 = ( xyz.x - 1 ) & 255;
  int ix1 =   xyz.x       & 255;
  int ix2 = ( xyz.x + 1 ) & 255;
  int iy0 = ( xyz.y - 1 ) & 255;
  int iy1 =   xyz.y       & 255;
  int iy2 = ( xyz.y + 1 ) & 255;
  int iz0 = ( xyz.z - 1 ) & 255;
  int iz1 =   xyz.z       & 255;
  int iz2 = ( xyz.z + 1 ) & 255;

  // compute b-spline blending as functions of input point coords du = d/dx.
  // Everything is a tensor product so we have only one "derivative" per dimension
  int noise_factor001 = rnd5( rnd1( ix0 ) ^ rnd2( iy0 ) ^ rnd3( iz0 ) ) ^ rnd4( ix0 ^ iy0 ^ iz0 );
  int noise_factor002 = rnd5( rnd1( ix1 ) ^ rnd2( iy0 ) ^ rnd3( iz0 ) ) ^ rnd4( ix1 ^ iy0 ^ iz0 );
  int noise_factor003 = rnd5( rnd1( ix2 ) ^ rnd2( iy0 ) ^ rnd3( iz0 ) ) ^ rnd4( ix2 ^ iy0 ^ iz0 );

  int noise_factor011 = rnd5( rnd1( ix0 ) ^ rnd2( iy1 ) ^ rnd3( iz0 ) ) ^ rnd4( ix0 ^ iy1 ^ iz0 );
  int noise_factor012 = rnd5( rnd1( ix1 ) ^ rnd2( iy1 ) ^ rnd3( iz0 ) ) ^ rnd4( ix1 ^ iy1 ^ iz0 );
  int noise_factor013 = rnd5( rnd1( ix2 ) ^ rnd2( iy1 ) ^ rnd3( iz0 ) ) ^ rnd4( ix2 ^ iy1 ^ iz0 );

  int noise_factor021 = rnd5( rnd1( ix0 ) ^ rnd2(iy2) ^ rnd3( iz0 ) ) ^ rnd4( ix0 ^ iy2 ^ iz0 );
  int noise_factor022 = rnd5( rnd1( ix1 ) ^ rnd2(iy2) ^ rnd3( iz0 ) ) ^ rnd4( ix1 ^ iy2 ^ iz0 );
  int noise_factor023 = rnd5( rnd1( ix2 ) ^ rnd2(iy2) ^ rnd3( iz0 ) ) ^ rnd4( ix2 ^ iy2 ^ iz0 );

  int noise_factor101 = rnd5( rnd1( ix0 ) ^ rnd2(iy0) ^ rnd3(iz1) ) ^ rnd4( ix0 ^ iy0 ^ iz1 );
  int noise_factor102 = rnd5( rnd1( ix1 ) ^ rnd2(iy0) ^ rnd3(iz1) ) ^ rnd4( ix1 ^ iy0 ^ iz1 );
  int noise_factor103 = rnd5( rnd1( ix2 ) ^ rnd2(iy0) ^ rnd3(iz1) ) ^ rnd4( ix2 ^ iy0 ^ iz1 );

  int noise_factor111 = rnd5( rnd1( ix0 ) ^ rnd2( iy1 ) ^ rnd3(iz1) ) ^ rnd4( ix0 ^ iy1 ^ iz1 );
  int noise_factor112 = rnd5( rnd1( ix1 ) ^ rnd2( iy1 ) ^ rnd3(iz1) ) ^ rnd4( ix1 ^ iy1 ^ iz1 );
  int noise_factor113 = rnd5( rnd1( ix2 ) ^ rnd2( iy1 ) ^ rnd3(iz1) ) ^ rnd4( ix2 ^ iy1 ^ iz1 );

  int noise_factor121 = rnd5( rnd1( ix0 ) ^ rnd2(iy2) ^ rnd3(iz1) ) ^ rnd4( ix0 ^ iy2 ^ iz1 );
  int noise_factor122 = rnd5( rnd1( ix1 ) ^ rnd2(iy2) ^ rnd3(iz1) ) ^ rnd4( ix1 ^ iy2 ^ iz1 );
  int noise_factor123 = rnd5( rnd1( ix2 ) ^ rnd2(iy2) ^ rnd3(iz1) ) ^ rnd4( ix2 ^ iy2 ^ iz1 );

  int noise_factor201 = rnd5( rnd1( ix0 ) ^ rnd2(iy0) ^ rnd3(iz2) ) ^ rnd4( ix0 ^ iy0 ^ iz2 );
  int noise_factor202 = rnd5( rnd1( ix1 ) ^ rnd2(iy0) ^ rnd3(iz2) ) ^ rnd4( ix1 ^ iy0 ^ iz2 );
  int noise_factor203 = rnd5( rnd1( ix2 ) ^ rnd2(iy0) ^ rnd3(iz2) ) ^ rnd4( ix2 ^ iy0 ^ iz2 );

  int noise_factor211 = rnd5( rnd1( ix0 ) ^ rnd2( iy1 ) ^ rnd3(iz2) ) ^ rnd4( ix0 ^ iy1 ^ iz2 );
  int noise_factor212 = rnd5( rnd1( ix1 ) ^ rnd2( iy1 ) ^ rnd3(iz2) ) ^ rnd4( ix1 ^ iy1 ^ iz2 );
  int noise_factor213 = rnd5( rnd1( ix2 ) ^ rnd2( iy1 ) ^ rnd3(iz2) ) ^ rnd4( ix2 ^ iy1 ^ iz2 );

  int noise_factor221 = rnd5( rnd1( ix0 ) ^ rnd2(iy2) ^ rnd3(iz2) ) ^ rnd4( ix0 ^ iy2 ^ iz2 );
  int noise_factor222 = rnd5( rnd1( ix1 ) ^ rnd2(iy2) ^ rnd3(iz2) ) ^ rnd4( ix1 ^ iy2 ^ iz2 );
  int noise_factor223 = rnd5( rnd1( ix2 ) ^ rnd2(iy2) ^ rnd3(iz2) ) ^ rnd4( ix2 ^ iy2 ^ iz2 );

  grad.x = 0.0078125f / 256.0f * ( noise_factor003 - noise_factor001 + noise_factor023 - noise_factor021 + noise_factor203 - noise_factor201 + noise_factor223 - noise_factor221 ) +
           0.046875f / 256.0f * ( noise_factor013 - noise_factor011 + noise_factor103 - noise_factor101 + noise_factor123 - noise_factor121 + noise_factor213 - noise_factor211 ) +
           0.28125f / 256.0f * ( noise_factor113 - noise_factor111 ); // x-partial

  int dyz00 = noise_factor001 + 6 * noise_factor002 + noise_factor003; // f(x)
  int dyz02 = noise_factor021 + 6 * noise_factor022 + noise_factor023;

  int dyz10 = noise_factor101 + 6 * noise_factor102 + noise_factor103;
  int dyz12 = noise_factor121 + 6 * noise_factor122 + noise_factor123;

  int dyz20 = noise_factor201 + 6 * noise_factor202 + noise_factor203;
  int dyz22 = noise_factor221 + 6 * noise_factor222 + noise_factor223;

  grad.y = 0.0078125f / 256.0f * ( dyz02 - dyz00 + dyz22 - dyz20 ) + 0.046875f / 256.0f * ( dyz12 - dyz10 ); // y-partial

  float dz0 = 0.001953125f / 256.0f * ( dyz00 + dyz02 ) + 0.01171875f / 256.0f * ( noise_factor011 + noise_factor013 ) + 0.0703125f / 256.0f * ( noise_factor012 ); // f(x,y)
  float dz2 = 0.001953125f / 256.0f * ( dyz20 + dyz22 ) + 0.01171875f / 256.0f * ( noise_factor211 + noise_factor213 ) + 0.0703125f / 256.0f * ( noise_factor212 );

  grad.z = 4.0f * ( dz2 - dz0 ); // z-partial

  float val = dz0 + dz2 + 0.01171875f / 256.0f * ( dyz10 + dyz12 ) + 0.0703125f / 256.0f * ( noise_factor111 + noise_factor113 ) + 0.421875f / 256.0f * ( noise_factor112 ); // f(x,y,z)
  return( vec4( grad, val ) );
}
