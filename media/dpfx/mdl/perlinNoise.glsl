float fade( in float x )
{
  return( x * x * ( 3.0f - 2.0f * x ) );
}

vec2 fade( in vec2 x )
{
  return( x * x * ( 3.0f - 2.0f * x ) );
}

vec3 fade( in vec3 x )
{
  return( x * x * ( 3.0f - 2.0f * x ) );
}

vec4 fade( in vec4 x )
{
  return( x * x * ( 3.0f - 2.0f * x ) );
}

float influence( in int hash, in float x )
{
  return( x * ( rnd5( hash ) % 3 - 1.0f ) );
}

float influence( in int hash, in float x, in float y )
{
  return( dot( vec2( x, y ), vec2( rnd5( hash ) % 3 - 1.0f, rnd6( hash ) % 3 - 1.0f ) ) );
}

float influence( in int hash, in float x, in float y, in float z )
{
  return( dot( vec3( x, y, z ), vec3( rnd5( hash ) % 3 - 1.0f, rnd6( hash ) % 3 - 1.0f, rnd7( hash ) % 3 - 1.0f ) ) );
}

float influence( in int hash, in float x, in float y, in float z, in float w )
{
  return( dot( vec4( x, y, z, w ), vec4( rnd5( hash ) % 3 - 1.0f, rnd6( hash ) % 3 - 1.0f, rnd7( hash ) % 3 - 1.0f, rnd8( hash ) % 3 - 1.0f ) ) );
}

float perlinNoise( in float pos )
{
  float floorPos = floor( pos );
  int intPos = int( floorPos );
  float fracPos = pos - floorPos;

  return( mix( influence( rnd1( intPos ), fracPos ), influence( rnd1( intPos+1 ), fracPos - 1.0f ), fade( fracPos ) ) );
}

float perlinNoise( in vec2 pos )
{
  vec2 floorPos = floor( pos );
  vec2 fracPos = pos - floorPos;
  vec2 fadedPos = fade( fracPos );

  int ax = rnd1( int(floorPos.x)     );
  int bx = rnd1( int(floorPos.x) + 1 );
  int ay = rnd2( int(floorPos.y)     );
  int by = rnd2( int(floorPos.y) + 1 );

  return( mix( mix( influence( ax^ay, fracPos.x,        fracPos.y        )
                  , influence( bx^ay, fracPos.x - 1.0f, fracPos.y        ), fadedPos.x )
             , mix( influence( ax^by, fracPos.x,        fracPos.y - 1.0f )
                  , influence( bx^by, fracPos.x - 1.0f, fracPos.y - 1.0f ), fadedPos.x ), fadedPos.y ) );
}

float perlinNoise( in vec3 pos )
{
  vec3 floorPos = floor( pos );
  vec3 fracPos = pos - floorPos;
  vec3 fadedPos = fade( fracPos );

  int ax = rnd1( int(floorPos.x)     );
  int bx = rnd1( int(floorPos.x) + 1 );
  int ay = rnd2( int(floorPos.y)     );
  int by = rnd2( int(floorPos.y) + 1 );
  int az = rnd3( int(floorPos.z)     );
  int bz = rnd3( int(floorPos.z) + 1 );

  int axay = ax ^ ay;
  int bxay = bx ^ ay;
  int axby = ax ^ by;
  int bxby = bx ^ by;

  return( mix( mix( mix( influence( axay^az, fracPos.x,        fracPos.y,        fracPos.z        )
                       , influence( bxay^az, fracPos.x - 1.0f, fracPos.y,        fracPos.z        ), fadedPos.x )
                  , mix( influence( axby^az, fracPos.x,        fracPos.y - 1.0f, fracPos.z        )
                       , influence( bxby^az, fracPos.x - 1.0f, fracPos.y - 1.0f, fracPos.z        ), fadedPos.x ), fadedPos.y )
             , mix( mix( influence( axay^bz, fracPos.x,        fracPos.y       , fracPos.z - 1.0f )
                       , influence( bxay^bz, fracPos.x - 1.0f, fracPos.y       , fracPos.z - 1.0f ), fadedPos.x )
                  , mix( influence( axby^bz, fracPos.x,        fracPos.y - 1.0f, fracPos.z - 1.0f )
                       , influence( bxby^bz, fracPos.x - 1.0f, fracPos.y - 1.0f, fracPos.z - 1.0f ), fadedPos.x ), fadedPos.y ), fadedPos.z ) );
}

float perlinNoise( in vec4 pos )
{
  vec4 floorPos = floor( pos );
  vec4 fracPos = pos - floorPos;
  vec4 fadedPos = fade( fracPos );

  int ax = rnd1( int(floorPos.x)     );
  int bx = rnd1( int(floorPos.x) + 1 );
  int ay = rnd2( int(floorPos.y)     );
  int by = rnd2( int(floorPos.y) + 1 );
  int az = rnd3( int(floorPos.z)     );
  int bz = rnd3( int(floorPos.z) + 1 );
  int aw = rnd4( int(floorPos.w)     );

  int axay = ax ^ ay;
  int bxay = bx ^ ay;
  int axby = ax ^ by;
  int bxby = bx ^ by;

  float result[2];
  for ( int i=0 ; i<2 ; i++ )
  {
    int azaw = az ^ aw;
    int bzaw = bz ^ aw;

    result[i] = mix( mix( mix( influence( axay^azaw, fracPos.x,        fracPos.y,        fracPos.z,        fracPos.w )
                             , influence( bxay^azaw, fracPos.x - 1.0f, fracPos.y,        fracPos.z,        fracPos.w ), fadedPos.x )
                        , mix( influence( axby^azaw, fracPos.x,        fracPos.y - 1.0f, fracPos.z,        fracPos.w )
                             , influence( bxby^azaw, fracPos.x - 1.0f, fracPos.y - 1.0f, fracPos.z,        fracPos.w ), fadedPos.x ), fadedPos.y )
                   , mix( mix( influence( axay^bzaw, fracPos.x,        fracPos.y,        fracPos.z - 1.0f, fracPos.w )
                             , influence( bxay^bzaw, fracPos.x - 1.0f, fracPos.y,        fracPos.z - 1.0f, fracPos.w ), fadedPos.x )
                        , mix( influence( axby^bzaw, fracPos.x,        fracPos.y - 1.0f, fracPos.z - 1.0f, fracPos.w )
                             , influence( bxby^bzaw, fracPos.x - 1.0f, fracPos.y - 1.0f, fracPos.z - 1.0f, fracPos.w ), fadedPos.x ), fadedPos.y ), fadedPos.z );
    aw = rnd4( int(floorPos.w) + 1 );
    fracPos.w -= 1.0f;
  }
  return( mix( result[0], result[1], fadedPos.w ) );
}