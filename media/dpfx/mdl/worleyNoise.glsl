float squareLength( in vec3 d )
{
  return( dot( d, d ) );
}

float lengthChebyshev( in vec3 a )
{
  return( max( max( abs( a.x ), abs( a.y ) ), abs( a.z ) ) );
}

float lengthManhattan( in vec3 a )
{
  return( abs( a.x ) + abs( a.y ) + abs( a.z ) );
}

struct worleyReturn
{
  vec3  nearest_pos_0;
  vec3  nearest_pos_1;
  vec2  val;
};

#define FLT_MAX         3.402823466e+38F        /* max value */

worleyReturn worleyNoise( in vec3 pos, in float jitter, in int metric )
{
  worleyReturn ret;
  vec3 cell = floor( pos );
  vec2 f1f2 = vec2( FLT_MAX, FLT_MAX );
  jitter *= 1.0f / 255.0f;

  for ( int i=-1 ; i<=1 ; i++ )
  {
    float localCellX = cell.x + float(i);
    int rndx = rnd1( int( floor( localCellX ) ) );
    for ( int j=-1 ; j<=1 ; j++ )
    {
      float localCellY = cell.y + float(j);
      int rndxy = rndx ^ rnd2( int( floor( localCellY ) ) );
      for ( int k=-1 ; k<=1 ; k++)
      {
        float localCellZ = cell.z + float(k);
        int rndxyz = rndxy ^ rnd3( int( floor( localCellZ ) ) );
        vec3 localPos = vec3( localCellX, localCellY, localCellZ ) + vec3( float(rnd4( rndxyz )), float(rnd1( rndxyz )), float(rnd2( rndxyz )) ) * jitter;
        vec3 diff = localPos - pos;
        float dist = ( metric == 0 /*TextureNoise_WorleyMetric_Euclidean*/ ) ? squareLength( diff ) : ( ( metric == 1 /*TextureNoise_WorleyMetric_Manhattan*/ ) ? lengthManhattan( diff ) : lengthChebyshev( diff ) );
        if ( dist < f1f2.x )
        {
          f1f2.y = f1f2.x;
          ret.nearest_pos_1 = ret.nearest_pos_0;
          f1f2.x = dist;
          ret.nearest_pos_0 = localPos;
        }
        else if ( dist < f1f2.y )
        {
          f1f2.y = dist;
          ret.nearest_pos_1 = localPos;
        }
      }
    }
  }
  ret.val = ( metric == 0 /*TextureNoise_WorleyMetric_Euclidean*/ ) ? vec2( sqrt( f1f2.x ), sqrt( f1f2.y ) ) : f1f2;
  return ret;
}

float worleyNoise( in vec3 pos, in vec3 turbulenceWeight, in float stepThreshold, in int mode, in int metric, in float jitter )
{
  worleyReturn wr = worleyNoise( pos, jitter, metric );
  vec2 f1f2 = wr.val;
  float result;

  switch ( mode )
  {
    default:
    case 0:   //TextureNoise_WorleyMode_Simple0:
      result = f1f2.x;
      break;
    case 1:   //TextureNoise_WorleyMode_Simple1: // Squared simple worley
      result = square( f1f2.x );
      break;
    case 8:   //TextureNoise_WorleyMode_Simple2:
      result = f1f2.y;
      break;
    case 9:   //TextureNoise_WorleyMode_Simple3:
      result = square( f1f2.y );
      break;
    case 2:   //TextureNoise_WorleyMode_Cell:
      result = float( rnd1( int(floor( wr.nearest_pos_0.x )) ) ^ rnd2( int(floor( wr.nearest_pos_0.y )) ) ^ rnd3( int(floor( wr.nearest_pos_0.z )) ) ) * ( 1.0f / 255.0f );
      break;
    case 3:   //TextureNoise_WorleyMode_Step0: // "Hard" Step
      result = ( f1f2.y - f1f2.x < stepThreshold ) ? 0.0f : 1.0f;
      break;
    case 4:   //TextureNoise_WorleyMode_Step1: // "Smooth" Step
      result = f1f2.y - f1f2.x;
      break;
    case 5:   //TextureNoise_WorleyMode_Step2: // Normalized "Smooth" Step
      result = ( f1f2.y - f1f2.x ) / ( f1f2.y + f1f2.x );
      break;
    case 6:   //TextureNoise_WorleyMode_Mul:
      result = f1f2.x * f1f2.y;
      break;
    case 7:   //TextureNoise_WorleyMode_Add:
      result = f1f2.x + f1f2.y;
      break;
    case 10:  //TextureNoise_WorleyMode_Manhattan:
      result = lengthManhattan( wr.nearest_pos_1 - wr.nearest_pos_0 );
      break;
    case 11:  //TextureNoise_WorleyMode_Chebyshev:
      result = lengthChebyshev( wr.nearest_pos_1 - wr.nearest_pos_0 );
      break;
  }

  if ( turbulenceWeight != vec3( 0.0f, 0.0f, 0.0f ) )
  {
    result = sin( dot( pos, turbulenceWeight ) + result );
  }

  return( result );
}
