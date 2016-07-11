int permuteFlow( in int x )
{
  //x %= 289; // correct, but doesn't matter in practice
  return( ( x * x * 34 + x ) % 289 );
}

// gradient mapping + extra rotation
vec2 gradFlow( in int p, in float rot )
{
  float u = p * ( 1.0f / 41.0f ) + rot;   // [0..7] + rot: (a bit) magic

  // map from line to diamond -> shift maps to rotation
  u = u - floor( u ) * 4.0f - 2.0f;
  return( vec2( abs( u ) - 1.0f, abs( abs( u + 1.0f ) - 2.0f ) - 1.0f ) );
}

float flowNoise( in vec2 p, in float rot )
{
#define SQRT3 1.7320508075688772935274463415059

  // transform to skewed simplex grid, round down to simplex origin
  vec2 pi = floor( p + ( p.x + p.y ) * ( SQRT3 / 2.0f - 0.5f ) );

  // transform simplex origin back to x,y, find x,y offsets from simplex origin to corner 1
  vec2 v0 = p - pi + ( pi.x + pi.y ) * ( 0.5f - SQRT3 / 6.0f );

  // offsets for other 2 corners
  vec2 v1 = v0 + ( ( v0.x < v0.y ) ? vec2( 0.5f - SQRT3 / 6.0f, -0.5f - SQRT3 / 6.0f ) : vec2( -0.5f - SQRT3 / 6.0, 0.5f - SQRT3 / 6.0f ) );
  vec2 v2 = v0 - 1.0f / SQRT3;

  // calc circularly symmetric part of each noise wiggle
  vec3 t = max( vec3( 0.5f - dot( v0, v0 ), 0.5f - dot( v1, v1 ), 0.5f - dot( v2, v2 ) ), vec3( 0.0f, 0.0f, 0.0f ) );
  vec3 t2 = t * t;
  vec3 t4 = t2 * t2;

  ivec2 pii = ivec2( pi );
  int tmpp0 = permuteFlow( pii.x ) + pii.y;
  int tmpp2 = permuteFlow( pii.x + 1 ) + pii.y;
  vec2 g0 = gradFlow( permuteFlow( tmpp0 ), rot );
  float g0v0 = dot( g0, v0 );
  vec2 g1 = gradFlow( permuteFlow( ( v0.x < v0.y ) ? tmpp0 + 1 : tmpp2 ), rot );
  float g1v1 = dot( g1, v1 );
  vec2 g2 = gradFlow( permuteFlow( tmpp2 + 1 ), rot );

  // compute noise contributions from each corner
  vec3 gv = vec3( g0v0, g1v1, dot( g2, v2 ) );    // ramp

  // add contributions from all 3 corners
  return( 40.0f * dot( t4, gv ) );   // circular kernel * ramp

#undef SQRT3
}

float flowNoise( in vec2 pos, in float time, in int iterations, in bool absoluteNoise, in float weightFactor, in float positionFactor, in float uProgressiveScale, in float vProgressiveOffset )
{
  float sum = 0.0f;
  float weight = 1.0f;
  vec3 p = vec3( pos, time );

  // iteration_offset is used to avoid creating concentric artifacts around the origin from having all
  // wave sizes originate from the same origin. The offset of 7.0 is copied from the lume ocean shader.
  float iterationOffset = 0.0f;
  float invIterations = 1.0f / float( iterations - 1 );
  for( int i=0 ; i<iterations ; ++i, iterationOffset += 7.0f )
  {
    float lerpPosition = ( 1 < iterations ) ? float(i) * invIterations : 1.0f;
    float n = flowNoise( vec2( p.x * ( 1.0f - lerpPosition + lerpPosition * uProgressiveScale ), p.y + lerpPosition * vProgressiveOffset + iterationOffset ), p.z );
    sum += weight * ( absoluteNoise ? abs( n ) : n );
    p *= positionFactor;
    weight *= weightFactor;
  }

  // Scale [-1,1] to [0,1]
  return( 0.5f * sum + 0.5f );
}
