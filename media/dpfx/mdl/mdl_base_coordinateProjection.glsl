mat4x4 computeCubicTransform( in mat4x4 projectionTransform )
{
  vec3 transformedNormal = ( vec4( varNormal, 0.0f ) * projectionTransform ).xyz;
  mat4x4 result = transpose( projectionTransform );

  // Find out on which cube face is the intersection point
  if (  abs( transformedNormal.y ) <= abs( transformedNormal.x )
     && abs( transformedNormal.z ) <= abs( transformedNormal.x ) )
  {
    vec4 tmp = result[2];
    result[2] = result[0];
    result[0] = result[1];
    result[1] = tmp;
    if ( transformedNormal.x <= 0.0f )
    {
      result[0] = - result[0];
    }
  }
  else if ( abs( transformedNormal.x ) <= abs( transformedNormal.y )
        &&  abs( transformedNormal.z ) <= abs( transformedNormal.y ) )
  {
    vec4 tmp = result[1];
    result[1] = result[2];
    result[2] = tmp;
    if ( 0.0f < transformedNormal.y )
    {
      result[0] = - result[0];
    }
  }
  else if ( abs( transformedNormal.x ) <= abs( transformedNormal.z )
        &&  abs( transformedNormal.y ) <= abs( transformedNormal.z ) )
  {
    if ( transformedNormal.z <= 0 )
    {
      result[0] = - result[0];
    }
  }
  return( result );
}

mat4x4 computeSphericTransform( in mat4x4 projectionTransform )
{
  vec3 transformedPosition = ( vec4( varWorldPos, 1.0f ) * projectionTransform ).xyz;
  float dist = length( transformedPosition );
  vec2 uvCoord = dist * vec2( atan( transformedPosition.y, transformedPosition.x )
                            , atan( transformedPosition.z, length( transformedPosition.xy ) ) );

  return( mat4x4( 1.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 1.0f, 0.0f,
                  -transformedPosition.x + uvCoord.x, -transformedPosition.y + uvCoord.y, -transformedPosition.z, 1.0f ) );
}

mat4x4 computeCylindricTransform( in mat4x4 projectionTransform )
{
  vec3 absTransformedNormal = abs( ( vec4( varNormal, 0.0f ) * projectionTransform ).xyz );
  if ( ( absTransformedNormal.x <= absTransformedNormal.z ) && ( absTransformedNormal.y <= absTransformedNormal.z ) )
  {
    return( transpose( projectionTransform ) );
  }

  vec3 transformedPosition = ( vec4( varWorldPos, 1.0f ) * projectionTransform ).xyz;
  vec2 uvCoord = vec2( 0.0f, transformedPosition.z );
  if ( ( 0.0f < abs( transformedPosition.x ) ) || ( 0.0f < abs( transformedPosition.y ) ) )
  {
    uvCoord.x = length( transformedPosition.xy ) * atan( transformedPosition.y, transformedPosition.x );
  }

  // Note: GLSL matrices construction are column major, that is the translation is in the fourth column here!
  return( mat4x4( 1.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 1.0f, 0.0f,
                  -transformedPosition.x + uvCoord.x, -transformedPosition.y + uvCoord.y, -transformedPosition.z, 1.0f ) );
}

mat4x4 computeInfiniteCylindricTransform( in mat4x4 projectionTransform )
{
  vec3 transformedPosition = ( vec4( varWorldPos, 1.0f ) * projectionTransform ).xyz;
  vec2 uvCoord = vec2( 0.0f, 0.0f );

  if ( 0.0f < abs( transformedPosition.x ) || 0.0f < abs( transformedPosition.y ) )
  {
    uvCoord.x = length( transformedPosition.xy ) * atan( transformedPosition.y, transformedPosition.x );
  }
  uvCoord.y = transformedPosition.z;

  return( mat4x4( 1.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 1.0f, 0.0f,
                  -transformedPosition.x + uvCoord.x, -transformedPosition.y + uvCoord.y, -transformedPosition.z, 1.0f ) );
}


_base_textureCoordinateInfo mdl_base_coordinateProjection( in int coordinateSystem, in int texturespace, in int projectionType, in mat4 projectionTransform )
{
  mat4 finalTransform;
  switch( projectionType )
  {
    // calculate a final transform here
    case projection_cubic :
      finalTransform = computeCubicTransform( projectionTransform );
      break;
    case projection_spherical :
      finalTransform = computeSphericTransform( projectionTransform );
      break;
    case projection_cylindrical :
      finalTransform = computeCylindricTransform( projectionTransform );
      break;
    case projection_infinite_cylindrical :
      finalTransform = computeInfiniteCylindricTransform( projectionTransform );
      break;
    case projection_planar :
    default :
      finalTransform = projectionTransform;
      break;
  }

  _base_textureCoordinateInfo tci;
  tci.position = ( finalTransform * vec4( varWorldPos, 1.0f ) ).xyz;
  tci.tangentU = vec3( finalTransform[0][0], finalTransform[1][0], finalTransform[2][0] );
  tci.tangentV = vec3( finalTransform[0][1], finalTransform[1][1], finalTransform[2][1] );
  return( tci );
}

