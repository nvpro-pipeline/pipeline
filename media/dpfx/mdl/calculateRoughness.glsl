float calculateRoughness( in vec3 N, in float roughnessU, in float roughnessV, in vec3 tangentU )
{
  float roughness = roughnessU;
  if ( roughnessU != roughnessV )
  {
    // determine major and minor radii a and b, and the vector along the major axis
    float a = roughnessU;
    float b = roughnessV;

    // we need the angle between the major axis and the projection of viewDir on the tangential plane
    // the major axis is the orthonormalization of tangentU with respect to N
    // the projection of viewDir is the orthonormalization of viewDir with respect to N
    // as both vectors would be calculated by orthonormalize, we can as well leave the second cross
    // product in those calculations away, as they don't change the angular relation.
    vec3 minorAxis = normalize( cross( tangentU, N ) );   // crossing this with N would give the major axis
                                                          // which is equivalent to orthonormalizing tangentU with respect to N
    if ( roughnessU < roughnessV )
    {
      a = roughnessV;
      b = roughnessU;
      minorAxis = cross( N, minorAxis );
    }

    vec3 po = normalize( cross( viewDir, N ) );
    float cosPhi = dot( po, minorAxis );

    // determine the polar coordinate of viewDir, take that radius as the roughness
    float excentricitySquare = 1.0f - square( b / a );
    roughness = b / sqrt( 1.0f - excentricitySquare * square( cosPhi ) );
  }
  return( roughness );
}

