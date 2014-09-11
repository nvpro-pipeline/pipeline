
// create a bezier triangle out of a triangle with positions (in positions) and normals (in normals)
// see for example http://alex.vlachos.com/graphics/CurvedPNTriangles.pdf
void createBezierFromPNTriangle( in vec3 positions[3], in vec3 normals[3], out vec3 bezierPoints[10] )
{
  bezierPoints[0] = positions[0];
  bezierPoints[3] = positions[1];
  bezierPoints[9] = positions[2];

  float w01 = dot( positions[1] - positions[0], normals[0] );
  float w10 = dot( positions[0] - positions[1], normals[1] );
  float w02 = dot( positions[2] - positions[0], normals[0] );
  float w20 = dot( positions[0] - positions[2], normals[2] );
  float w12 = dot( positions[2] - positions[1], normals[1] );
  float w21 = dot( positions[1] - positions[2], normals[2] );

  bezierPoints[1] = ( 2.0f * positions[0] + positions[1] - w01 * normals[0] ) / 3.0f;
  bezierPoints[2] = ( 2.0f * positions[1] + positions[0] - w10 * normals[1] ) / 3.0f;
  bezierPoints[4] = ( 2.0f * positions[0] + positions[2] - w02 * normals[0] ) / 3.0f;
  bezierPoints[6] = ( 2.0f * positions[1] + positions[2] - w12 * normals[1] ) / 3.0f;
  bezierPoints[7] = ( 2.0f * positions[2] + positions[0] - w20 * normals[2] ) / 3.0f;
  bezierPoints[8] = ( 2.0f * positions[2] + positions[1] - w21 * normals[2] ) / 3.0f;

  vec3 E = ( bezierPoints[1] + bezierPoints[2] + bezierPoints[4] + bezierPoints[6] + bezierPoints[7] + bezierPoints[8] ) / 6.0f;
  vec3 V = ( positions[0] + positions[1] + positions[2] ) / 3.0f;
  bezierPoints[5] = E + 0.5f * ( E - V );
}