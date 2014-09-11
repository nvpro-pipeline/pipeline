
// Evaluate a bezier triangle given by 10 vertices in bezierPoints at the sample point uvw
// calculate normal at that point, as well
vec3 evalBezierTriangle( in vec3 bezierPoints[10], in vec3 uvw, out vec3 normal )
{
  vec3 p10 = uvw.x * bezierPoints[0] + uvw.y * bezierPoints[1] + uvw.z * bezierPoints[4];
  vec3 p11 = uvw.x * bezierPoints[1] + uvw.y * bezierPoints[2] + uvw.z * bezierPoints[5];
  vec3 p12 = uvw.x * bezierPoints[2] + uvw.y * bezierPoints[3] + uvw.z * bezierPoints[6];
  vec3 p13 = uvw.x * bezierPoints[4] + uvw.y * bezierPoints[5] + uvw.z * bezierPoints[7];
  vec3 p14 = uvw.x * bezierPoints[5] + uvw.y * bezierPoints[6] + uvw.z * bezierPoints[8];
  vec3 p15 = uvw.x * bezierPoints[7] + uvw.y * bezierPoints[8] + uvw.z * bezierPoints[9];

  vec3 p20 = uvw.x * p10 + uvw.y * p11 + uvw.z * p13;
  vec3 p21 = uvw.x * p11 + uvw.y * p12 + uvw.z * p14;
  vec3 p22 = uvw.x * p13 + uvw.y * p14 + uvw.z * p15;

  vec3 p30 = uvw.x * p20 + uvw.y * p21 + uvw.z * p22;

  vec3 du = normalize( p21 - p20 );
  vec3 dv = normalize( p22 - p20 );
  normal = normalize( cross( du, dv ) );

  return( p30 );
}
