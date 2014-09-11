
layout( triangles ) in;

void main()
{
  // construct a bezier triangle out of the given triangle, using the normals to calculate the missing vertices
  // Note: This calculates the bezierTriangle for each generated vertex.
  //       Doing that calculation in the TCS (which means: do it per (original) vertex) and
  //       - pass the 10 bezier points via patch out/in is slightly slower!
  //       - pass the bezier points as 10 vertices (via layout( vertices = 10 ) out;) is substantially slower!
  // Note: copying positions and normals seems to be free
  vec3 positions[3], normals[3], bezierPoints[10];
  for ( int i=0 ; i<3 ; i++ )
  {
    positions[i] = tcPosition[i];
    normals[i] = tcNormal[i];
  }
  createBezierFromPNTriangle( positions, normals, bezierPoints );

  vec3 normal;
  vec3 position = evalBezierTriangle( bezierPoints, gl_TessCoord, normal );

  vec4 worldPos = sys_WorldMatrix * vec4( position, 1.0f );
  varNormal     = ( sys_WorldMatrixIT * vec4( normal, 0.0f ) ).xyz;
  varWorldPos   = worldPos.xyz;
  varEyePos     = vec3( sys_ViewMatrixI[3][0], sys_ViewMatrixI[3][1], sys_ViewMatrixI[3][2] );
  gl_Position   = sys_ViewProjMatrix * worldPos;

  varTexCoord0  = gl_TessCoord.x * tcTexCoord0[0] + gl_TessCoord.y * tcTexCoord0[1] + gl_TessCoord.z * tcTexCoord0[2];
}
