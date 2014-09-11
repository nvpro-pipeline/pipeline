
float calculateTessFactor( vec2 p0, vec2 p1 )
{
  return( max( distance( p0, p1 ) / tessellationAccuracy, 1.0f ) );
}

void calculateTessLevels()
{
  mat4 mvp = sys_ViewProjMatrix * sys_WorldMatrix;

  vec2 screenPosition[3];
  for ( int i=0 ; i<3 ; i++ )
  {
    vec4 p = mvp * vec4( vPosition[i], 1.0f );
    p /= p.w;
    screenPosition[i] = 0.5f * ( p.xy * sys_ViewportSize + sys_ViewportSize );
  }

  gl_TessLevelOuter[0] = calculateTessFactor( screenPosition[1], screenPosition[2] );
  gl_TessLevelOuter[1] = calculateTessFactor( screenPosition[0], screenPosition[2] );
  gl_TessLevelOuter[2] = calculateTessFactor( screenPosition[0], screenPosition[1] );
  gl_TessLevelOuter[3] = 0.0f;

  gl_TessLevelInner[0] = max( max( gl_TessLevelOuter[0], gl_TessLevelOuter[1] ), gl_TessLevelOuter[2] );
  gl_TessLevelInner[1] = 0.0f;
}

