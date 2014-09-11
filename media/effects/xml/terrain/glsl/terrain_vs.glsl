out gl_PerVertex
{
  vec4 gl_Position;
};

void main(void)
{
  ivec2 heightMapSize = textureSize( heightMap, 0 );
  
  // determine quad first
  int baseVertex = gl_VertexID / 6;
  int localVertex = gl_VertexID - (baseVertex * 6);
  
  // determine position in grid
  
  ivec2 position;
  position.y = baseVertex / ( heightMapSize.x - 1 );
  position.x = baseVertex - (position.y * ( heightMapSize.x - 1 ) );
  
  if ( localVertex == 0)
    position += ivec2(0,0);
  else if ( localVertex == 1)
    position += ivec2(1,0);
  else if ( localVertex == 2)
    position += ivec2(1,1);
  else if ( localVertex == 3)
    position += ivec2(1,1);
  else if ( localVertex == 4)
    position += ivec2(0,1);
  else if ( localVertex == 5)
    position += ivec2(0,0);
    
  float height = texelFetch( heightMap, position, 0).x;

  // calculate position  
  vec3 gridPosition = vec3( float( position.x ), float( position.y ), height ) * resolution + offset;
  
  // calculate normal vector
  int left = position.x > 0 ? position.x - 1 : position.x;
  int right = position.x < (heightMapSize.x - 1) ? position.x + 1 : position.x;
  vec3 dx = vec3( 2.0 * resolution.x, 0.0, ( texelFetch( heightMap, ivec2( right, position.y ), 0 ).x - texelFetch( heightMap, ivec2( left, position.y ), 0 ).x ) * resolution.z );
  
  int up = position.y > 0 ? position.y - 1 : position.y;
  int down = position.y < (heightMapSize.y - 1) ? position.y + 1 : position.y;
  vec3 dy = vec3( 0.0, 2.0 * resolution.y, ( texelFetch( heightMap, ivec2( position.x, down), 0 ).x - texelFetch( heightMap, ivec2( position.x, up ), 0).x ) * resolution.z );
  
  vec3 normal = normalize( cross( dx, dy ) );
  
  // calculate texture coordinates
  varTexCoord0  = vec2(position.xy) / vec2( heightMapSize.xy );
  
  vec4 worldPos = sys_WorldMatrix * vec4(gridPosition, 1.0);
  varNormal     = ( sys_WorldMatrixIT * vec4( normal, 0.0 ) ).xyz;
  varWorldPos   = worldPos.xyz;
  varEyePos     = vec3( sys_ViewMatrixI[3][0], sys_ViewMatrixI[3][1], sys_ViewMatrixI[3][2] );
  gl_Position   = sys_ViewProjMatrix * worldPos;
}
