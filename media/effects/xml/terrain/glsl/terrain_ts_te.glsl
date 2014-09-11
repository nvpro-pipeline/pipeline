// triangles, quads, or isolines
layout (quads, equal_spacing, ccw) in;
 
// could use a displacement map here
 
uniform mat4 viewmat;
uniform mat4 projmat;

out gl_PerVertex
{
  vec4 gl_Position;
};

out vec3 varNormal;
out vec3 varWorldPos;
out vec3 varEyePos;
out vec2 varTexCoord0;

in ivec2 tc_gridPosition[];
in float te_tessLevelMax[];
 
out vec3 color;
 
void emit( ivec2 position )
{
  ivec2 heightMapSize = textureSize( heightMap, 0 );

  float height = texelFetch( heightMap, position, 0).x;

  // calculate position  
  vec3 gridPositionLocal = vec3( float( position.x ), float( position.y ), height ) * resolution + offset;
  gridPositionLocal.z += position.x;
  
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
  
  vec4 worldPos = sys_WorldMatrix * vec4(gridPositionLocal, 1.0);
  varNormal    = ( sys_WorldMatrixIT * vec4( normal, 0.0 ) ).xyz;
  varWorldPos  = worldPos.xyz;
  varEyePos    = vec3( sys_ViewMatrixI[3][0], sys_ViewMatrixI[3][1], sys_ViewMatrixI[3][2] );
  gl_Position   = sys_ViewProjMatrix * worldPos;
}
 
 
void main () {
  float height = 0;
  ivec2 position = tc_gridPosition[0] + ivec2(round(gl_TessCoord.xy * te_tessLevelMax[0]));
 
  emit( position );
}
