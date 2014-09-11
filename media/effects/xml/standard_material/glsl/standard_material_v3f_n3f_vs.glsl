void main(void)
{
  vec4 worldPos = sys_WorldMatrix * attrPosition;
  varNormal     = ( sys_WorldMatrixIT * vec4( attrNormal, 0.0 ) ).xyz;
  varWorldPos   = worldPos.xyz;
  varEyePos     = vec3( sys_ViewMatrixI[3][0], sys_ViewMatrixI[3][1], sys_ViewMatrixI[3][2] );
  gl_Position   = sys_ViewProjMatrix * worldPos;
}
