void main(void)
{
  vec4 worldPos = sys_WorldMatrix * attrPosition;
  varNormal     = ( sys_WorldMatrixIT * vec4( attrNormal, 0.0 ) ).xyz;
  varWorldPos   = worldPos.xyz;
  varEyePos     = vec3( sys_ViewMatrixI[3][0], sys_ViewMatrixI[3][1], sys_ViewMatrixI[3][2] );
  gl_Position   = sys_ViewProjMatrix * worldPos;
  varTexCoord0  = attrTexCoord0;
  varTangent    = (sys_WorldMatrix * vec4(attrTangent,0)).xyz;
  varBinormal   = (sys_WorldMatrix * vec4(attrBinormal,0)).xyz;
}
