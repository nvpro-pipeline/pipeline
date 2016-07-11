
void main(void)
{
  stateNormal = normalize( attrNormal );
  texCoord0   = attrTexCoord0;
  tangent     = normalize( attrTangent );
  binormal    = normalize( attrBinormal );

  evalTemporaries(stateNormal);
  worldPos = evalWorldPos();
  evalVaryings();
  gl_Position   = sys_ViewProjMatrix * worldPos;
}
