
void main(void)
{
  stateNormal = normalize( attrNormal );
  evalTemporaries(stateNormal);
  worldPos = evalWorldPos();
  gl_Position = sys_ViewProjMatrix * worldPos;
}
