void main(void)
{
  varTexCoord0  = attrTexCoord0;
  gl_Position   = sys_ViewProjMatrix * sys_WorldMatrix * attrPosition;
}
