
layout( vertices = 3 ) out;

void main(void)
{
  tcPosition[gl_InvocationID] = vPosition[gl_InvocationID];
  tcNormal[gl_InvocationID] = vNormal[gl_InvocationID];

  if ( gl_InvocationID == 0 )
  {
    calculateTessLevels();
  }
}
