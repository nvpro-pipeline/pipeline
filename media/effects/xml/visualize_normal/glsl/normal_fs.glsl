
void main(void)
{
  vec3 ns = normalize(varNormal);
  Color = vec4(ns * 0.5 + 0.5, 1.0);
}
