// Phong lighting
vec3 eval(in vec3 wo, in vec3 ns, in vec3 wi, 
          in vec3 ambientColor, in vec3 diffuseColor, in vec3 specularColor,
          in vec3 ambient, in vec3 diffuse, in vec3 specular, in float exponent)
{
  float shine = 0.0;
  float ns_dot_wi = max(0.0, dot(ns, wi));
  if (0.0f < ns_dot_wi)
  {
    // Phong
    vec3 R = reflect(-wi, ns);
    float r_dot_wo = max(0.0, dot(R, wo));
    shine = (0.0 < exponent) ? pow(r_dot_wo, exponent) : 1.0;
  }
  return ambient * ambientColor +
         ns_dot_wi * diffuse * diffuseColor +
         shine * specular * specularColor;
}

void main(void)
{
  vec3 wo = normalize(varEyePos - varWorldPos);
  vec3 ns = normalize(varNormal);
  
  vec3 rgb = emissiveColor +
             ambientColor * sys_SceneAmbientLight;
             
  if ( gl_FrontFacing )
  {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 wi;
    for (int i = 0; i < sys_NumLights; ++i)
    {
      sampleLight(sys_Lights[i], varWorldPos, wi, ambient, diffuse, specular);
      rgb += eval(wo, ns, wi, 
                 ambientColor, diffuseColor, specularColor, 
                 ambient, diffuse, specular, specularExponent);
    }
  }
  Color = vec4( rgb, 1.0 );
}
